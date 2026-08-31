"""Reusable model-backed search quality verification.

The API accepts already-loaded models and deliberately leaves checkpoint
loading and native-extension setup to its caller. A bounded invocation can
therefore be run independently for each selected search backend.
"""

from __future__ import annotations

from collections.abc import Iterable
import operator
import time
from typing import Literal, Protocol, SupportsIndex, cast

import numpy as np

from src.evaluation.statistics import elo_difference, wilson_interval
from src.evaluation.tactical import (
    TacticalCase,
    build_tactical_cases,
    evaluate_tactical_suite,
)
from src.game.pente.pente_game import PenteGame
from src.mcts.mcts_v2 import MCTSArgs
from src.model.model_v1 import PenteNet
from src.train.arena import Arena, ArenaStats
from src.train.player import Player
from src.train.player_builder import build_player
from src.train.random_player import RandomPlayer
from src.train.self_play_args import SearchBackend
from src.train.tactical_heuristic_player import TacticalHeuristicPlayer
from src.verification.search_quality_reports import (
    BackendSearchQualityReport,
    ColorResults,
    DirectTacticalMetrics,
    GameResults,
    MatchParity,
    MatchReport,
    PairResults,
    SearchQualityConfig,
    SearchTacticalCaseResult,
    SearchTacticalReport,
    StatisticalParityCriteria,
    StatisticalParityReport,
    WilsonInterval,
)


class _EvaluableModel(Protocol):
    def eval(self) -> object:
        """Switch the model to evaluation mode."""


def evaluate_search_quality(
    current: PenteNet,
    previous: PenteNet,
    game: PenteGame,
    mcts_args: MCTSArgs,
    games: int,
    opening_plies: int = 4,
    seed: int = 0,
    search_backend: SearchBackend = "python",
    native_worker_threads: int = 1,
) -> BackendSearchQualityReport:
    """Evaluate one backend against the previous, random, and heuristic players.

    The current model is Arena Player 1 in every match. The accepted
    model-match seed offsets are retained for current and previous models;
    random, heuristic, and tactical players use distinct deterministic
    offsets. C++ is forwarded through ``build_player`` without fallback or
    preflight.
    """

    _validate_inputs(
        current,
        previous,
        game,
        mcts_args,
        games,
        opening_plies,
        seed,
        search_backend,
        native_worker_threads,
    )
    current.eval()
    previous.eval()
    config = SearchQualityConfig(
        games=games,
        opening_plies=opening_plies,
        seed=seed,
        mcts_args=mcts_args,
        native_worker_threads=native_worker_threads,
        board_size=game.get_board_size(),
        ruleset=game.ruleset.value,
    )
    tactical_cases = build_tactical_cases(game.get_board_size())
    started = time.perf_counter()

    current_direct = DirectTacticalMetrics.from_stats(
        evaluate_tactical_suite(current, game),
    )
    previous_direct = DirectTacticalMetrics.from_stats(
        evaluate_tactical_suite(previous, game),
    )

    current_previous = _evaluate_match(
        _build_search_player(
            current,
            "current",
            game,
            mcts_args,
            search_backend,
            seed + 1,
            native_worker_threads,
        ),
        _build_search_player(
            previous,
            "previous",
            game,
            mcts_args,
            search_backend,
            seed + 2,
            native_worker_threads,
        ),
        game,
        "previous",
        games,
        opening_plies,
        seed,
    )
    current_random = _evaluate_match(
        _build_search_player(
            current,
            "current",
            game,
            mcts_args,
            search_backend,
            seed + 3,
            native_worker_threads,
        ),
        RandomPlayer(np.random.default_rng(seed + 4)),
        game,
        "random",
        games,
        opening_plies,
        seed + 11,
    )
    current_heuristic = _evaluate_match(
        _build_search_player(
            current,
            "current",
            game,
            mcts_args,
            search_backend,
            seed + 5,
            native_worker_threads,
        ),
        TacticalHeuristicPlayer(),
        game,
        "heuristic",
        games,
        opening_plies,
        seed + 12,
    )
    search_tactical = _evaluate_search_tactical(
        current,
        game,
        tactical_cases,
        search_backend,
        mcts_args,
        seed + 6,
        native_worker_threads,
    )
    elapsed_seconds = time.perf_counter() - started
    if elapsed_seconds < 0:
        raise ValueError("Elapsed time cannot be negative")
    return BackendSearchQualityReport(
        backend=search_backend,
        config=config,
        current_vs_previous=current_previous,
        current_vs_random=current_random,
        current_vs_heuristic=current_heuristic,
        current_direct_tactical=current_direct,
        previous_direct_tactical=previous_direct,
        search_tactical=search_tactical,
        elapsed_seconds=elapsed_seconds,
    )


def summarize_statistical_parity(
    left: BackendSearchQualityReport,
    right: BackendSearchQualityReport,
    criteria: StatisticalParityCriteria | None = None,
) -> StatisticalParityReport:
    """Compare backend rates without requiring identical trajectories or counts."""

    if left.backend == right.backend:
        raise ValueError("Statistical parity requires two distinct backends")
    if left.config != right.config:
        raise ValueError("Statistical parity requires matching configurations")
    selected = criteria or StatisticalParityCriteria()
    matches = tuple(
        _match_parity(getattr(left, name), getattr(right, name))
        for name in (
            "current_vs_previous",
            "current_vs_random",
            "current_vs_heuristic",
        )
    )
    tactical_accuracy_difference = abs(
        left.search_tactical.accuracy - right.search_tactical.accuracy,
    )
    failures: list[str] = []
    for result in matches:
        if result.decisive_win_rate_difference > selected.maximum_decisive_win_rate_difference:
            failures.append(
                f"{result.opponent} decisive win-rate difference exceeds tolerance",
            )
        if result.paired_win_rate_difference > selected.maximum_paired_win_rate_difference:
            failures.append(
                f"{result.opponent} paired win-rate difference exceeds tolerance",
            )
        if selected.require_wilson_overlap and not result.decisive_interval_overlaps:
            failures.append(f"{result.opponent} decisive Wilson intervals do not overlap")
        if selected.require_wilson_overlap and not result.paired_interval_overlaps:
            failures.append(f"{result.opponent} paired Wilson intervals do not overlap")
    if tactical_accuracy_difference > selected.maximum_tactical_accuracy_difference:
        failures.append("search tactical accuracy difference exceeds tolerance")
    return StatisticalParityReport(
        left_backend=left.backend,
        right_backend=right.backend,
        matches=matches,
        tactical_accuracy_difference=tactical_accuracy_difference,
        passed=not failures,
        failures=tuple(failures),
    )


def _build_search_player(
    model: PenteNet,
    name: str,
    game: PenteGame,
    mcts_args: MCTSArgs,
    search_backend: SearchBackend,
    seed: int,
    native_worker_threads: int,
) -> Player:
    return build_player(
        model,
        None,
        name,
        search_backend=search_backend,
        game=game,
        mcts_args=mcts_args,
        seed=seed,
        native_worker_threads=native_worker_threads,
    )


def _evaluate_match(
    current: Player,
    opponent: Player,
    game: PenteGame,
    opponent_name: Literal["previous", "random", "heuristic"],
    games: int,
    opening_plies: int,
    opening_seed: int,
) -> MatchReport:
    arena = Arena(
        current,
        opponent,
        game,
        opening_plies=opening_plies,
        rng=np.random.default_rng(opening_seed),
    )
    started = time.perf_counter()
    stats = arena.play_games(games)
    return _summarize_match(opponent_name, stats, time.perf_counter() - started)


def _summarize_match(
    opponent: Literal["previous", "random", "heuristic"],
    stats: ArenaStats,
    elapsed_seconds: float,
) -> MatchReport:
    if elapsed_seconds < 0:
        raise ValueError("Elapsed time cannot be negative")
    games = stats.p1_wins + stats.p2_wins + stats.draws
    decisive_games = stats.p1_wins + stats.p2_wins
    decisive_rate = stats.p1_wins / decisive_games if decisive_games else 0.0
    decisive_pairs = stats.p1_pair_wins + stats.p1_pair_losses
    paired_rate = stats.p1_pair_wins / decisive_pairs if decisive_pairs else 0.0
    score = (stats.p1_wins + 0.5 * stats.draws) / games if games else 0.0
    return MatchReport(
        opponent=opponent,
        game=GameResults(
            games=games,
            current_wins=stats.p1_wins,
            opponent_wins=stats.p2_wins,
            draws=stats.draws,
            decisive_games=decisive_games,
            current_decisive_win_rate=decisive_rate,
            current_decisive_win_rate_95pct=WilsonInterval(
                *wilson_interval(stats.p1_wins, decisive_games),
            ),
            current_score=score,
            current_elo=elo_difference(score),
        ),
        color=ColorResults(
            current_as_player_one_wins=stats.p1_as_player_one_wins,
            current_as_player_two_wins=stats.p1_as_player_two_wins,
            player_one_color_wins=stats.player_one_color_wins,
            player_two_color_wins=stats.player_two_color_wins,
        ),
        pair=PairResults(
            opening_plies=stats.opening_plies,
            unique_openings=stats.unique_openings,
            paired_openings=stats.paired_openings,
            current_pair_wins=stats.p1_pair_wins,
            current_pair_losses=stats.p1_pair_losses,
            pair_ties=stats.pair_ties,
            current_paired_win_rate=paired_rate,
            current_paired_win_rate_95pct=WilsonInterval(
                *wilson_interval(stats.p1_pair_wins, decisive_pairs),
            ),
        ),
        average_moves=stats.avg_moves,
        elapsed_seconds=elapsed_seconds,
        moves_per_second=(
            games * stats.avg_moves / elapsed_seconds if elapsed_seconds else 0.0
        ),
    )


def _evaluate_search_tactical(
    current: PenteNet,
    game: PenteGame,
    cases: Iterable[TacticalCase],
    search_backend: SearchBackend,
    mcts_args: MCTSArgs,
    seed: int,
    native_worker_threads: int,
) -> SearchTacticalReport:
    player = _build_search_player(
        current,
        "current-tactical",
        game,
        mcts_args,
        search_backend,
        seed,
        native_worker_threads,
    )
    results: list[SearchTacticalCaseResult] = []
    for case in cases:
        player.reset()
        selected_action = int(
            player.play(game, case.position, case.position.current_player),
        )
        legal = game.get_valid_moves(case.position, case.position.current_player)
        if not 0 <= selected_action < game.get_action_size() or not legal[selected_action]:
            raise ValueError(
                f"Search player selected illegal tactical action {selected_action}",
            )
        expected_actions = tuple(case.expected_actions)
        results.append(
            SearchTacticalCaseResult(
                name=case.name,
                category=case.category,
                selected_action=selected_action,
                expected_actions=expected_actions,
                correct=selected_action in expected_actions,
            ),
        )
    selected_cases = tuple(results)
    correct = sum(case.correct for case in selected_cases)
    return SearchTacticalReport(
        cases=selected_cases,
        correct=correct,
        accuracy=correct / len(selected_cases) if selected_cases else 0.0,
    )


def _match_parity(left: MatchReport, right: MatchReport) -> MatchParity:
    if left.opponent != right.opponent:
        raise ValueError("Cannot compare different opponents")
    return MatchParity(
        opponent=left.opponent,
        decisive_win_rate_difference=abs(
            left.game.current_decisive_win_rate
            - right.game.current_decisive_win_rate,
        ),
        paired_win_rate_difference=abs(
            left.pair.current_paired_win_rate - right.pair.current_paired_win_rate,
        ),
        decisive_interval_overlaps=_intervals_overlap(
            left.game.current_decisive_win_rate_95pct,
            right.game.current_decisive_win_rate_95pct,
        ),
        paired_interval_overlaps=_intervals_overlap(
            left.pair.current_paired_win_rate_95pct,
            right.pair.current_paired_win_rate_95pct,
        ),
    )


def _intervals_overlap(left: WilsonInterval, right: WilsonInterval) -> bool:
    return max(left.lower, right.lower) <= min(left.upper, right.upper)


def _validate_inputs(
    current: _EvaluableModel,
    previous: _EvaluableModel,
    game: PenteGame,
    mcts_args: MCTSArgs,
    games: int,
    opening_plies: int,
    seed: int,
    search_backend: SearchBackend,
    native_worker_threads: int,
) -> None:
    for name, model in (("current", current), ("previous", previous)):
        if model is None or not callable(getattr(model, "eval", None)):
            raise TypeError(f"{name} must provide eval()")
    if not isinstance(game, PenteGame):
        raise TypeError("game must be a PenteGame instance")
    if not isinstance(mcts_args, MCTSArgs):
        raise TypeError("mcts_args must be an MCTSArgs instance")
    if mcts_args.num_simulations < 1:
        raise ValueError("mcts_args.num_simulations must be positive")
    if game.get_board_size() < 9:
        raise ValueError("Search quality requires a board of at least 9 by 9")
    _require_positive_int(games, "games")
    if games < 2:
        raise ValueError("games must be at least two")
    if games % 2:
        raise ValueError("games must be even for paired openings")
    _require_nonnegative_int(opening_plies, "opening_plies")
    _require_nonnegative_int(seed, "seed")
    if search_backend not in ("python", "cpp"):
        raise ValueError("Search backend must be 'python' or 'cpp'")
    _require_positive_int(native_worker_threads, "native_worker_threads")
    _validate_model_compatibility(current, game, "current")
    _validate_model_compatibility(previous, game, "previous")
    _validate_matching_model_configurations(current, previous)


def _validate_model_compatibility(model: object, game: PenteGame, name: str) -> None:
    config = getattr(model, "config", None)
    if config is None:
        return
    board_size = getattr(config, "board_size", None)
    action_size = getattr(config, "action_size", None)
    if board_size is not None and board_size != game.get_board_size():
        raise ValueError(f"{name} board size does not match game")
    if action_size is not None and action_size != game.get_action_size():
        raise ValueError(f"{name} action size does not match game")


def _validate_matching_model_configurations(current: object, previous: object) -> None:
    current_config = getattr(current, "config", None)
    previous_config = getattr(previous, "config", None)
    if current_config is None or previous_config is None:
        return
    for field in (
        "board_size",
        "action_size",
        "input_planes",
        "num_res_blocks",
        "num_channels",
        "hidden_fc_size",
    ):
        current_value = getattr(current_config, field, None)
        previous_value = getattr(previous_config, field, None)
        if current_value is not None and previous_value is not None and current_value != previous_value:
            raise ValueError(f"current and previous model configurations differ in {field}")


def _require_positive_int(value: object, name: str) -> int:
    parsed = _require_nonnegative_int(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        parsed = int(operator.index(cast(SupportsIndex, value)))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < 0:
        raise ValueError(f"{name} cannot be negative")
    return parsed
