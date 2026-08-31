import unittest

import numpy as np

from src.game.game import TerminalResult
from src.game.pente.pente_board import PenteBoard
from src.monitoring.models import ReplaySample
from src.train.replay_samples import emit_replay_samples
from src.train.self_play_generation import PlayedGame, finalize_training_examples


class RecordingReplaySink:
    def __init__(self) -> None:
        self.samples: list[ReplaySample] = []

    def emit(self, sample: ReplaySample) -> None:
        self.samples.append(sample)


class ReplaySamplesTest(unittest.TestCase):
    def test_emits_bounded_stable_safe_game_samples(self) -> None:
        games = [self._played_game(action) for action in range(12)]
        first = RecordingReplaySink()
        second = RecordingReplaySink()

        first_count = emit_replay_samples(
            first,
            games,
            "run-id",
            3,
            5,
            "freestyle",
        )
        second_count = emit_replay_samples(
            second,
            list(reversed(games)),
            "run-id",
            3,
            5,
            "freestyle",
        )

        self.assertEqual(8, first_count)
        self.assertEqual(8, second_count)
        self.assertEqual(
            [sample.actions for sample in first.samples],
            [sample.actions for sample in second.samples],
        )
        self.assertEqual(
            [sample.game_id for sample in first.samples],
            [sample.game_id for sample in second.samples],
        )
        self.assertTrue(all(sample.run_id == "run-id" for sample in first.samples))
        self.assertTrue(all(sample.board_size == 5 for sample in first.samples))

    def test_can_disable_samples(self) -> None:
        sink = RecordingReplaySink()

        count = emit_replay_samples(
            sink,
            [self._played_game(0)],
            "run-id",
            1,
            5,
            "freestyle",
            maximum_samples=0,
        )

        self.assertEqual(0, count)
        self.assertEqual([], sink.samples)

    @staticmethod
    def _played_game(action: int) -> PlayedGame:
        position = PenteBoard.new_board(5)
        policy = np.zeros(25, dtype=np.float32)
        policy[action] = 1.0
        result = TerminalResult.win(1, "line")
        return PlayedGame(
            examples=finalize_training_examples([(position, policy)], result),
            actions=(action,),
            winner=1,
            win_reason="line",
            root_telemetry=(),
        )


if __name__ == "__main__":
    unittest.main()
