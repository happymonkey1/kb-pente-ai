import os
import pickle
import tempfile
import unittest

import numpy as np

from src.game.game import Game
from src.game.pente.rules import PenteRuleset
from src.train.profession_game_loader import ProfessionGameLoader


class ProfessionGameLoaderTest(unittest.TestCase):
    VALID_GAME = "C3;A1;B3;B1;A3;C1;D3;D1;E3;1-0;\n"
    INVALID_GAME = "C3;A1;C3;1-0;\n"
    MOVE_AFTER_WIN = "C3;A1;B3;B1;A3;C1;D3;D1;E3;E1;1-0;\n"

    def test_stores_legal_pre_move_states_and_relative_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = os.path.join(directory, "games.txt")
            processed_path = os.path.join(directory, "games.pkl")
            with open(raw_path, "w", encoding="utf-8") as stream:
                stream.write(self.VALID_GAME)
            loader = ProfessionGameLoader(
                raw_path,
                processed_path,
                board_size=5,
                force=True,
                ruleset=PenteRuleset.STANDARD,
                validation_fraction=0.0,
            )

            examples = loader.load_games()

            self.assertEqual(9, len(examples))
            first = examples[0]
            first_action = int(np.argmax(first.policy))
            self.assertEqual(12, first_action)
            self.assertEqual(0, first.position.board.reshape(-1)[first_action])
            self.assertEqual(Game.PLAYER_ONE, first.position.current_player)
            self.assertEqual(1.0, first.value)
            second = next(example for example in examples if example.position.ply == 1)
            self.assertEqual(Game.PLAYER_TWO, second.position.current_player)
            self.assertEqual(-1.0, second.value)
            for example in examples:
                action = int(np.argmax(example.policy))
                self.assertEqual(0, example.position.board.reshape(-1)[action])

            cached_loader = ProfessionGameLoader(
                raw_path,
                processed_path,
                board_size=5,
                ruleset=PenteRuleset.STANDARD,
                validation_fraction=0.0,
            )
            cached = cached_loader.load_games()
            self.assertEqual(
                [example.position.state_key() for example in examples],
                [example.position.state_key() for example in cached],
            )
            self.assertFalse(cached[0].position.board.flags.writeable)
            self.assertFalse(cached[0].policy.flags.writeable)

    def test_rejects_entire_illegal_game_without_partial_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = os.path.join(directory, "games.txt")
            processed_path = os.path.join(directory, "games.pkl")
            with open(raw_path, "w", encoding="utf-8") as stream:
                stream.write(self.VALID_GAME)
                stream.write(self.INVALID_GAME)
            loader = ProfessionGameLoader(
                raw_path,
                processed_path,
                board_size=5,
                force=True,
                ruleset=PenteRuleset.STANDARD,
                validation_fraction=0.0,
            )

            examples = loader.load_games()

            self.assertEqual(9, len(examples))
            self.assertIsNotNone(loader.last_stats)
            assert loader.last_stats is not None
            self.assertEqual(1, loader.last_stats.accepted_games)
            self.assertEqual(1, loader.last_stats.rejected_games)
            self.assertEqual({"illegal_move": 1}, loader.last_stats.rejection_reasons)

    def test_rejects_legacy_processed_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = os.path.join(directory, "games.txt")
            processed_path = os.path.join(directory, "games.pkl")
            with open(raw_path, "w", encoding="utf-8") as stream:
                stream.write(self.VALID_GAME)
            with open(processed_path, "wb") as stream:
                pickle.dump([], stream)
            loader = ProfessionGameLoader(
                raw_path,
                processed_path,
                board_size=5,
                ruleset=PenteRuleset.STANDARD,
                validation_fraction=0.0,
            )

            with self.assertRaisesRegex(ValueError, "legacy format"):
                loader.load_games()

    def test_rejects_moves_after_terminal_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = os.path.join(directory, "games.txt")
            processed_path = os.path.join(directory, "games.pkl")
            with open(raw_path, "w", encoding="utf-8") as stream:
                stream.write(self.MOVE_AFTER_WIN)
            loader = ProfessionGameLoader(
                raw_path,
                processed_path,
                board_size=5,
                force=True,
                ruleset=PenteRuleset.STANDARD,
                validation_fraction=0.0,
            )

            examples = loader.load_games()

            self.assertEqual([], examples)
            assert loader.last_stats is not None
            self.assertEqual({"move_after_terminal": 1}, loader.last_stats.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
