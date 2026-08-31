import os
import pickle
import tempfile
import unittest

import numpy as np

from src.artifacts import POSITION_SCHEMA_VERSION, TRAINING_EXAMPLE_SCHEMA_VERSION
from src.game.pente.pente_board import PenteBoard
from src.train.replay_buffer import ReplayBuffer, ReplaySnapshot, ReplaySource
from src.train.training_example import TrainingExample


class ReplayBufferTest(unittest.TestCase):
    def test_capacity_age_and_unique_position_metrics(self) -> None:
        replay = ReplayBuffer(3)
        position = PenteBoard.new_board(5)
        policy = np.full(25, 1.0 / 25, dtype=np.float32)
        first = TrainingExample(position, policy, 0.0)
        moved = position.apply_move(position.current_player, (0, 0))
        moved_policy = np.full(25, 1.0 / 24, dtype=np.float32)
        moved_policy[0] = 0.0
        second = TrainingExample(moved, moved_policy, 0.0)

        replay.extend(
            (first, second),
            generation=1,
            source=ReplaySource.PROFESSIONAL,
        )
        replay.extend(
            (first, second),
            generation=2,
            source=ReplaySource.SELF_PLAY,
        )
        stats = replay.stats(current_generation=3)

        self.assertEqual(3, len(replay))
        self.assertEqual(2, stats.oldest_age)
        self.assertEqual(1, stats.newest_age)
        self.assertEqual(2, stats.unique_positions)
        self.assertEqual(1, stats.professional_positions)
        self.assertEqual(2, stats.self_play_positions)

    def test_snapshot_round_trip(self) -> None:
        replay = ReplayBuffer(4)
        position = PenteBoard.new_board(5)
        replay.extend(
            (TrainingExample(position, np.full(25, 1.0 / 25, dtype=np.float32), 0.0),),
            generation=2,
            source=ReplaySource.SELF_PLAY,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "replay.pkl")
            replay.save(
                path,
                generation=3,
                training_run_id="test-run",
                board_size=5,
                ruleset="freestyle",
            )
            loaded = ReplayBuffer.load(
                path,
                expected_training_run_id="test-run",
                expected_board_size=5,
                expected_ruleset="freestyle",
            )

        self.assertEqual(4, loaded.capacity)
        self.assertEqual(1, len(loaded))
        self.assertEqual(3, loaded.snapshot_generation)
        self.assertEqual(position.state_key(), loaded.examples()[0].position.state_key())

    def test_sampling_is_bounded_and_seeded(self) -> None:
        replay = ReplayBuffer(4)
        position = PenteBoard.new_board(5)
        examples = []
        for action in range(4):
            policy = np.zeros(25, dtype=np.float32)
            policy[action] = 1.0
            examples.append(TrainingExample(position, policy, 0.0))
        replay.extend(
            examples,
            generation=1,
            source=ReplaySource.SELF_PLAY,
        )

        first = replay.sample(2, np.random.default_rng(4))
        second = replay.sample(2, np.random.default_rng(4))

        self.assertEqual(
            [example.policy.argmax() for example in first],
            [example.policy.argmax() for example in second],
        )
        self.assertEqual(2, len(first))

        replacement_sample = replay.sample(6, np.random.default_rng(5))
        self.assertEqual(6, len(replacement_sample))

    def test_mixed_sampling_enforces_requested_source_fraction(self) -> None:
        replay = ReplayBuffer(8)
        position = PenteBoard.new_board(5)
        policy = np.full(25, 1.0 / 25, dtype=np.float32)
        example = TrainingExample(position, policy, 0.0)
        replay.extend(
            [example],
            generation=1,
            source=ReplaySource.PROFESSIONAL,
        )
        replay.extend(
            [example],
            generation=2,
            source=ReplaySource.SELF_PLAY,
        )

        sample = replay.sample_mixed(
            20,
            np.random.default_rng(6),
            professional_fraction=0.25,
        )

        self.assertEqual(20, len(sample.examples))
        self.assertEqual(5, sample.professional_positions)
        self.assertEqual(15, sample.self_play_positions)

    def test_rejects_unknown_snapshot_schema(self) -> None:
        snapshot = ReplaySnapshot(
            schema_version=999,
            capacity=4,
            generation=1,
            training_run_id="test-run",
            board_size=5,
            ruleset="freestyle",
            position_schema_version=POSITION_SCHEMA_VERSION,
            training_example_schema_version=TRAINING_EXAMPLE_SCHEMA_VERSION,
            entries=(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "replay.pkl")
            with open(path, "wb") as stream:
                pickle.dump(snapshot, stream)

            with self.assertRaisesRegex(ValueError, "incompatible"):
                ReplayBuffer.load(
                    path,
                    expected_training_run_id="test-run",
                    expected_board_size=5,
                    expected_ruleset="freestyle",
                )

    def test_rejects_replay_from_another_ruleset(self) -> None:
        replay = ReplayBuffer(4)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "replay.pkl")
            replay.save(
                path,
                generation=1,
                training_run_id="test-run",
                board_size=5,
                ruleset="standard",
            )

            with self.assertRaisesRegex(ValueError, "ruleset"):
                ReplayBuffer.load(
                    path,
                    expected_training_run_id="test-run",
                    expected_board_size=5,
                    expected_ruleset="freestyle",
                )

    def test_rejects_replay_from_another_training_run(self) -> None:
        replay = ReplayBuffer(4)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "replay.pkl")
            replay.save(
                path,
                generation=1,
                training_run_id="first-run",
                board_size=5,
                ruleset="freestyle",
            )

            with self.assertRaisesRegex(ValueError, "different training run"):
                ReplayBuffer.load(
                    path,
                    expected_training_run_id="second-run",
                    expected_board_size=5,
                    expected_ruleset="freestyle",
                )


if __name__ == "__main__":
    unittest.main()
