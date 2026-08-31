import subprocess
import sys
import threading
import time
import unittest

import torch


def _initial_root(board_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((board_size, board_size), dtype=torch.int8),
        torch.zeros(2, dtype=torch.int16),
    )


class SearchBatchBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.extension = __import__("kb_pente_native")

    def make_batch(self, board_size: int = 5, simulations: int = 2):
        return self.extension.SearchBatch(
            board_size=board_size,
            ruleset="freestyle",
            simulations=simulations,
            active_games=4,
            threads=2,
            seed=103,
            pin_memory=False,
        )

    def test_staging_shapes_cpu_unpinned_and_shared_feature_storage(self) -> None:
        batch = self.make_batch()
        self.assertEqual(tuple(batch.features.shape), (4, 4, 5, 5))
        self.assertEqual(tuple(batch.policies.shape), (4, 361))
        self.assertEqual(tuple(batch.values.shape), (4,))
        self.assertFalse(batch.pin_memory_requested)
        self.assertFalse(batch.pin_memory_realized)

        stones, captures = _initial_root(5)
        batch.add(stones, captures, 1, 0)
        selected = batch.select()
        self.assertEqual(selected.size, 1)
        self.assertEqual(selected.raw_size, 1)
        self.assertEqual(
            selected.features.data_ptr(), batch.features.data_ptr()
        )
        self.assertEqual(tuple(selected.features.shape), (1, 4, 5, 5))

    def test_duplicate_roots_and_policy_completion(self) -> None:
        batch = self.make_batch(simulations=1)
        stones, captures = _initial_root(5)
        batch.add(stones, captures, 1, 0, temperature=0.0)
        batch.add(stones.clone(), captures.clone(), 1, 0, temperature=0.0)
        selected = batch.select()
        self.assertEqual(selected.size, 1)
        self.assertEqual(selected.raw_size, 2)

        batch.policies[0].zero_()
        batch.policies[0, 0] = 1.0
        batch.values[0] = 0.25
        batch.backup(selected.token, selected.size)
        self.assertTrue(batch.complete())
        policy = batch.root_policy(0)
        self.assertEqual(tuple(policy.shape), (25,))
        self.assertAlmostEqual(float(policy.sum()), 1.0)

        repeat = self.make_batch(simulations=1)
        repeat.add(stones, captures, 1, 0, temperature=0.0)
        repeat.add(stones.clone(), captures.clone(), 1, 0, temperature=0.0)
        repeat_selection = repeat.select()
        repeat.policies[0].zero_()
        repeat.policies[0, 0] = 1.0
        repeat.values[0] = 0.25
        repeat.backup(repeat_selection.token, repeat_selection.size)
        self.assertTrue(torch.equal(policy, repeat.root_policy(0)))

    def test_rejected_backup_is_retryable(self) -> None:
        batch = self.make_batch(simulations=1)
        stones, captures = _initial_root(5)
        batch.add(stones, captures, 1, 0)
        selected = batch.select()
        with self.assertRaises((ValueError, RuntimeError)):
            batch.backup(selected.token + 1, selected.size)
        self.assertTrue(batch.has_pending)
        batch.policies[0].fill_(float("nan"))
        with self.assertRaises((ValueError, RuntimeError)):
            batch.backup(selected.token, selected.size)
        self.assertTrue(batch.has_pending)
        batch.policies[0].zero_()
        batch.policies[0, 0] = 1.0
        batch.values[0] = 0.0
        batch.backup(selected.token, selected.size)

    def test_strict_root_admission(self) -> None:
        batch = self.make_batch()
        stones, captures = _initial_root(5)
        with self.assertRaises(ValueError):
            batch.add(stones.to(torch.float32), captures, 1, 0)

        noncontiguous_stones = stones.transpose(0, 1)
        self.assertEqual(tuple(noncontiguous_stones.shape), (5, 5))
        self.assertFalse(noncontiguous_stones.is_contiguous())
        with self.assertRaises(ValueError):
            batch.add(noncontiguous_stones, captures, 1, 0)

        noncontiguous_captures = torch.zeros(4, dtype=torch.int16)[::2]
        self.assertEqual(tuple(noncontiguous_captures.shape), (2,))
        self.assertFalse(noncontiguous_captures.is_contiguous())
        with self.assertRaises(ValueError):
            batch.add(stones, noncontiguous_captures, 1, 0)
        with self.assertRaises((TypeError, ValueError)):
            batch.add(stones, captures, True, 0)
        with self.assertRaises(ValueError):
            batch.add(
                torch.zeros((6, 6), dtype=torch.int8),
                captures,
                1,
                0,
            )

    def test_constructor_validation_and_feature_perspective(self) -> None:
        with self.assertRaises(ValueError):
            self.extension.SearchBatch(
                board_size=4,
                ruleset="freestyle",
                pin_memory=False,
            )
        with self.assertRaises(ValueError):
            self.extension.SearchBatch(
                board_size=6,
                ruleset="standard",
                pin_memory=False,
            )
        with self.assertRaises(TypeError):
            self.extension.SearchBatch(unexpected=True, pin_memory=False)

        batch = self.make_batch(simulations=1)
        stones, captures = _initial_root(5)
        stones[0, 0] = 1
        stones[1, 1] = -1
        captures[:] = torch.tensor([2, 4], dtype=torch.int16)
        batch.add(stones, captures, 1, 14, last_action=6)
        selected = batch.select()
        self.assertTrue(selected.features.is_contiguous())
        self.assertEqual(float(selected.features[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(selected.features[0, 0, 1, 1]), 0.0)
        self.assertEqual(float(selected.features[0, 1, 1, 1]), 1.0)
        self.assertAlmostEqual(float(selected.features[0, 2, 0, 0]), 0.4)
        self.assertAlmostEqual(float(selected.features[0, 3, 0, 0]), 0.8)

    def test_staging_pointers_are_stable_across_waves(self) -> None:
        batch = self.make_batch(simulations=3)
        stones, captures = _initial_root(5)
        batch.add(stones, captures, 1, 0)
        pointers = (
            batch.features.data_ptr(),
            batch.policies.data_ptr(),
            batch.values.data_ptr(),
        )
        while not batch.complete():
            selected = batch.select()
            self.assertGreater(selected.size, 0)
            self.assertEqual(batch.features.data_ptr(), pointers[0])
            self.assertEqual(batch.policies.data_ptr(), pointers[1])
            self.assertEqual(batch.values.data_ptr(), pointers[2])
            batch.policies[: selected.size].zero_()
            batch.policies[: selected.size, 0] = 1.0
            batch.values[: selected.size].zero_()
            batch.backup(selected.token, selected.size)
        self.assertEqual(batch.features.data_ptr(), pointers[0])

    def test_conditional_pinned_storage(self) -> None:
        try:
            pinned = self.extension.SearchBatch(
                board_size=5,
                ruleset="freestyle",
                simulations=1,
                active_games=1,
                threads=1,
                pin_memory=True,
            )
        except RuntimeError:
            return
        self.assertTrue(pinned.pin_memory_requested)
        self.assertTrue(pinned.pin_memory_realized)
        self.assertTrue(pinned.features.is_pinned())
        self.assertTrue(pinned.policies.is_pinned())
        self.assertTrue(pinned.values.is_pinned())

    def test_mutex_getters_do_not_deadlock_repeated_search(self) -> None:
        scenario = r"""
import threading
import time

import torch

import kb_pente_native


batch = kb_pente_native.SearchBatch(
    board_size=19,
    ruleset="freestyle",
    simulations=12,
    active_games=32,
    threads=4,
    seed=103,
    pin_memory=False,
)
captures = torch.zeros(2, dtype=torch.int16)
for action in range(32):
    stones = torch.zeros((19, 19), dtype=torch.int8)
    stones.reshape(-1)[action] = 1
    batch.add(stones, captures, -1, 1, last_action=action)


started = threading.Event()
go = threading.Event()
worker_errors = []
completed_waves = [0]


def search_waves() -> None:
    try:
        started.set()
        if not go.wait(timeout=2.0):
            raise RuntimeError("search worker did not receive the start signal")
        for _ in range(12):
            selection = batch.select()
            batch.policies[: selection.size].fill_(1.0)
            batch.values[: selection.size].zero_()
            batch.backup(selection.token, selection.size)
            completed_waves[0] += 1
    except BaseException as error:
        worker_errors.append(error)


worker = threading.Thread(target=search_waves)
worker.start()
if not started.wait(timeout=1.0):
    raise RuntimeError("search worker did not start")
go.set()

deadline = time.monotonic() + 8.0
getter_iterations = 0
while worker.is_alive() and time.monotonic() < deadline:
    getter_iterations += 1
    batch.status()
    _ = batch.complete()
    _ = batch.slot_active(0)
    _ = batch.slot_complete(0)
    try:
        batch.root_policy(0)
    except (IndexError, RuntimeError, ValueError):
        # The root policy is unavailable until the concurrent search finishes.
        pass
    _ = batch.active_count
    _ = batch.pending_request_count
    _ = batch.pending_selected_count
    _ = batch.has_pending
    _ = batch.pending_token
    _ = batch.last_token
    time.sleep(0)

worker.join(timeout=2.0)
if worker.is_alive():
    raise RuntimeError("select/getter lock-order deadlock")
if worker_errors:
    raise worker_errors[0]
if completed_waves[0] != 12:
    raise RuntimeError(f"search completed {completed_waves[0]} of 12 waves")
if getter_iterations < 2:
    raise RuntimeError("getter loop did not repeatedly overlap native search")
"""
        # A subprocess bounds a regression in which a getter holds the GIL
        # while waiting for the mutex needed by select to finish.
        try:
            completed = subprocess.run(
                [sys.executable, "-c", scenario],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15.0,
            )
        except subprocess.TimeoutExpired as error:
            self.fail(f"GIL/mutex lock-order scenario timed out: {error}")
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_python_thread_progress_during_native_search(self) -> None:
        batch = self.extension.SearchBatch(
            board_size=19,
            ruleset="freestyle",
            simulations=1,
            active_games=16,
            threads=4,
            seed=103,
            pin_memory=False,
        )
        captures = torch.zeros(2, dtype=torch.int16)
        for action in range(16):
            stones = torch.zeros((19, 19), dtype=torch.int8)
            stones.reshape(-1)[action] = 1
            batch.add(stones, captures, -1, 1, last_action=action)

        stop = threading.Event()
        progress_count = [0]

        def progress_loop() -> None:
            while not stop.is_set():
                progress_count[0] += 1

        progress = threading.Thread(target=progress_loop)
        progress.start()
        time.sleep(0.01)
        before = progress_count[0]
        selected = batch.select()
        during = progress_count[0]
        stop.set()
        progress.join(timeout=30.0)
        self.assertGreater(during, before)

        batch.policies[: selected.size].zero_()
        batch.policies[: selected.size, 0] = 1.0
        batch.values[: selected.size].zero_()
        batch.backup(selected.token, selected.size)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SearchBatchBindingTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
