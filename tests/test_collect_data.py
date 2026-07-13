"""Regression tests for multiprocess collector orchestration.

Author: Rui Heng Yang
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch.multiprocessing as mp

from script import collect_data
from tests.multiprocessing_fixtures import (
    exit_without_message,
    output_then_ignore_sigterm,
    output_then_sleep,
    report_done,
    report_done_then_sleep,
)


class MultiprocessCollectorTests(unittest.TestCase):
    """Verify output-aware completion and worker-failure diagnostics."""

    def setUp(self) -> None:
        """Create an isolated spawn context for each test."""
        self.context = mp.get_context("spawn")

    def test_required_outputs_complete_without_worker_done(self) -> None:
        """A complete phase should not require a final worker sentinel."""
        queue = self.context.Queue()
        received_workers: set[int] = set()

        def on_message(message: tuple) -> None:
            self.assertEqual(message[0], "item")
            received_workers.add(int(message[1]))

        with patch.object(collect_data, "WORKER_SHUTDOWN_GRACE_SECONDS", 0.05):
            stats = collect_data._spawn_and_consume(
                self.context,
                output_then_sleep,
                [(0, queue), (1, queue)],
                queue,
                on_message,
                is_complete=lambda: received_workers == {0, 1},
            )

        self.assertEqual(received_workers, {0, 1})
        self.assertEqual(stats, [])

    def test_premature_exit_reports_worker_identity_and_exit_code(self) -> None:
        """An incomplete phase should expose actionable dead-worker details."""
        queue = self.context.Queue()

        with (
            patch.object(collect_data, "WORKER_QUEUE_TIMEOUT_SECONDS", 0.05),
            patch.object(collect_data, "WORKER_SHUTDOWN_GRACE_SECONDS", 0.05),
            self.assertRaisesRegex(
                RuntimeError,
                r"worker=0 pid=\d+ exitcode=0",
            ),
        ):
            collect_data._spawn_and_consume(
                self.context,
                exit_without_message,
                [(0, queue)],
                queue,
                lambda _message: None,
            )

    def test_output_completion_kills_sigterm_resistant_worker(self) -> None:
        """Rendering must not start while a cleanup-stuck worker survives."""
        queue = self.context.Queue()
        received_workers: set[int] = set()

        with patch.object(collect_data, "WORKER_SHUTDOWN_GRACE_SECONDS", 0.05):
            collect_data._spawn_and_consume(
                self.context,
                output_then_ignore_sigterm,
                [(0, queue)],
                queue,
                lambda message: received_workers.add(int(message[1])),
                is_complete=lambda: received_workers == {0},
            )

        self.assertEqual(received_workers, {0})

    def test_sentinel_phase_allows_slow_normal_teardown(self) -> None:
        """Render/regen workers retain their original unbounded join behavior."""
        queue = self.context.Queue()

        with patch.object(collect_data, "WORKER_SHUTDOWN_GRACE_SECONDS", 0.01):
            stats = collect_data._spawn_and_consume(
                self.context,
                report_done_then_sleep,
                [(0, queue)],
                queue,
                lambda _message: None,
            )

        self.assertEqual(stats, [{"tries": 1, "fails": 0}])

    def test_normal_worker_sentinels_preserve_stats(self) -> None:
        """Sentinel-driven phases should retain their existing stats contract."""
        queue = self.context.Queue()

        stats = collect_data._spawn_and_consume(
            self.context,
            report_done,
            [(0, queue), (1, queue)],
            queue,
            lambda _message: None,
        )

        self.assertCountEqual(
            stats,
            [{"tries": 1, "fails": 0}, {"tries": 2, "fails": 0}],
        )


if __name__ == "__main__":
    unittest.main()
