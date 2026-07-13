"""Picklable worker fixtures for collector multiprocessing tests.

Author: Rui Heng Yang
"""

from __future__ import annotations

import time
import signal


def output_then_sleep(worker_id: int, queue) -> None:
    """Publish one required output, then emulate a worker stuck in cleanup."""
    queue.put(("item", worker_id))
    time.sleep(10.0)


def output_then_ignore_sigterm(worker_id: int, queue) -> None:
    """Publish required output, ignore SIGTERM, and wait for kill fallback."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    queue.put(("item", worker_id))
    time.sleep(10.0)


def exit_without_message(_worker_id: int, _queue) -> None:
    """Exit normally without publishing output or a completion sentinel."""


def report_done(worker_id: int, queue) -> None:
    """Publish a normal completion sentinel with deterministic stats."""
    queue.put(("worker_done", worker_id, {"tries": worker_id + 1, "fails": 0}))


def report_done_then_sleep(worker_id: int, queue) -> None:
    """Publish completion, then emulate slow but valid interpreter teardown."""
    queue.put(("worker_done", worker_id, {"tries": 1, "fails": 0}))
    time.sleep(0.2)
