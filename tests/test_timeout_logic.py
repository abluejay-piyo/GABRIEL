"""Tests for timeout selection helpers."""

import math
import time

import pandas as pd
import pytest

from gabriel.utils.openai_utils import (
    _collect_successful_time_taken_samples,
    _compute_dynamic_timeout_from_samples,
    _resolve_effective_timeout,
    _should_cancel_inflight_task,
)


def test_resolve_effective_timeout_uses_task_budget_when_available() -> None:
    """Retries keep their extended timeout budgets when dynamic timeouts run."""

    assert _resolve_effective_timeout(90.0, 135.0, True) == 135.0


def test_resolve_effective_timeout_falls_back_to_global_timeout() -> None:
    """Tasks dispatched before initialization should respect the global limit."""

    assert _resolve_effective_timeout(90.0, math.inf, True) == 90.0


def test_resolve_effective_timeout_respects_explicit_timeouts_when_static() -> None:
    """Static timeout configuration should always use the provided value."""

    assert _resolve_effective_timeout(math.inf, 40.0, False) == 40.0


def test_should_cancel_inflight_honors_dynamic_budget() -> None:
    """Tasks dispatched before initialization adopt the global timeout."""

    start = time.time() - 100.0
    now = time.time()
    assert _should_cancel_inflight_task(start, now, 90.0, math.inf, True)


def test_should_cancel_inflight_skips_infinite_budgets() -> None:
    """When no timeout applies the watcher should not cancel tasks."""

    start = time.time() - 10.0
    now = time.time()
    assert not _should_cancel_inflight_task(start, now, math.inf, math.inf, True)


def test_collect_successful_time_taken_samples_filters_invalid_values() -> None:
    df = pd.DataFrame(
        {
            "Time Taken": [0.15, "bad", -1, None, 0.5],
            "Successful": [True, True, True, True, False],
        }
    )
    success_mask = pd.Series([True, True, True, True, False])

    assert _collect_successful_time_taken_samples(df, success_mask=success_mask) == [0.15]


def test_compute_dynamic_timeout_from_samples_applies_cap() -> None:
    timeout_stats = _compute_dynamic_timeout_from_samples(
        [1.0, 2.0, 3.0, 4.0],
        timeout_factor=2.0,
        max_timeout=5.0,
    )

    assert timeout_stats is not None
    timeout, p90 = timeout_stats
    assert p90 == pytest.approx(3.7)
    assert timeout == pytest.approx(5.0)


def test_compute_dynamic_timeout_from_samples_returns_none_without_valid_data() -> None:
    timeout_stats = _compute_dynamic_timeout_from_samples(
        [float("nan"), -1.0, 0.0],
        timeout_factor=2.0,
        max_timeout=10.0,
    )

    assert timeout_stats is None
