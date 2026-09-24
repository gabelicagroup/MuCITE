from __future__ import annotations

import csv

import numpy as np
import pytest

from src.config import PARTICLE_Z_EXIT
from src.agents.events import TerminalEventBatch
from src.core.metrics import BoundedMacroHistory, OnlineTimeStepStatistics
from src.data.report.terminal_summary import _terminal_event_summary
from src.data.terminal_recorder import TerminalEventRecorder


def _terminal_batch(start: int, count: int, *, weight: float = 1.0) -> TerminalEventBatch:
    track_ids = np.arange(start, start + count, dtype=np.int64)
    positions_m = np.zeros((count, 3), dtype=np.float64)
    positions_m[:, 0] = track_ids * 1.0e-9
    positions_m[:, 2] = 0.065
    velocities_m_per_s = np.zeros((count, 3), dtype=np.float64)
    velocities_m_per_s[:, 2] = 1_000.0
    return TerminalEventBatch(
        particle_indices=np.arange(count, dtype=np.int32),
        status_codes=np.full(count, PARTICLE_Z_EXIT, dtype=np.int16),
        event_time_s=(track_ids + 1) * 1.0e-9,
        positions_m=positions_m,
        velocities_m_per_s=velocities_m_per_s,
        temperatures_k=np.full(count, 300.0, dtype=np.float64),
        represented_real_ions=np.full(count, weight, dtype=np.float64),
        masses_kg=np.full(count, 100.0 * 1.66053906660e-27, dtype=np.float64),
        track_ids=track_ids,
        birth_time_s=np.zeros(count, dtype=np.float64),
        collision_counts=np.zeros(count, dtype=np.int64),
        tof_s=(track_ids + 1) * 1.0e-9,
        electrode_ids=np.full(count, -1, dtype=np.int32),
        surface_distance_m=np.full(count, np.nan, dtype=np.float64),
    )


def test_terminal_stream_only_is_bounded_and_event_ids_remain_monotonic(tmp_path) -> None:
    event_count = 30_000
    sample_size = 37
    stream_path = tmp_path / "terminal_events.csv"
    recorder = TerminalEventRecorder(sample_seed=17)
    recorder.open_csv_stream(stream_path, stream_only=True, sample_size=sample_size)

    for start in range(0, event_count, 1_000):
        recorder.record(
            _terminal_batch(start, 1_000),
            source_mode="packet",
            terminal_event_mode="transport-only",
            max_terminal_event_rows=0,
        )
        assert recorder.retained_row_count <= sample_size
    recorder.close()

    assert recorder.recorded_row_count == event_count
    assert recorder.streamed_row_count == event_count
    assert recorder.retained_row_count == sample_size
    assert recorder.next_event_id == event_count
    assert recorder.omitted_rows_by_status == {}
    assert recorder.recorded_summary_by_status()["z_exit"]["macro_event_count"] == event_count
    assert recorder.recorded_summary_by_status()["z_exit"]["represented_real_ions"] == pytest.approx(
        event_count
    )
    retained_ids = [int(row["event_id"]) for row in recorder.report_rows()]
    assert retained_ids == sorted(retained_ids)
    assert len(set(retained_ids)) == sample_size

    csv_row_count = 0
    with stream_path.open(newline="", encoding="utf-8") as handle:
        for expected_event_id, row in enumerate(csv.DictReader(handle)):
            assert int(row["event_id"]) == expected_event_id
            csv_row_count += 1
    assert csv_row_count == event_count

    summary = _terminal_event_summary(
        recorder.report_rows(),
        total_row_count=recorder.recorded_row_count,
        exact_by_status=recorder.recorded_summary_by_status(),
        distribution_sample_is_complete=recorder.sample_is_complete,
    )
    assert summary["row_count"] == event_count
    assert summary["distribution_sample_row_count"] == sample_size
    assert summary["distribution_sample_is_complete"] is False
    assert summary["distribution_statistics"] == "bounded_uniform_reservoir_approximation"
    assert summary["by_status"]["z_exit"]["macro_event_count"] == event_count
    assert summary["by_status"]["z_exit"]["distribution_sample_macro_event_count"] == sample_size


def test_terminal_stream_only_max_rows_uses_output_count_and_accounts_omissions(tmp_path) -> None:
    stream_path = tmp_path / "terminal_events.csv"
    recorder = TerminalEventRecorder(sample_seed=3)
    recorder.open_csv_stream(stream_path, stream_only=True, sample_size=2)

    recorder.record(
        _terminal_batch(0, 5, weight=2.0),
        source_mode="packet",
        terminal_event_mode="transport-only",
        max_terminal_event_rows=7,
    )
    recorder.record(
        _terminal_batch(5, 5, weight=2.0),
        source_mode="packet",
        terminal_event_mode="transport-only",
        max_terminal_event_rows=7,
    )
    recorder.record(
        _terminal_batch(10, 4, weight=2.0),
        source_mode="packet",
        terminal_event_mode="transport-only",
        max_terminal_event_rows=7,
    )
    recorder.close()

    assert recorder.recorded_row_count == 7
    assert recorder.streamed_row_count == 7
    assert recorder.retained_row_count == 2
    assert recorder.next_event_id == 7
    assert recorder.max_rows_reached is True
    assert recorder.omitted_rows_by_status == {"z_exit": 7}
    assert recorder.omitted_real_ions_by_status == {"z_exit": pytest.approx(14.0)}
    assert recorder.recorded_summary_by_status()["z_exit"] == {
        "macro_event_count": 7,
        "represented_real_ions": pytest.approx(14.0),
        "parent_real_ions_remaining": pytest.approx(14.0),
        "fragmented_real_ions_represented": pytest.approx(0.0),
    }

    with stream_path.open(newline="", encoding="utf-8") as handle:
        event_ids = [int(row["event_id"]) for row in csv.DictReader(handle)]
    assert event_ids == list(range(7))


def test_terminal_stream_only_clear_resets_ids_counters_and_sample(tmp_path) -> None:
    stream_path = tmp_path / "terminal_events.csv"
    recorder = TerminalEventRecorder()
    recorder.open_csv_stream(stream_path, stream_only=True, sample_size=3)
    recorder.record(
        _terminal_batch(0, 10),
        source_mode="packet",
        max_terminal_event_rows=0,
    )

    recorder.clear()
    recorder.record(
        _terminal_batch(100, 2),
        source_mode="packet",
        max_terminal_event_rows=0,
    )
    recorder.close()

    assert recorder.recorded_row_count == 2
    assert recorder.streamed_row_count == 2
    assert recorder.next_event_id == 2
    assert [int(row["event_id"]) for row in recorder.report_rows()] == [0, 1]
    with stream_path.open(newline="", encoding="utf-8") as handle:
        assert [int(row["event_id"]) for row in csv.DictReader(handle)] == [0, 1]


def test_online_dt_statistics_keep_exact_moments_and_fixed_quantile_memory() -> None:
    reservoir_size = 257
    values = np.linspace(1.0e-12, 1.0e-6, 200_000, dtype=np.float64)
    statistics = OnlineTimeStepStatistics(
        quantile_reservoir_size=reservoir_size,
        seed=11,
        limiter_names=("collision", "rf", "macro_end"),
    )

    statistics.update_many(values[:100_000])
    assert statistics.quantile_sample_count == reservoir_size
    statistics.update_many(values[100_000:])
    statistics.update(2.0e-9, limiter="collision", raw_limiter="rf")
    statistics.update(3.0e-9, limiter="collision", raw_limiter="collision")
    statistics.update(4.0e-9, limiter="macro_end", raw_limiter="rf")

    expected = np.concatenate((values, np.array([2.0e-9, 3.0e-9, 4.0e-9])))
    summary = statistics.summary()
    assert summary["micro_step_count"] == expected.size
    assert summary["min_dt_s"] == pytest.approx(float(np.min(expected)))
    assert summary["max_dt_s"] == pytest.approx(float(np.max(expected)))
    assert summary["mean_dt_s"] == pytest.approx(float(np.mean(expected)), rel=1.0e-12)
    assert summary["quantile_sample_count"] == reservoir_size
    assert summary["quantile_reservoir_size"] == reservoir_size
    assert summary["quantiles_approximate"] is True
    assert summary["p50_dt_s"] == pytest.approx(float(np.quantile(expected, 0.50)), rel=0.15)
    assert summary["p95_dt_s"] == pytest.approx(float(np.quantile(expected, 0.95)), rel=0.15)
    assert statistics.limiter_counts == {"collision": 2, "rf": 0, "macro_end": 1}
    assert statistics.raw_limiter_counts == {"collision": 1, "rf": 2, "macro_end": 0}

    with pytest.raises(ValueError, match="finite and positive"):
        statistics.update(float("nan"))
    with pytest.raises(ValueError, match="finite and positive"):
        statistics.update(0.0)


def test_bounded_macro_history_retains_endpoints_and_constant_sample_size() -> None:
    history = BoundedMacroHistory(
        max_sample_rows=32,
        summary_fields=("time_s", "ion_count"),
        seed=23,
    )
    macro_count = 100_000
    for index in range(macro_count):
        history.record(
            {
                "time_s": index * 1.0e-8,
                "ion_count": 100.0 + index % 5,
                "stage_name": "loading" if index < 50_000 else "hold",
            }
        )
        if index in {9_999, 49_999, 99_999}:
            assert history.retained_row_count == 32

    sample = history.sample_rows()
    summary = history.summary()
    assert history.count == macro_count
    assert len(sample) == 32
    assert sample[0]["time_s"] == 0.0
    assert sample[-1]["time_s"] == pytest.approx((macro_count - 1) * 1.0e-8)
    assert [float(row["time_s"]) for row in sample] == sorted(float(row["time_s"]) for row in sample)
    assert summary["macro_step_count"] == macro_count
    assert summary["sample_row_count"] == 32
    assert summary["sample_is_complete"] is False
    assert summary["first"]["stage_name"] == "loading"
    assert summary["last"]["stage_name"] == "hold"
    assert summary["numeric_fields"]["time_s"]["count"] == macro_count
    assert summary["numeric_fields"]["time_s"]["min"] == 0.0
    assert summary["numeric_fields"]["time_s"]["max"] == pytest.approx((macro_count - 1) * 1.0e-8)
    assert summary["numeric_fields"]["ion_count"]["mean"] == pytest.approx(102.0)
