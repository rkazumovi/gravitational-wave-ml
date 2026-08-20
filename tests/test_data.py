"""Tests for src/data/ligo_loader.py against real cached GW150914 strain files.

Uses the real data downloaded by running ligo_loader.py directly (cached
under data/) rather than mocks, per this project's validate-against-real-data
approach — see docs/step2_data_loading.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.ligo_loader import CBC_ANALYSIS_READY_MASK, load_event_strain  # noqa: E402

GW150914_GPS = 1126259462.4


@pytest.fixture(scope="module")
def h1_segment():
    return load_event_strain("GW150914", detector="H1")


@pytest.fixture(scope="module")
def l1_segment():
    return load_event_strain("GW150914", detector="L1")


def test_segment_shape_and_timing(h1_segment):
    assert h1_segment.sample_rate == 4096
    assert len(h1_segment.strain) == 4096 * 4096
    assert h1_segment.duration == pytest.approx(4096.0)
    assert h1_segment.gps_end == pytest.approx(h1_segment.gps_start + 4096.0)


def test_strain_amplitude_is_physically_reasonable(h1_segment):
    finite = h1_segment.strain[np.isfinite(h1_segment.strain)]
    # Real GW strain is O(1e-21 to 1e-18): far smaller than 1, far larger than 0.
    assert 1e-22 < finite.std() < 1e-15


def test_analysis_ready_at_real_merger_time(h1_segment, l1_segment):
    assert h1_segment.is_analysis_ready(GW150914_GPS)
    assert l1_segment.is_analysis_ready(GW150914_GPS)


def test_is_analysis_ready_rejects_out_of_range_gps(h1_segment):
    with pytest.raises(ValueError):
        h1_segment.is_analysis_ready(h1_segment.gps_start - 10)


def test_slice_gps_returns_expected_window(h1_segment):
    window = h1_segment.slice_gps(GW150914_GPS, half_width=8.0)
    assert len(window.strain) == int(2 * 8.0 * h1_segment.sample_rate)
    assert window.gps_start <= GW150914_GPS <= window.gps_end


def test_slice_gps_rejects_out_of_bounds_window(h1_segment):
    with pytest.raises(ValueError):
        h1_segment.slice_gps(h1_segment.gps_start, half_width=100.0)


def test_l1_has_a_real_nan_gap_correctly_flagged_by_dqmask(l1_segment):
    """See docs/step2_data_loading.md: L1's raw array has a genuine 189s NaN
    gap at the start of the segment, far from the merger. The DQmask's DATA
    bit must read 0 throughout that gap and 1 immediately after."""
    nan_mask = np.isnan(l1_segment.strain)
    assert nan_mask.any(), "expected file is the known one with a leading NaN gap"

    nan_seconds = np.where(nan_mask)[0] // l1_segment.sample_rate
    first_gap_second, last_gap_second = nan_seconds.min(), nan_seconds.max()

    assert np.all(l1_segment.dq_mask[first_gap_second:last_gap_second + 1] & 1 == 0)
    assert l1_segment.dq_mask[last_gap_second + 1] & 1 == 1

    # The merger window itself must be untouched by the gap.
    merger_window = l1_segment.slice_gps(GW150914_GPS, half_width=16.0)
    assert not np.isnan(merger_window.strain).any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
