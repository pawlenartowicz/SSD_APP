"""Pure-math chart helpers from ssdiff_gui.utils.charts.

render_sweep_plot is not covered here (it renders a QPixmap — covered
end-to-end in integration/test_result_export.py::test_sweep_plot_writes_png).
"""

import numpy as np
import pytest

from ssdiff_gui.utils.charts import _nice_ticks, _rolling_median


class TestNiceTicks:
    def test_simple_range(self):
        ticks = _nice_ticks(0.0, 100.0, target=5)
        assert ticks[0] <= 0.0
        assert ticks[-1] >= 100.0
        assert 4 <= len(ticks) <= 7

    def test_tiny_range_returns_single_tick(self):
        """hi - lo < 1e-12 is the degenerate case."""
        assert _nice_ticks(1.0, 1.0 + 1e-15) == [1.0]

    def test_negative_range(self):
        ticks = _nice_ticks(-5.0, 5.0, target=5)
        # Ticks live inside [lo, hi], within one step of each endpoint.
        assert -5.0 <= ticks[0] < 0.0
        assert 0.0 < ticks[-1] <= 5.0
        assert 0.0 in ticks or any(abs(t) < 1e-9 for t in ticks)

    def test_ticks_monotonic(self):
        ticks = _nice_ticks(0.1, 0.9)
        assert all(ticks[i] < ticks[i + 1] for i in range(len(ticks) - 1))

    def test_small_decimal_range(self):
        ticks = _nice_ticks(0.001, 0.002)
        assert len(ticks) >= 2
        assert ticks[0] <= 0.001
        assert ticks[-1] >= 0.002


class TestRollingMedian:
    def test_constant_input(self):
        x = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        result = _rolling_median(x, window=3)
        assert np.allclose(result, 3.0)

    def test_handles_nan(self):
        x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = _rolling_median(x, window=3)
        assert not np.isnan(result[2])
        assert result[2] == 3.0

    def test_all_nan_stays_nan(self):
        x = np.array([np.nan, np.nan, np.nan])
        result = _rolling_median(x, window=3)
        assert np.all(np.isnan(result))

    def test_odd_window(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rolling_median(x, window=3)
        assert result[2] == 3.0

    def test_output_length_matches_input(self):
        x = np.arange(10, dtype=float)
        result = _rolling_median(x, window=5)
        assert len(result) == len(x)
