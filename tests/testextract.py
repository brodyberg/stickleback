import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import numpy as np
from numpy.testing import assert_array_equal
import unittest
from stickleback.preprocessing import extract_all, extract_nested, sample_nonevents

n = 20
idx1 = pd.date_range(start="2020-01-01 00:00", freq="S", periods=n, tz="Etc/GMT+7")
idx2 = pd.date_range(start="2020-02-01 00:00", freq="S", periods=n, tz="Etc/GMT+3")
sensors = {
        "d1": pd.DataFrame({
            "a": np.arange(n),
            "b": np.arange(n) ** 2
        }, index = idx1),
        "d2": pd.DataFrame({
            "a": -np.arange(n),
            "b": -(np.arange(n) ** 2)
        }, index = idx2)
    }
events = {
    "d1": idx1[[2, 6]],
    "d2": idx2[[3, 8]]
}

class ExtractTestCase(unittest.TestCase):
    
    def test_extract_all(self):
        winsz = 3
        result = extract_all(sensors, nth=1, win_size=winsz)

        # Size and contents
        self.assertEqual(len(result["d1"]), n - winsz + 1)
        self.assertEqual(len(result["d1"].iloc[0, 0]), winsz)
        assert_array_equal(result["d2"].iloc[-1, -1], -np.arange(n - winsz, n) ** 2)

        # Keys and index
        self.assertEqual(result.keys(), sensors.keys())
        self.assertIsInstance(result["d1"].index, pd.DatetimeIndex)
        assert_index_equal(result["d1"].index, idx1[1:-1])

    def test_extract_nested(self):
        idx = {"d1": idx1[[2, 4]], "d2": idx2[[3, 5]]}
        result = extract_nested(sensors, idx, win_size=3)

        for i in idx:
            assert_index_equal(result[i].index, idx[i])

    def test_sample_nonevents(self):
        result1 = sample_nonevents(sensors, events, win_size=3, seed=0x1234)
        result2 = sample_nonevents(sensors, events, win_size=3, seed=0x1234)

        for r in result1:
            assert_frame_equal(result1[r], result2[r])
        # TODO assert nonevent windows don't overlap event windows
