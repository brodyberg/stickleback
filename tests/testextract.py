import pandas as pd
from pandas.util.testing import assert_frame_equal
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
        self.assertEqual(len(result), n * 2 - winsz - 1)
        self.assertEqual(len(result.iloc[0, 0]), winsz)
        assert_array_equal(result.iloc[-1, -1], -np.arange(n - winsz, n) ** 2)

        # Index
        self.assertIsInstance(result.index, pd.MultiIndex)
        self.assertEqual(result.index.nlevels, 2)
        self.assertEqual(set(result.index.get_level_values(0)), set(sensors.keys()))
        assert_array_equal(result.loc["d1"].index, idx1[1:-1])

    def test_extract_nested(self):
        idx = {"d1": idx1[[2, 4]], "d2": idx2[[3, 5]]}
        result = extract_nested(sensors, idx, win_size=3)

        self.assertEqual(len(result), 4)
        result_idx = pd.MultiIndex.from_tuples([(d, i) for d in idx for i in idx[d]])
        assert_array_equal(result.index, result_idx)

    def test_sample_nonevents(self):
        result1 = sample_nonevents(sensors, events, win_size=3, seed=0x1234)
        result2 = sample_nonevents(sensors, events, win_size=3, seed=0x1234)

        assert_frame_equal(result1, result2)
        #TODO assert nonevent windows don't overlap event windows
