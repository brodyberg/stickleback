import pandas as pd
import numpy as np
import unittest
from stickleback.preprocessing import extract_all, extract_nested

idx1 = pd.date_range(start="2020-01-01 00:00", freq="S", periods=10, tz="Etc/GMT+7")
idx2 = pd.date_range(start="2020-02-01 00:00", freq="S", periods=10, tz="Etc/GMT+3")
sensors = {
        "d1": pd.DataFrame({
            "a": np.arange(10),
            "b": np.arange(10) ** 2
        }, index = idx1),
        "d2": pd.DataFrame({
            "a": -np.arange(10),
            "b": -(np.arange(10) ** 2)
        }, index = idx2)
    }

class ExtractTestCase(unittest.TestCase):
    
    def test_extract_all(self):
        result = extract_all(sensors, nth=1, win_size=3)

        # Size and contents
        self.assertEqual(len(result), 16)
        self.assertEqual(len(result.iloc[0, 0]), 3)
        self.assertTrue(np.array_equal(result.iloc[-1, -1], [-49, -64, -81]))

        # Index
        self.assertIsInstance(result.index, pd.MultiIndex)
        self.assertEqual(result.index.nlevels, 2)
        self.assertEqual(set(result.index.get_level_values(0)), set(sensors.keys()))
        self.assertTrue(np.array_equal(result.loc["d1"].index, idx1[1:-1]))

    def test_extract_nested(self):
        idx = {"d1": idx1[[2, 4]], "d2": idx2[[3, 5]]}
        result = extract_nested(sensors, idx, win_size=3)

        self.assertEqual(len(result), 4)
        result_idx = pd.MultiIndex.from_tuples([(d, i) for d in idx for i in idx[d]])
        self.assertTrue(np.array_equal(result.index, result_idx))
