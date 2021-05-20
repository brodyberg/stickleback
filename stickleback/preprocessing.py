import pandas as pd
import numpy as np
from sktime.utils.data_processing import from_3d_numpy_to_nested
from typing import Dict

def extract_all(sensors: Dict[str, pd.DataFrame], nth: int, win_size: int) -> pd.DataFrame:
    win_size_2 = int(win_size / 2)

    def _extract(_deployid: str, _sensors: pd.DataFrame):
        idx = np.arange(win_size_2, len(_sensors) - win_size_2, nth)
        data_3d = np.empty([len(idx), len(_sensors.columns), win_size], float)
        data_arr = _sensors.to_numpy().transpose()
        start_idx = idx - win_size_2
        for i, start in enumerate(start_idx):
            data_3d[i] = data_arr[:, start:(start + win_size)]
        nested = from_3d_numpy_to_nested(data_3d)
        nested.columns = _sensors.columns
        nested.index = pd.MultiIndex.from_product([[_deployid], _sensors.index[idx]])
        return nested

    return pd.concat([_extract(d, s) for d, s in sensors.items()])
    