import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sktime.utils.data_processing import from_3d_numpy_to_nested
from typing import Dict, Tuple

def extract_all(sensors: Dict[str, pd.DataFrame], nth: int, win_size: int) -> pd.DataFrame:
    win_size_2 = int(win_size / 2)
    idx = {d: sensors[d].iloc[win_size_2:-win_size_2:nth].index for d in sensors}
    return extract_nested(sensors, idx, win_size)

def extract_nested(sensors: Dict[str, pd.DataFrame], idx: Dict[str, pd.DatetimeIndex], 
                   win_size: int) -> pd.DataFrame:
    win_size_2 = int(win_size / 2)

    def _extract(_deployid: str, _idx: pd.DatetimeIndex):
        _sensors = sensors[_deployid]
        idx = _sensors.index.get_indexer(_idx)
        data_3d = np.empty([len(idx), len(_sensors.columns), win_size], float)
        data_arr = _sensors.to_numpy().transpose()
        start_idx = idx - win_size_2
        for i, start in enumerate(start_idx):
            data_3d[i] = data_arr[:, start:(start + win_size)]
        nested = from_3d_numpy_to_nested(data_3d)
        nested.columns = _sensors.columns
        nested.index = pd.MultiIndex.from_product([[_deployid], _sensors.index[idx]])
        return nested

    return pd.concat([_extract(d, i) for d, i in idx.items()])
    
def sample_nonevents(sensors: Dict[str, pd.DataFrame], events: Dict[str, pd.DatetimeIndex], win_size: int, 
                     seed: int = None) -> Dict[str, pd.DataFrame]:
    win_size_2 = int(win_size / 2)
    rg = np.random.Generator(np.random.PCG64(seed))

    def _diff_from(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return np.array([np.min(np.abs(x - ys)) for x in xs])

    def _sample(_sensors: pd.DataFrame, _events: pd.DatetimeIndex):
        nonevent_choices = np.array(range(win_size_2, len(_sensors) - win_size_2 - 1, win_size))
        diff_from_event = _diff_from(nonevent_choices, _sensors.index.searchsorted(_events))
        nonevent_choices = nonevent_choices[diff_from_event > win_size]
        return _sensors.index[rg.choice(nonevent_choices, size=len(_events), replace=False)]

    idx = {d: _sample(sensors[d], events[d]) for d in sensors}
    return extract_nested(sensors, idx, win_size)
