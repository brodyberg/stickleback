import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sktime.utils import from_3d_numpy_to_nested
from typing import Dict, Tuple

def extract_nested(sensors: Dict[str, pd.DataFrame], idx: Dict[str, pd.DatetimeIndex], 
                   win_size: int) -> Dict[str, pd.DataFrame]:
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
        nested.index = _sensors.index[idx]
        return nested

    return {d: _extract(d, i) for d, i in idx.items()}

def extract_all(sensors: Dict[str, pd.DataFrame], nth: int, win_size: int, mask: Dict[str, np.ndarray] = None) -> Dict[str, pd.DataFrame]:
    if mask is None:
        mask = {d: np.full(len(sensors[d]), True) for d in sensors}
    win_size_2 = int(win_size / 2)
    idx = dict()
    for d in sensors:
        _idx = np.arange(win_size_2, len(sensors[d]) - win_size_2, nth)
        _idx = _idx[mask[d]]
        idx[d] = sensors[d].index[_idx]
    return extract_nested(sensors, idx, win_size)
    
def sample_nonevents(sensors: Dict[str, pd.DataFrame], events: Dict[str, pd.DatetimeIndex], win_size: int, 
                     mask: Dict[str, np.ndarray] = None, seed: int = None) -> Dict[str, pd.DataFrame]:
    win_size_2 = int(win_size / 2)
    rg = np.random.Generator(np.random.PCG64(seed))
    if mask is None:
        mask = {d: np.full(len(sensors[d]), True) for d in sensors}

    def _diff_from(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return np.array([np.min(np.abs(x - ys)) for x in xs])

    def _sample(_sensors: pd.DataFrame, _events: pd.DatetimeIndex, _mask: np.ndarray):
        nonevent_choices = np.array(range(win_size_2, len(_sensors) - win_size_2 - 1, win_size))
        nonevent_choices = nonevent_choices[_mask[nonevent_choices]]
        diff_from_event = _diff_from(nonevent_choices, _sensors.index.searchsorted(_events))
        nonevent_choices = nonevent_choices[diff_from_event > win_size]
        return _sensors.index[rg.choice(nonevent_choices, size=len(_events), replace=False)]

    idx = {d: _sample(sensors[d], events[d], mask[d]) for d in sensors}
    return extract_nested(sensors, idx, win_size)

def extract_peaks(local_proba: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
    def _extract_peaks(x: pd.Series) -> pd.DataFrame:
        peak_idxs, peak_props = find_peaks(x.fillna(0), height=0.1, prominence=0.1, width=1, rel_height=0.5)
        result = pd.DataFrame(peak_props, index=x.index[peak_idxs])[["peak_heights", "prominences", "widths"]]
        return result
    return {d: _extract_peaks(p) for d, p in local_proba.items()}

def align_events(events: Dict[str, pd.DatetimeIndex], sensors: Dict[str, pd.DataFrame]) -> Dict[str, pd.DatetimeIndex]:
    return {d: sensors[d].index[sensors[d].index.searchsorted(e)] 
            for d, e in events.items()}

def split_dict(dict: dict, keys: set) -> Tuple[dict, dict]:
    dict1 = {k: v for k, v in dict.items() if k in keys}
    dict2 = {k: v for k, v in dict.items() if k not in keys}
    return dict1, dict2
