import pandas as pd
import pickle
import numpy as np
import sktime.datatypes._panel._convert as convert 
from stickleback.types import *
from typing import Collection, Dict, Tuple

def extract_nested(sensors: sensors_T, 
                   idx: Dict[str, pd.DatetimeIndex], 
                   win_size: int) -> nested_T:
    win_size_2 = int(win_size / 2)

    def _extract(_deployid: str, _idx: pd.DatetimeIndex):
        _sensors = sensors[_deployid]
        idx = _sensors.index.get_indexer(_idx)
        data_3d = np.empty([len(idx), len(_sensors.columns), win_size], float)
        data_arr = _sensors.to_numpy().transpose()
        start_idx = idx - win_size_2
        for i, start in enumerate(start_idx):
            data_3d[i] = data_arr[:, start:(start + win_size)]
        nested = convert.from_3d_numpy_to_nested(data_3d)
        nested.columns = _sensors.columns
        nested.index = _sensors.index[idx]
        return nested

    return {d: _extract(d, i) for d, i in idx.items()}

def extract_all(sensors: sensors_T, 
                nth: int, 
                win_size: int, 
                mask: mask_T = None) -> nested_T:
    if mask is None:
        mask = {d: np.full(len(sensors[d]), True) for d in sensors}
        
    win_size_2 = int(win_size / 2)
    idx = dict()
    for d in sensors:
        _idx = np.arange(win_size_2, len(sensors[d]) - win_size_2, nth)
        # Next line (admittedly) confusing. Look up _idx in mask[d] and keep
        # only those where mask is True
        _idx = _idx[mask[d][_idx]]
        idx[d] = sensors[d].index[_idx]
        
    return extract_nested(sensors, idx, win_size)
    
def sample_nonevents(sensors: sensors_T, 
                     events: events_T, 
                     win_size: int, 
                     mask: mask_T = None, 
                     seed: int = None) -> nested_T:
    win_size_2 = int(win_size / 2)
    rg = np.random.Generator(np.random.PCG64(seed))
    if mask is None:
        mask = {d: np.full(len(sensors[d]), True) for d in sensors}

    def _diff_from(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return np.array([np.min(np.abs(x - ys)) for x in xs])

    def _sample(_sensors: pd.DataFrame, 
                _events: pd.DatetimeIndex, 
                _mask: np.ndarray):
        nonevent_choices = np.arange(win_size_2, 
                                     len(_sensors) - win_size_2, 
                                     win_size)
        nonevent_choices = nonevent_choices[_mask[nonevent_choices]]
        diff_from_event = _diff_from(nonevent_choices, 
                                     _sensors.index.searchsorted(_events))
        nonevent_choices = nonevent_choices[diff_from_event > win_size]
        return _sensors.index[rg.choice(nonevent_choices, 
                                        size=len(_events), 
                                        replace=True)]

    idx = {d: _sample(sensors[d], events[d], mask[d]) for d in sensors}
    return extract_nested(sensors, idx, win_size)

def align_events(events: events_T, sensors: sensors_T) -> events_T:
    return {d: sensors[d].index[sensors[d].index.searchsorted(e)] 
            for d, e in events.items()}

def filter_dict(d: Dict, ks: Collection) -> Dict:
    return {k: v for k, v in d.items() if k in ks}

def save_fitted(sb: "Stickleback", 
                fp: str,
                sensors: sensors_T = None, 
                events: events_T = None, 
                mask: mask_T = None, 
                predicted: prediction_T = None) -> None:
    objects = (sb, sensors, events, mask, predicted)
    with open(fp, 'wb') as f:
        pickle.dump(objects, f)
        
def load_fitted(fp: str) -> Tuple["Stickleback", 
                                  sensors_T, 
                                  events_T, 
                                  mask_T, 
                                  prediction_T]:
    with open(fp, 'rb') as f:
        result = pickle.load(f)
    
    return result
