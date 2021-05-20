from functools import reduce
from stickleback.preprocessing import extract_all, extract_nested, sample_nonevents
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sktime.classification.compose import TimeSeriesForestClassifier, ColumnEnsembleClassifier
from typing import Dict, Tuple, Union

from ipdb import set_trace

def fit_local(local_clf, local_X: pd.DataFrame, local_y: list) -> None:
    local_clf.fit(local_X, local_y)

def predict_local(local_clf, sensors: Dict[str, pd.DataFrame], nth: int, win_size: int) -> Dict[str, pd.Series]:
    X = extract_all(sensors, nth, win_size)
    return {deployid: local_clf.predict_proba(X[deployid]) for deployid in X}

def extract_peaks(local_proba: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
    def _extract_peaks(x: pd.Series) -> pd.DataFrame:
        peak_idxs, peak_props = find_peaks(x.fillna(0), height=0.25, prominence=0.01, width=1, rel_height=0.5)
        return pd.DataFrame(peak_props, index=x.index[peak_idxs])[["peak_heights", "prominences", "widths"]]
    return {d: _extract_peaks(p) for d, p in local_proba.items()}

def label_peaks(peaks: Dict[str, pd.DataFrame], events: Dict[str, pd.DatetimeIndex], tol: pd.Timedelta) -> Dict[str, pd.Series]:
    def _label_peaks(_peaks: pd.DataFrame, _events: pd.DatetimeIndex) -> pd.Series:
        times = _peaks.index

        def find_nearest_peak(_event):
            near = np.where(np.logical_and(_event - tol <= times, times <= _event + tol))[0]
            near_heights = _peaks["peak_heights"][near]
            return near[np.argmax(near_heights)] if len(near_heights) > 0 else None
        
        nearest_peaks = [find_nearest_peak(e) for e in _events]
        nearest_peaks = [i for i in nearest_peaks if i is not None]
        tps = times[nearest_peaks]
        outcomes = pd.Series(0, index=times)
        outcomes[tps] = 1
        return outcomes
    return {d: _label_peaks(peaks[d], events[d]) for d in peaks}

def fit_global(global_clf, global_X: Dict[str, pd.DataFrame], global_y: Dict[str, pd.Series]) -> None:
    global_clf.fit(global_X, global_y)

def predict_global(global_clf, local_proba: Dict[str, pd.Series]) -> Dict[str, pd.DatetimeIndex]:
    peaks = extract_peaks(local_proba)
    return {deployid: global_clf.predict(peaks[deployid]) for deployid in local_proba}

def assess_predictions(predicted: Dict[str, pd.DatetimeIndex], events: Dict[str, pd.DatetimeIndex], tol: pd.Timedelta) -> Dict[str, pd.Series]:
    def _assess_predictions(_predicted: pd.DatetimeIndex, _events: pd.DatetimeIndex) -> pd.Series:
        # Find closest predicted to each actual and their distance
        # TODO handle no predicted events case
        closest = _predicted[[np.argmin(np.abs(_predicted - a)) for a in _events]]
        distance = np.abs(_events - closest)
        
        # Initialize outcomes
        outcomes = pd.Series(index=_predicted, dtype="string", name="outcome")
        
        # Iterate through actual events. The closest predicted event within the tolerance is a true positive. If no
        # predicted events are within the tolerance, the actual event is a false negative.
        for i, (c, d) in enumerate(zip(closest, distance)):
            if d <= tol:
                outcomes[c] = "TP" 
            else:
                outcomes[_events[i]] = "FN"

        # Iterate through predicted events. Any predicted events that aren't the closest to an actual event are false
        # positives.
        for i, p in enumerate(_predicted):
            if p not in closest:
                outcomes[p] = "FP" 

        return outcomes
    return {deployid: _assess_predictions(predicted[deployid], events[deployid]) for deployid in predicted}
    
def boost(local_X: pd.DataFrame, local_y: list, sensors: Dict[str, pd.DataFrame], 
            outcomes: Dict[str, pd.Series], win_size: int) -> Tuple[pd.DataFrame, list]:
    fps = {d: o.index[o == "FP"] for d, o in outcomes.items()}
    nfps = np.sum([len(f) for f in fps.values()])
    boosted_X = local_X.append(extract_nested(sensors, fps, win_size))
    boosted_y = local_y.append(["nonevent"] * nfps)
    return boosted_X, boosted_y

def fit(sensors: Dict[str, pd.DataFrame], events: Dict[str, pd.DatetimeIndex], local_clf, global_clf,
        nth: int, win_size: int, tol: pd.Timedelta) -> None:
    # Local step
    local_X = extract_nested(sensors, events, win_size).append(sample_nonevents(sensors, events, win_size))
    local_y = ["event"] * (len(local_X) / 2) + ["nonevent"] * (len(local_X) / 2)
    fit_local(local_clf, local_X, local_y)

    # Global step
    local_proba = predict_local(local_clf, sensors, nth, win_size)
    global_X = extract_peaks(local_proba)
    global_y = label_peaks(global_X, events, tol)
    fit_global(global_clf, global_X, global_y)

    # Boost
    global_pred = predict_global(global_clf, local_proba)
    outcomes = assess_predictions(global_pred, events, tol)
    boosted_X, boosted_y = boost(local_X, local_y, sensors, outcomes, win_size)
    fit_local(local_clf, boosted_X, boosted_y)
    local_proba2 = predict_local(local_clf, sensors, nth, win_size)
    global_X2 = extract_peaks(local_proba2)
    global_y2 = label_peaks(global_X2, events, tol)
    fit_global(global_clf, global_X2, global_y2)

def predict(sensors: Dict[str, pd.DataFrame], local_clf, global_clf, nth: int, win_size: int) -> Dict[str, pd.DatetimeIndex]:
    def _predict(_sensors: pd.DataFrame):
        local_proba = predict_local(local_clf, _sensors, nth, win_size)
        return predict_global(global_clf, local_proba)
    {deployid: _predict(s) for deployid, s in sensors.items()}
    