from stickleback.util import extract_all, extract_nested, sample_nonevents, extract_peaks
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class Stickleback:
    event = "event"
    nonevent = "nonevent"

    def __init__(self, local_clf, global_clf, win_size: int, tol: pd.Timedelta, nth: int = 1) -> None:
        self.local_clf = local_clf
        self.global_clf = global_clf
        self.win_size = win_size
        self.tol = tol
        self.nth = nth

    def _fit_local(self, local_X: pd.DataFrame, local_y: list) -> None:
        self.local_clf.fit(local_X, local_y)

    def _predict_local(self, sensors: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        X = extract_all(sensors, self.nth, self.win_size)
        def _predict_local(_X: pd.DataFrame, i: pd.DatetimeIndex):
            return pd.Series(self.local_clf.predict_proba(_X)[:, 0], name="local_proba", index=_X.index) \
                .reindex(i) \
                .interpolate(method="cubic") \
                .fillna(0)
        return {deployid: _predict_local(X[deployid], sensors[deployid].index) for deployid in X}

    def _label_peaks(self, peaks: Dict[str, pd.DataFrame], events: Dict[str, pd.DatetimeIndex]) -> Dict[str, pd.Series]:
        def _label_peaks(_peaks: pd.DataFrame, _events: pd.DatetimeIndex) -> pd.Series:
            times = _peaks.index

            def find_nearest_peak(_event):
                near = np.where(np.logical_and(_event - self.tol <= times, times <= _event + self.tol))[0]
                near_heights = _peaks["peak_heights"][near]
                return near[np.argmax(near_heights)] if len(near_heights) > 0 else None
            
            nearest_peaks = [find_nearest_peak(e) for e in _events]
            nearest_peaks = [i for i in nearest_peaks if i is not None]
            tps = times[nearest_peaks]
            outcomes = pd.Series(0, index=times)
            outcomes[tps] = 1
            return outcomes
        return {d: _label_peaks(peaks[d], events[d]) for d in peaks}

    def _fit_global(self, global_X: Dict[str, pd.DataFrame], global_y: Dict[str, pd.Series]) -> None:
        self.global_clf.fit(global_X, global_y)

    def _predict_global(self, local_proba: Dict[str, pd.Series]) -> Dict[str, pd.DatetimeIndex]:
        peaks = extract_peaks(local_proba)
        return {deployid: self.global_clf.predict(peaks[deployid]) for deployid in local_proba}

    def _assess_predictions(self, predicted: Dict[str, pd.DatetimeIndex], events: Dict[str, pd.DatetimeIndex]) -> Dict[str, pd.Series]:
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
                if d <= self.tol:
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
        
    def _boost(self, local_X: pd.DataFrame, local_y: list, sensors: Dict[str, pd.DataFrame], 
               outcomes: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, list]:
        fps = {d: o.index[o == "FP"] for d, o in outcomes.items()}
        nfps = np.sum([len(f) for f in fps.values()])
        boosted_X = local_X.append(extract_nested(sensors, fps, self.win_size))
        boosted_y = local_y.append([Stickleback.nonevent] * nfps)
        return boosted_X, boosted_y

    def fit(self, sensors: Dict[str, pd.DataFrame], events: Dict[str, pd.DatetimeIndex]) -> None:
        # Local step
        local_X = extract_nested(sensors, events, self.win_size) \
            .append(sample_nonevents(sensors, events, self.win_size))
        local_y = [Stickleback.event] * (len(local_X) / 2) + [Stickleback.nonevent] * (len(local_X) / 2)
        self._fit_local(local_X, local_y)

        # Global step
        local_proba = self._predict_local(sensors)
        global_X = extract_peaks(local_proba)
        global_y = self._label_peaks(global_X, events)
        self._fit_global(global_X, global_y)

        # Boost
        global_pred = self._predict_global(local_proba)
        outcomes = self._assess_predictions(global_pred, events)
        boosted_X, boosted_y = self._boost(local_X, local_y, sensors, outcomes)
        self._fit_local(boosted_X, boosted_y)
        local_proba2 = self._predict_local(sensors)
        global_X2 = extract_peaks(local_proba2)
        global_y2 = self._label_peaks(global_X2, events)
        self._fit_global(global_X2, global_y2)

    def predict(self, sensors: Dict[str, pd.DataFrame]) -> Dict[str, pd.DatetimeIndex]:
        def _predict(_sensors: pd.DataFrame):
            return self._predict_global(self._predict_local(_sensors))
        return {deployid: _predict(s) for deployid, s in sensors.items()}
    