from stickleback.util import extract_all, extract_nested, sample_nonevents, extract_peaks
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from ipdb import set_trace

class Stickleback:
    event = "event"
    nonevent = "nonevent"

    def __init__(self, local_clf, global_clf, win_size: int, tol: pd.Timedelta, nth: int = 1) -> None:
        self.local_clf = local_clf
        self.global_clf = global_clf
        self.win_size = win_size
        self.tol = tol
        self.nth = nth

    def fit(self, sensors: Dict[str, pd.DataFrame], events: Dict[str, pd.DatetimeIndex]) -> None:
        # Local step
        event_X = pd.concat(extract_nested(sensors, events, self.win_size).values())
        nonevent_X = pd.concat(sample_nonevents(sensors, events, self.win_size).values())
        local_X = event_X.append(nonevent_X)
        event_y = np.full(len(event_X), Stickleback.event)
        nonevent_y = np.full(len(nonevent_X), Stickleback.nonevent)
        local_y = np.concatenate([event_y, nonevent_y])
        self._fit_local(local_X, local_y)

        # Global step
        local_proba = self._predict_local(sensors)
        peaks = extract_peaks(local_proba)
        global_X = pd.concat(peaks.values())
        peak_labels = self._label_peaks(peaks, events)
        global_y = pd.concat(peak_labels.values())
        self._fit_global(global_X, global_y)

        # Boost
        global_pred = self._predict_global(local_proba)
        outcomes = self.assess(global_pred, events)
        boosted_X, boosted_y = self._boost(local_X, local_y, sensors, outcomes)
        self._fit_local(boosted_X, boosted_y)
        local_proba2 = self._predict_local(sensors)
        peaks2 = extract_peaks(local_proba2)
        global_X2 = pd.concat(peaks2.values())
        peak_labels2 = self._label_peaks(peaks2, events)
        global_y2 = pd.concat(peak_labels2.values())
        self._fit_global(global_X2, global_y2)

    def predict(self, sensors: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[pd.Series, pd.DatetimeIndex]]:
        local_proba = self._predict_local(sensors)
        global_pred = self._predict_global(local_proba)
        return {d: (local_proba[d], global_pred[d]) for d in global_pred}

    def assess(self, predicted: Dict[str, Tuple[pd.Series, pd.DatetimeIndex]], events: Dict[str, pd.DatetimeIndex]) -> Dict[str, pd.Series]:
        def _assess(_predicted: pd.DatetimeIndex, _events: pd.DatetimeIndex) -> pd.Series:
            # Find closest predicted to each actual and their distance
            # TODO handle no predicted events case
            closest = _predicted[[np.argmin(np.abs(_predicted - e)) for e in _events]]
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
        predicted_times = {deployid: p[1].index[p[1]["is_event"] == 1] for deployid, p in predicted.items()}
        return {deployid: _assess(predicted_times[deployid], events[deployid]) for deployid in predicted}

    def _fit_local(self, local_X: pd.DataFrame, local_y: np.ndarray) -> None:
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

    def _predict_global(self, local_proba: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
        peaks = extract_peaks(local_proba)
        def predict_peaks(_peaks: pd.DataFrame):
            result = pd.DataFrame({"global_proba": self.global_clf.predict_proba(_peaks)[:, 0],
                                   "is_event": self.global_clf.predict(_peaks)},
                                   index = _peaks.index)
            return result
        return {deployid: predict_peaks(p) for deployid, p in peaks.items()}
        
    def _boost(self, local_X: pd.DataFrame, local_y: np.ndarray, sensors: Dict[str, pd.DataFrame], 
               outcomes: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, list]:
        fps = {d: o.index[o == "FP"] for d, o in outcomes.items()}
        nfps = np.sum([len(f) for f in fps.values()])

        boosted_X = local_X.append(pd.concat(extract_nested(sensors, fps, self.win_size).values()))
        boosted_y = np.concatenate([local_y, np.full(nfps, Stickleback.nonevent)])
        return boosted_X, boosted_y
    