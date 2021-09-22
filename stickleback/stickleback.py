from sklearn import clone
from sklearn.model_selection import KFold
from stickleback.util import extract_all, extract_nested, sample_nonevents, extract_peaks
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class Stickleback:
    event = "event"
    nonevent = "nonevent"

    def __init__(self, local_clf, global_clf, win_size: int, tol: pd.Timedelta, nth: int = 1, n_folds: int = 5) -> None:
        self.local_clf = local_clf
        self.__local_clf2 = clone(local_clf)
        self.global_clf = global_clf
        self.win_size = win_size
        self.tol = tol
        self.nth = nth
        self.n_folds = n_folds

    def fit(self, sensors: Dict[str, pd.DataFrame], events: Dict[str, pd.DatetimeIndex], 
            mask: Dict[str, np.ndarray] = None) -> None:
        # Local step
        events_nested = extract_nested(sensors, events, self.win_size)
        event_X = pd.concat(events_nested.values())
        nonevents_nested = sample_nonevents(sensors, events, self.win_size, mask)
        nonevent_X = pd.concat(nonevents_nested.values())
        local_X = event_X.append(nonevent_X)
        event_y = np.full(len(event_X), 1.0)
        nonevent_y = np.full(len(nonevent_X), 0.0)
        local_y = np.concatenate([event_y, nonevent_y])
        self._fit_local(local_X, local_y)

        # Global step (using internal cross validation)
        n_folds = min(self.n_folds, len(sensors.keys()))
        kf = KFold(n_folds)
        deployids = np.array(list(sensors.keys()))
        global_X, global_y = [], []
        for train_idx, test_idx in kf.split(deployids):
            event_train_X = pd.concat([v for k, v in events_nested.items() if k in deployids[train_idx]])
            nonevent_train_X = pd.concat([v for k, v in nonevents_nested.items() if k in deployids[train_idx]])
            train_X = event_train_X.append(nonevent_train_X)
            train_y = np.concatenate([np.full(len(event_train_X), 1.0), np.full(len(nonevent_train_X), 0.0)])
            test_sensors = {k: v for k, v in sensors.items() if k in deployids[test_idx]}
            test_events = {k: v for k, v in events.items() if k in deployids[test_idx]}
            self._fit_local(train_X, train_y, clone=True)
            local_proba = self._predict_local(test_sensors, mask, clone=True)
            peaks = extract_peaks(local_proba)
            global_X.append(pd.concat(peaks.values()))
            peak_labels = self._label_peaks(peaks, test_events)
            global_y.append(pd.concat(peak_labels.values()))
        global_X, global_y = pd.concat(global_X), pd.concat(global_y)
        self._fit_global(global_X, global_y)

        # Boost
        global_pred = self._predict_global(local_proba)
        predictions = {d: (local_proba[d], global_pred[d]) for d in global_pred}
        outcomes = self.assess(predictions, events)
        boosted_X, boosted_y = self._boost(local_X, local_y, sensors, outcomes)
        self._fit_local(boosted_X, boosted_y)
        global_X2, global_y2 = [], []
        for train_idx, test_idx in kf.split(deployids):
            train_X, train_y = boosted_X[train_idx], boosted_y[train_idx]
            test_sensors = {k: v for k, v in sensors.items() if k in deployids[test_idx]}
            test_events = {k: v for k, v in events.items() if k in deployids[test_idx]}
            self._fit_local(train_X, train_y, clone=True)
            local_proba2 = self._predict_local(test_sensors, mask, clone=True)
            peaks2 = extract_peaks(local_proba2)
            global_X2.append(pd.concat(peaks2.values()))
            peak_labels2 = self._label_peaks(peaks2, test_events)
            global_y2.append(pd.concat(peak_labels2.values()))
        global_X2, global_y2 = pd.concat(global_X2), pd.concat(global_y2)
        self._fit_global(global_X2, global_y2)

    def predict(self, sensors: Dict[str, pd.DataFrame], mask: Dict[str, np.ndarray] = None) -> Dict[str, Tuple[pd.Series, pd.DataFrame]]:
        local_proba = self._predict_local(sensors, mask)
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
        predicted_times = {d: gbl.index[gbl["is_event"] == 1] for d, (_, gbl) in predicted.items()}
        return {deployid: _assess(predicted_times[deployid], events[deployid]) for deployid in predicted}

    def _fit_local(self, local_X: pd.DataFrame, local_y: np.ndarray, clone: bool = False) -> None:
        clf = self.__local_clf2 if clone else self.local_clf
        clf.fit(local_X, local_y)

    def _predict_local(self, sensors: Dict[str, pd.DataFrame], mask: Dict[str, np.ndarray] = None,
                       clone: bool = False) -> Dict[str, pd.Series]:
        clf = self.__local_clf2 if clone else self.local_clf
        X = extract_all(sensors, self.nth, self.win_size, mask)
        def _predict_local(_X: pd.DataFrame, i: pd.DatetimeIndex):
            return pd.Series(clf.predict_proba(_X)[:, 0], name="local_proba", index=_X.index) \
                .reindex(i) \
                .interpolate(method="cubic") \
                .fillna(0)
        return {deployid: _predict_local(X[deployid], sensors[deployid].index) for deployid in X}

    def _label_peaks(self, peaks: Dict[str, pd.DataFrame], events: Dict[str, pd.DatetimeIndex]) -> Dict[str, pd.Series]:
        def __label_peaks(_peaks: pd.DataFrame, _events: pd.DatetimeIndex) -> pd.Series:
            times = _peaks.index

            # Find the most prominent nearby peak (within tol)
            def find_nearest_peak(_event):
                near = np.where(np.logical_and(_event - self.tol <= times, times <= _event + self.tol))[0]
                near_prom = _peaks["prominences"][near]
                return near[np.argmax(near_prom)] if len(near_prom) > 0 else None
            
            nearest_peaks = [find_nearest_peak(e) for e in _events]
            nearest_peaks = [i for i in nearest_peaks if i is not None]
            tps = times[nearest_peaks]
            outcomes = pd.Series(0, index=times)
            outcomes[tps] = 1
            return outcomes
        return {d: __label_peaks(peaks[d], events[d]) for d in peaks}

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
    