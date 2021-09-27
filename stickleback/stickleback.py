from scipy.signal import find_peaks
from sklearn import clone
from sklearn.model_selection import KFold
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from stickleback.util import extract_all, extract_nested, sample_nonevents, extract_peaks, filter_dict
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from pdb import set_trace

class Stickleback:

    def __init__(self, local_clf, win_size: int, tol: pd.Timedelta, nth: int = 1, n_folds: int = 5) -> None:
        self.local_clf = local_clf
        self.__local_clf2 = clone(local_clf)
        self.prominence = None
        self.win_size = win_size
        self.tol = tol
        self.nth = nth
        self.n_folds = n_folds

    def fit(self, sensors: Dict[str, pd.DataFrame], events: Dict[str, pd.DatetimeIndex], 
            mask: Dict[str, np.ndarray] = None, max_events: int = None) -> None:
        # Local step
        if max_events is None:
            training_events = events
        else:
            n_events = np.array([len(v) for v in events.values()])
            if n_events.sum() <= max_events:
                training_events = events
            else:
                rg = np.random.Generator(np.random.PCG64())
                keep = max_events / n_events.sum()
                training_events = {k: rg.choice(v, size=int(len(v)*keep), replace=False) 
                                   for k, v in events.items()}
        events_nested = extract_nested(sensors, training_events, self.win_size)
        event_X = pd.concat(events_nested.values())
        nonevents_nested = sample_nonevents(sensors, training_events, self.win_size, mask)
        nonevent_X = pd.concat(nonevents_nested.values())
        local_X = event_X.append(nonevent_X)
        event_y = np.full(len(event_X), 1.0)
        nonevent_y = np.full(len(nonevent_X), 0.0)
        local_y = np.concatenate([event_y, nonevent_y])
        self._fit_local(local_X, local_y)

        # Global step (using internal cross validation)
        local_proba_cv = self._fit_global(events_nested, nonevents_nested, sensors, events, mask)

        # Boost
        predictions = self._predict_global(local_proba_cv)
        outcomes = self._assess(predictions, events)
        boosted_nonevents = self._boost(nonevents_nested, sensors, outcomes)
        n_nonevents = np.sum([len(v) for v in boosted_nonevents.values()])
        boosted_X = event_X.append(pd.concat(boosted_nonevents.values()))
        boosted_y = np.concatenate([event_y, np.full(n_nonevents, 0.0)])
        self._fit_local(boosted_X, boosted_y)
        self._fit_global(events_nested, boosted_nonevents, sensors, events, mask)

    def predict(self, sensors: Dict[str, pd.DataFrame], mask: Dict[str, np.ndarray] = None) -> Dict[str, Tuple[pd.Series, pd.DatetimeIndex]]:
        local_proba = self._predict_local(sensors, mask)
        global_pred = self._predict_global(local_proba)
        return {d: (local_proba[d], global_pred[d]) for d in global_pred}

    def assess(self, predicted: Dict[str, Tuple[pd.Series, pd.DatetimeIndex]], events: Dict[str, pd.DatetimeIndex]) -> Dict[str, pd.Series]:
        predicted2 = {k: v[1] for k, v in predicted.items()}
        return self._assess(predicted2, events)

    def _assess(self, predicted: Dict[str, pd.DatetimeIndex], events: Dict[str, pd.DatetimeIndex]) -> Dict[str, pd.Series]:
        result = dict()
        for deployid in predicted:
            _predicted, _actual = predicted[deployid], events[deployid]
            # Find closest predicted to each actual and their distance
            # TODO handle no predicted events case
            closest = _predicted[[np.argmin(np.abs(_predicted - e)) for e in _actual]]
            distance = np.abs(_actual - closest)
            
            # Initialize outcomes
            outcomes = pd.Series(index=_predicted, dtype="string", name="outcome")
            
            # Iterate through actual events. The closest predicted event within the tolerance is a true positive. If no
            # predicted events are within the tolerance, the actual event is a false negative.
            for i, (c, d) in enumerate(zip(closest, distance)):
                if d <= self.tol:
                    outcomes[c] = "TP" 
                else:
                    outcomes[_actual[i]] = "FN"

            # Iterate through predicted events. Any predicted events that aren't the closest to an actual event are false
            # positives.
            for i, p in enumerate(_predicted):
                if p not in closest:
                    outcomes[p] = "FP" 

            result[deployid] = outcomes
        return result

    def _fit_local(self, local_X: pd.DataFrame, local_y: np.ndarray, clone: bool = False) -> None:
        if len(local_X.columns) > 1:
            local_X = from_nested_to_3d_numpy(local_X)

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

    def _fit_global(self, events_nested, nonevents_nested, sensors, events, mask) -> Dict[str, pd.Series]:
        # Global step (using internal cross validation)
        n_folds = min(self.n_folds, len(sensors.keys()))
        kf = KFold(n_folds)
        deployids = np.array(list(sensors.keys()))
        local_proba = dict()
        for train_idx, test_idx in kf.split(deployids):
            event_train_X = pd.concat([v for k, v in events_nested.items() if k in deployids[train_idx]])
            nonevent_train_X = pd.concat([v for k, v in nonevents_nested.items() if k in deployids[train_idx]])
            train_X = event_train_X.append(nonevent_train_X)
            train_y = np.concatenate([np.full(len(event_train_X), 1.0), np.full(len(nonevent_train_X), 0.0)])
            test_sensors = filter_dict(sensors, deployids[test_idx])
            self._fit_local(train_X, train_y, clone=True)
            local_proba.update(self._predict_local(test_sensors, mask, clone=True))
        prominence = np.linspace(0, 1, 25)
        f1 = np.zeros(prominence.shape)
        for i, p in enumerate(prominence):
            peaks = dict()
            for k, v in local_proba.items():
                idx, _ = find_peaks(v.fillna(0), prominence=p)
                peaks[k] = v.index[idx]
            outcomes = self._assess(peaks, events)
            tps = fps = fns = 0
            for o in outcomes.values():
                tps += (o == "TP").sum()
                fps += (o == "FP").sum()
                fns += (o == "FN").sum()
            f1[i] = tps / (tps + (fps + fns) / 2)
        self.prominence = prominence[np.argmax(f1)]
        return local_proba

    def _predict_global(self, local_proba: Dict[str, pd.Series]) -> Dict[str, pd.DatetimeIndex]:
        peaks = dict()
        for k, v in local_proba.items():
            idx, _ = find_peaks(v.fillna(0), prominence=self.prominence)
            peaks[k] = v.index[idx]
        return peaks
        
    # boosted_nonevents = self._boost(nonevents_nested, sensors, outcomes)
    def _boost(self, nonevents: Dict[str, pd.DataFrame], sensors: Dict[str, pd.DataFrame], 
               outcomes: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, list]:
        fps = {d: o.index[o == "FP"] for d, o in outcomes.items()}
        nested_fps = extract_nested(sensors, fps, self.win_size)
        boosted_nonevents = dict()
        for k in nonevents:
            if k in nested_fps: 
                boosted_nonevents[k] = pd.concat([nonevents[k], nested_fps[k]])
            else:
                boosted_nonevents[k] = nonevents[k]
        return boosted_nonevents
    