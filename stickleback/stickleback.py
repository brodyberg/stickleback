import scipy.signal as signal 
import sklearn
import sklearn.model_selection as selection 
import sktime.datatypes._panel._convert as convert 
from stickleback.types import *
import stickleback.util as sb_util
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from pdb import set_trace

class Stickleback:

    def __init__(self, 
                 local_clf, 
                 win_size: int, 
                 tol: pd.Timedelta, 
                 nth: int = 1, 
                 n_folds: int = 5, 
                 max_events: int = None) -> None:
        self.local_clf = local_clf
        self.__local_clf2 = sklearn.clone(local_clf)
        self.prominence = None
        self.win_size = win_size
        self.tol = tol
        self.nth = nth
        self.n_folds = n_folds
        self.max_events = max_events

    def fit(self, 
            sensors: sensors_T, 
            events: events_T, 
            mask: mask_T = None) -> None:
        # Local step
        if self.max_events is None:
            training_events = events
        else:
            n_events = np.array([len(v) for v in events.values()])
            if n_events.sum() <= self.max_events:
                training_events = events
            else:
                rg = np.random.Generator(np.random.PCG64())
                keep = self.max_events / n_events.sum()
                training_events = {k: rg.choice(v, 
                                                size=int(len(v)*keep), 
                                                replace=False) 
                                   for k, v in events.items()}
                
        events_nested = sb_util.extract_nested(sensors, 
                                               training_events, 
                                               self.win_size)
        event_X = pd.concat(events_nested.values())
        nonevents_nested = sb_util.sample_nonevents(sensors, 
                                            training_events, 
                                            self.win_size,
                                            mask)
        nonevent_X = pd.concat(nonevents_nested.values())
        local_X = event_X.append(nonevent_X)
        event_y = np.full(len(event_X), 1.0)
        nonevent_y = np.full(len(nonevent_X), 0.0)
        local_y = np.concatenate([event_y, nonevent_y])
        self._fit_local(local_X, local_y)

        # Global step (using internal cross validation)
        local_proba_cv = self._fit_global(events_nested, 
                                          nonevents_nested, 
                                          sensors, 
                                          events, 
                                          mask)

        # Boost
        predictions_gbl = self._predict_global(local_proba_cv)
        predictions = {k: (None, v) for k, v in predictions_gbl.items()}
        outcomes = self.assess(predictions, events)
        boosted_nonevents = self._boost(nonevents_nested, sensors, outcomes)
        n_nonevents = np.sum([len(v) for v in boosted_nonevents.values()])
        boosted_X = event_X.append(pd.concat(boosted_nonevents.values()))
        boosted_y = np.concatenate([event_y, np.full(n_nonevents, 0.0)])
        self._fit_local(boosted_X, boosted_y)
        self._fit_global(events_nested, 
                         boosted_nonevents, 
                         sensors, 
                         events, 
                         mask)

    def predict(self, 
                new_sensors: sensors_T, 
                new_mask: mask_T = None) -> prediction_T:
        local_proba = self._predict_local(new_sensors, new_mask)
        global_pred = self._predict_global(local_proba)
        return {d: (local_proba[d], global_pred[d]) for d in global_pred}

    def assess(self, predicted: prediction_T, events: events_T) -> outcomes_T:
        result = dict()
        pred_gbl = {k: v[1] for k, v in predicted.items()}
        for deployid in pred_gbl:
            _predicted, _actual = pred_gbl[deployid], events[deployid]
            # Find closest predicted to each actual and their distance
            # TODO handle no predicted events case
            closest = _predicted[[np.argmin(np.abs(_predicted - e)) 
                                  for e in _actual]]
            distance = np.abs(_actual - closest)
            
            # Initialize outcomes
            outcomes = pd.Series(index=_predicted, 
                                 dtype="string", 
                                 name="outcome")

            # Iterate through actual events. The closest predicted event within
            # the tolerance is a true positive. If no predicted events are
            # within the tolerance, the actual event is a false negative.
            for i, (c, d) in enumerate(zip(closest, distance)):
                if d <= self.tol:
                    outcomes[c] = "TP" 
                else:
                    outcomes[_actual[i]] = "FN"

            # Remaining predictions are false positives
            outcomes[outcomes.isna()] = "FP"

            result[deployid] = outcomes
            
        return result

    def _fit_local(self, 
                   local_X: pd.DataFrame, 
                   local_y: np.ndarray, 
                   clone: bool = False) -> None:
        if len(local_X.columns) > 1:
            local_X = convert.from_nested_to_3d_numpy(local_X)

        clf = self.__local_clf2 if clone else self.local_clf
        clf.fit(local_X, local_y)

    def _predict_local(self, 
                       sensors: sensors_T, 
                       mask: mask_T = None,
                       clone: bool = False) -> pred_lcl_T:
        clf = self.local_clf if not clone else self.__local_clf2
        X = sb_util.extract_all(sensors, self.nth, self.win_size, mask)

        def _predict_local(_X: pd.DataFrame, 
                           i: pd.DatetimeIndex, 
                           _mask: np.ndarray):
            local = pd.Series(clf.predict_proba(_X)[:, 1], 
                              name="local_proba", 
                              index=_X.index). \
                    reindex(i)
            local[np.logical_not(_mask)] = 0.
            return local.interpolate(method="cubic").fillna(0)
        
        if mask is None:
            mask = {k: np.full(len(v), True) for k, v in sensors.items()}
        
        return {deployid: _predict_local(X[deployid],
                                         sensors[deployid].index, 
                                         mask[deployid]) 
                for deployid in X}

    def _fit_global(self, 
                    events_nested: nested_T, 
                    nonevents_nested: nested_T, 
                    sensors: sensors_T, 
                    events: events_T, 
                    mask: mask_T) -> pred_lcl_T:
        # Global step (using internal cross validation)
        n_folds = min(self.n_folds, len(sensors.keys()))
        kf = selection.KFold(n_folds)
        deployids = np.array(list(sensors.keys()))
        local_proba = dict()
        for train_idx, test_idx in kf.split(deployids):
            event_train_X = pd.concat([v 
                                       for k, v in events_nested.items() 
                                       if k in deployids[train_idx]])
            nonevent_train_X = pd.concat([v 
                                          for k, v in nonevents_nested.items() 
                                          if k in deployids[train_idx]])
            train_X = event_train_X.append(nonevent_train_X)
            train_y = np.concatenate([np.full(len(event_train_X), 1.0), 
                                      np.full(len(nonevent_train_X), 0.0)])
            test_sensors = sb_util.filter_dict(sensors, deployids[test_idx])
            self._fit_local(train_X, train_y, clone=True)
            local_proba.update(
                self._predict_local(test_sensors, mask, clone=True)
            )
            
        prominence = np.linspace(0, 1, 25)
        f1 = np.zeros(prominence.shape)
        for i, p in enumerate(prominence):
            peaks = dict()
            for k, v in local_proba.items():
                idx, _ = signal.find_peaks(v.fillna(0), prominence=p)
                peaks[k] = (v, v.index[idx])
                
            outcomes = self.assess(peaks, events)
            tps = fps = fns = 0
            for o in outcomes.values():
                tps += (o == "TP").sum()
                fps += (o == "FP").sum()
                fns += (o == "FN").sum()
                
            f1[i] = tps / (tps + (fps + fns) / 2)
            
        self.prominence = prominence[np.argmax(f1)]
        return local_proba

    def _predict_global(self, local_proba: pred_lcl_T) -> pred_gbl_T:
        peaks = dict()
        for k, v in local_proba.items():
            idx, _ = signal.find_peaks(v.fillna(0), prominence=self.prominence)
            peaks[k] = v.index[idx]
            
        return peaks
        
    def _boost(self, 
               nonevents: nested_T, 
               sensors: sensors_T, 
               outcomes: outcomes_T) -> nested_T:
        fps = {d: o.index[o == "FP"] for d, o in outcomes.items()}
        nested_fps = sb_util.extract_nested(sensors, fps, self.win_size)
        boosted_nonevents = dict()
        for k in nonevents:
            if k in nested_fps: 
                boosted_nonevents[k] = pd.concat([nonevents[k], nested_fps[k]])
            else:
                boosted_nonevents[k] = nonevents[k]
                
        return boosted_nonevents
    