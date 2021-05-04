from functools import reduce
from matplotlib.figure import Figure as matplotlibFigure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure as plotlyFigure
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sktime.classification.compose import TimeSeriesForestClassifier, ColumnEnsembleClassifier
from sktime.utils.data_processing import from_3d_numpy_to_nested
from typing import Dict, Tuple, Union

from ipdb import set_trace

# Utility functions
def _diff_from(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Return the array-wise least difference from another array

        Parameters:
            xs: the basis array
            ys: the target array
            
        Returns:
            The minimum absolute difference for each element of xs from the closest value in ys
    """
    np.zeros_like
    return np.array([np.min(np.abs(x - ys)) for x in xs])

def _extract_peaks(x: pd.Series) -> pd.DataFrame:
    peak_idxs, peak_props = find_peaks(x.fillna(0), height=0.25, prominence=0.01, width=1, rel_height=0.5)
    peaks = pd.DataFrame(peak_props, index=x.index[peak_idxs])
    peaks = peaks[["peak_heights", "prominences", "widths"]]
    return peaks

class Stickleback:
    event = "event"
    nonevent = "nonevent"

    """Identify point behaviors in longitudinal sensor data.
    """

    def __init__(self, sensors: Dict[str, pd.DataFrame], events: Dict[str, pd.Series], win_size: int, seed: int = None, 
                 proba_thr: float = 0.5, proba_prom: float = 0.25):
        """Instantiate a Stickleback object 

        Args:
            sensors (pd.DataFrame): Longitudinal sensor data, e.g. depth, pitch, and jerk.
            events (pd.MultiIndex): Index tuples (ID, times of point behavior events).
            win_size (int): Size of sliding window (in records).
            seed (int, optional): Random number generator seed. Defaults to None.
            min_period (int, optional): Minimum period between events (in records). Defaults to 1.
            proba_thr (float, optional): Probability threshold for classifying events. Defaults to 0.5.
            proba_prom (float, optional): Prominence of probability peak for classifying events. Defaults to 0.25.
        """
        self.sensors = sensors
        self.events = events
        self.win_size = win_size
        self.rg = np.random.Generator(np.random.PCG64(seed))

        # Workflow flags
        self.nonevents_sampled = False
        self.train_extracted = False
        self.fitted = False

        # Classifier attributes
        self.event_idx = {i: sensors[i].index[sensors[i].index.searchsorted(e)] for i, e in events.items()} 
        self.nonevent_idx = dict()
        self.local_data = pd.DataFrame()
        self.local_labels = []
        self.clf_local = ColumnEnsembleClassifier(
            estimators=[("TSF_" + c, TimeSeriesForestClassifier(n_estimators=100, class_weight="balanced"), [i]) 
                        for i, c in enumerate(next(iter(sensors.values())).columns)]
        )
        self.global_data = pd.DataFrame()
        self.global_labels = []
        self.clf_global = LogisticRegression()

        # Predictions
        self.proba_thr = proba_thr
        self.proba_prom = proba_prom

    def sample_nonevents(self) -> None:
        """Sample non-events for model training.
        """
        for deployid in self.events.keys():
            sensors = self.sensors[deployid]
            event_idx = self.event_idx[deployid]

            # Valid indices for negatives
            nonevent_choices = np.array(range(self.win_size, len(sensors) - self.win_size - 1, self.win_size))
            diff_from_event = _diff_from(nonevent_choices, sensors.index.searchsorted(event_idx))
            nonevent_choices = nonevent_choices[diff_from_event > self.win_size]

            # Randomly choose nonevents
            nonevent_idx = self.rg.choice(nonevent_choices, size=len(event_idx), replace=False)
            nonevent_idx.sort()
            self.nonevent_idx[deployid] = sensors.index[nonevent_idx]

        self.nonevents_sampled = True
    
    def _extract_nested(self, idx: Dict[str, pd.DatetimeIndex]) -> pd.DataFrame:
        """Extract windows of data in nested sktime DataFrame format

        Args:
            idx (pd.MultiIndex): Indices of window centers.

        Returns:
            pd.DataFrame: Extracted windows in nested sktime DataFrame format.
        """
        win_size_2 = int(self.win_size / 2)
        nested_list = []
        for deployid, times in idx.items():
            sensors = self.sensors[deployid]
            timeidx = np.array([sensors.index.get_loc(t) for t in times])
            out_of_bounds = timeidx[np.logical_or(timeidx < win_size_2, timeidx >= len(sensors) - win_size_2)]
            assert len(out_of_bounds) == 0, \
                   "Index {} out of bounds in {} (and possibly more)".format(out_of_bounds[0], deployid)
            data_3d = np.empty([len(timeidx), len(sensors.columns), self.win_size], float)
            data_arr = sensors.to_numpy().transpose()
            start_idx = timeidx - win_size_2
            for i, start in enumerate(start_idx):
                data_3d[i] = data_arr[:, start:(start + self.win_size)]
            nested = from_3d_numpy_to_nested(data_3d)
            nested.columns = sensors.columns
            nested.index = times
            nested_list.append(nested)

        return pd.concat(nested_list)

    def _extract_all(self, sensors: pd.DataFrame, nth: int):
        win_size_2 = int(self.win_size / 2)
        
        idx = np.arange(win_size_2, len(sensors) - win_size_2, nth)
        data_3d = np.empty([len(idx), len(sensors.columns), self.win_size], float)
        data_arr = sensors.to_numpy().transpose()
        start_idx = idx - win_size_2
        for i, start in enumerate(start_idx):
            data_3d[i] = data_arr[:, start:(start + self.win_size)]
        
        nested = from_3d_numpy_to_nested(data_3d)
        nested.columns = sensors.columns
        nested.index = sensors.index[idx]
        
        return nested

    def extract_training_data(self):
        """Extract training data from longitudinal sensor data
        """
        assert self.nonevents_sampled, "Can't extract training data until nonevents sampled"
        nested_events = self._extract_nested(self.event_idx)
        nested_nonevents = self._extract_nested(self.nonevent_idx)
        self.local_data = pd.concat([nested_events, nested_nonevents])
        self.local_labels = [Stickleback.event] * len(nested_events) + [Stickleback.nonevent] * len(nested_nonevents)
        self.train_extracted = True

    def fit_local(self) -> None:
        """Fit classifier to training data
        """
        assert self.train_extracted, "Can't fit classifier until training data extracted"
        self.clf_local.fit(self.local_data, self.local_labels)

    def fit_global(self, nth: int, tol: pd.Timedelta) -> None:
        assert self.clf_local.is_fitted, "Fit local classifier before global classifier"
        local_proba = {deployid: self._predict_local(self.sensors[deployid], nth) 
                       for deployid in self.sensors}
        peaks = {deployid: _extract_peaks(prob) for deployid, prob in local_proba.items()}
        outcomes = [self._assess_peaks(deployid, p, tol) for deployid, p in peaks.items()]

        self.global_data = pd.concat(peaks.values())
        self.global_labels = pd.concat(outcomes)
        self.clf_global.fit(self.global_data, self.global_labels)

    def _predict_local(self, sensors: pd.DataFrame, nth: int) -> pd.Series:
        nested = self._extract_all(sensors, nth)
        return pd.Series(self.clf_local.predict_proba(nested)[:, 0], 
                         name="local_proba", index=nested.index) \
            .reindex(sensors.index) \
            .interpolate(method="cubic") \
            .fillna(0)

    def _predict_global(self, local_proba: pd.Series) -> pd.DataFrame:
        peaks = _extract_peaks(local_proba)
        peak_proba = self.clf_global.predict_proba(peaks)[:, 1]
        event_proba = pd.DataFrame(local_proba)
        event_proba.loc[peaks.index, "global_proba"] = peak_proba
        event_proba["predicted"] = 0
        event_proba.loc[peaks.index[peak_proba >= 0.5], "predicted"] = 1
        return event_proba

    def _predict(self, sensors: pd.DataFrame, nth: int) -> Tuple[pd.Series, pd.DatetimeIndex]:
        local_proba = self._predict_local(sensors, nth)
        return self._predict_global(local_proba)

    def predict_self(self, nth: int = 1) -> Dict[str, pd.DataFrame]:
        """Predict in-sample events

        Args:
            nth (int, optional): Predict every nth window and interpolate probabilities in between. Defaults to 1.
        """
        return {deployid: self._predict(sensors, nth) for deployid, sensors in self.sensors.items()}

    def _assess(self, predicted: pd.DatetimeIndex, actual: pd.DatetimeIndex, tol: pd.Timedelta) -> pd.Series:
        # Find closest predicted to each actual and their distance
        # TODO handle no predicted events case
        closest = predicted[[np.argmin(np.abs(predicted - a)) for a in actual]]
        distance = np.abs(actual - closest)
        
        # Initialize outcomes
        outcomes = pd.Series(index=predicted, dtype="string", name="outcome")
        
        # Iterate through actual events. The closest predicted event within the tolerance is a true positive. If no
        # predicted events are within the tolerance, the actual event is a false negative.
        for i, (c, d) in enumerate(zip(closest, distance)):
            if d <= tol:
                outcomes[c] = "TP" 
            else:
                outcomes[actual[i]] = "FN"

        # Iterate through predicted events. Any predicted events that aren't the closest to an actual event are false
        # positives.
        for i, p in enumerate(predicted):
            if p not in closest:
                outcomes[p] = "FP" 

        return outcomes

    def _assess_peaks(self, deployid: str, peaks: pd.DataFrame, tol: pd.Timedelta) -> pd.Series:
        actual = self.event_idx[deployid]
        peak_times = peaks.index

        def find_near_peak(a, peak_times, peak_heights, tol):
            near = np.where(np.logical_and(a - tol <= peak_times, peak_times <= a + tol))[0]
            heights = peak_heights[near]
            return near[np.argmax(heights)] if len(heights) > 0 else None
        
        near_idx = [find_near_peak(a, peak_times, peaks["peak_heights"], tol) for a in actual]
        near_idx = [i for i in near_idx if i is not None]
        tps = peak_times[near_idx]
        outcomes = pd.Series(0, index=peak_times)
        outcomes[tps] = 1
        return outcomes
        
    def assess_self(self, predicted: Dict[str, pd.DataFrame], tol: pd.Timedelta) -> Dict[str, pd.Series]:
        """Assess prediction accuracy
        
        The closest predicted event to each actual event (within tolerance) is considered a true positive. Predicted 
        events that are not the closest prediction to an actual event (or farther than tolerance) are considered false
        positives. Actual events with no predicted event within the tolerance are considered false negatives.

        Args:
            deployid (str): deployment identifier
            pred_idx (pd.MultiIndex): indices of predicted events (deployid, timestamp)
            tol (int, optional): Tolerance for linking predicted and actual events (in records). Defaults to 1.
        """ 
        predicted_idx = {deployid: predicted[deployid].index[predicted[deployid]["predicted"] == 1] 
                         for deployid in predicted}
        outcomes = {deployid: self._assess(pred, self.event_idx[deployid], tol) 
                    for deployid, pred in predicted_idx.items()}
        return outcomes

    def refit(self, false_positives: Dict[str, pd.DatetimeIndex], nth: int, tol: pd.Timedelta) -> None:
        """Refit model

        Adds data to training dataset and re-fits classifer.
        """
        self.local_data = self.local_data.append(self._extract_nested(false_positives))
        n_fps = reduce(lambda a, b: len(a) + len(b), false_positives.values())
        self.local_labels += [Stickleback.nonevent] * n_fps
        self.fit_local()
        self.fit_global(nth, tol)

    def loo(self, nth: int, tol: pd.Timedelta):
        loo_pred = dict()
        for deployid in self.events.unique(0):
            loo_sensors = self.sensors.loc[self.sensors.index.get_level_values(0) != deployid]
            loo_sb = Stickleback(
                sensors=self.sensors.loc[self.sensors.index.get_level_values(0) != deployid],
                events=self.events[self.events.get_level_values(0) != deployid],
                win_size=self.win_size, min_period=self.min_period,
                proba_thr=self.proba_thr, proba_prom=self.proba_prom
            )
            loo_sb.sample_nonevents()
            loo_sb.extract_training_data()
            loo_sb.fit_local()
            loo_proba, loo_idx = loo_sb.predict_self(nth)
            loo_outcomes = loo_sb.assess(loo_idx, tol)
            loo_fp = loo_outcomes[loo_outcomes == "FP"].index
            loo_sb.refit(loo_fp, [Stickleback.nonevent] * len(loo_fp))
            loo_proba2, loo_idx2 = loo_sb.predict_self(nth)
            loo_outcomes2 = loo_sb.assess(loo_idx2, tol)
            loo_pred[deployid] = loo_outcomes2
        return loo_pred
    
    def plot_sensors_events(self, deployid, interactive=False) -> Union[plotlyFigure, matplotlibFigure]:
        """Plot longitudinal sensor data and events

        Returns:
            Figure: a plotly figure with sensor data in subplots and events marked with points.
        """
        assert(deployid in self.sensors.keys())
        sensors = self.sensors[deployid]
        event_sensors = sensors.loc[self.event_idx[deployid]]
        
        if interactive:
            return self.__plot_sensors_events_interactive(sensors, event_sensors)
        else:
            return self.__plot_sensors_events_static(sensors, event_sensors)

    def __plot_sensors_events_interactive(self, sensors, event_sensors) -> plotlyFigure:
        fig = make_subplots(rows=len(sensors.columns), cols=1,
                            shared_xaxes=True,)
        for i, col in enumerate(sensors.columns):
            fig.append_trace(go.Scatter(
                x=sensors.index,
                y=sensors[col],
                mode="lines"
            ), row=i + 1, col=1)
            fig.append_trace(go.Scatter(
                x=event_sensors.index,
                y=event_sensors[col],
                mode="markers"
            ), row=i + 1, col=1)
            fig.update_yaxes(title_text=col, row=i + 1, col=1)
            
        fig.update_layout(showlegend=False)
        return fig

    def __plot_sensors_events_static(self, sensors, event_sensors) -> matplotlibFigure:
        fig, axs = plt.subplots(len(sensors.columns), 1)
        for i, col in enumerate(sensors.columns):
            # sensor data
            axs[i].plot(sensors.index, sensors[col], "-", zorder=1)
            # events
            axs[i].scatter(event_sensors.index, event_sensors[col], facecolors="none", edgecolors="r", zorder=2)
            axs[i].set_ylabel(col)
            if col == "depth":
                axs[i].invert_yaxis()
            
        return fig

    def plot_predictions(self, deployid: str, predictions: Dict[str, pd.DataFrame], 
                         sensors: Dict[str, pd.DataFrame] = None, outcomes: Dict[str, pd.Series] = None, 
                         interactive=False) -> Union[plotlyFigure, matplotlibFigure]:
        """Plot model predictions

        Returns:
            Figure: a plotly figure with sensor data and predictions probabilities in subplots. Open markers indicate actual events, blue points indicate true positive predictions, and red points indicated false positives.
        """
        assert (sensors is None and deployid in self.sensors) or (deployid in sensors)
        # Join sensor data with predictions
        if sensors is None:
            sensors = self.sensors
        data = self.sensors[deployid].join(predictions[deployid]["local_proba"])

        if outcomes is not None:
            data = data.join(outcomes[deployid])

        predicted_only = data[data["outcome"].isin(["TP", "FP"])]
        actual_only = data[data["outcome"].isin(["TP", "FN"])] if deployid in self.event_idx else None
        data.drop("outcome", axis="columns", inplace=True)

        if interactive:
            return self.__plot_predictions_interactive(data, predicted_only, actual_only)
        else:
            return self.__plot_predictions_static(data, predicted_only, actual_only)

    def __plot_predictions_interactive(self, data, predicted, actual) -> plotlyFigure:
        fig = make_subplots(rows=len(data.columns), cols=1,
                            shared_xaxes=True,)

        for i, col in enumerate(data):
            # Line plot
            fig.append_trace(go.Scatter(
                x=data.index,
                y=data[col],
                mode="lines"
            ), row=i + 1, col=1)
            # Predicted events
            fig.append_trace(go.Scatter(
                x=predicted.index,
                y=predicted[col],
                marker_color=["blue" if o == "TP" else "red" for o in predicted["outcome"]],
                mode="markers"
            ), row=i + 1, col=1)
            # Actual events
            if actual is not None:
                fig.append_trace(go.Scatter(
                    x=actual.index,
                    y=actual[col],
                    mode="markers",
                    marker_symbol="circle-open",
                    marker_size=10,
                    marker_color=["red" if o == "FN" else "blue" for o in actual["outcome"]],
                ), row=i + 1, col=1)
            if col == "depth":
                fig.update_yaxes(autorange="reversed", row=i + 1, col=1)
            fig.update_yaxes(title_text=col, row=i + 1, col=1)
            
        fig.update_layout(showlegend=False)
        return fig

    def __plot_predictions_static(self, data, predicted, actual) -> matplotlibFigure:
        fig, axs = plt.subplots(len(data.columns), 1)
        for i, col in enumerate(data):
            # sensor data
            axs[i].plot(data.index, data[col], "-", zorder=1)
            axs[i].set_ylabel(col)
            # predicted events
            axs[i].scatter(predicted.index, 
                           predicted[col], 
                           c=["blue" if o == "TP" else "red" for o in predicted["outcome"]], zorder=2)
            # actual events
            axs[i].scatter(actual.index, 
                           actual[col], 
                           edgecolors=["blue" if o == "TP" else "red" for o in actual["outcome"]],
                           facecolors="none",
                           zorder=3)
            if col == "depth":
                axs[i].invert_yaxis()
            
        return fig
