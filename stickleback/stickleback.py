from functools import reduce
from matplotlib.figure import Figure as matplotlibFigure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure as plotlyFigure
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from sktime.classification.compose import TimeSeriesForestClassifier, ColumnEnsembleClassifier
from sktime.utils.data_processing import from_3d_numpy_to_nested
from typing import Tuple, Union

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

class Stickleback:
    event = "event"
    nonevent = "nonevent"

    """Identify point behaviors in longitudinal sensor data.
    """

    def __init__(self, sensors: pd.DataFrame, events: pd.MultiIndex, win_size: int, seed: int = None, 
                 min_period: int = 1, proba_thr: float = 0.5, proba_prom: float = 0.25):
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
        self.event_idx = sensors.index[sensors.index.searchsorted(events)]
        self.nonevent_idx = pd.MultiIndex.from_product([[], []])
        self.clf_data = pd.DataFrame()
        self.clf_labels = []
        self.clf = ColumnEnsembleClassifier(
            estimators=[("TSF_" + c, TimeSeriesForestClassifier(n_estimators=100), [i]) 
                        for i, c in enumerate(self.sensors.columns)]
        )

        # Predictions
        self.min_period = min_period
        self.proba_thr = proba_thr
        self.proba_prom = proba_prom

    def sample_nonevents(self) -> None:
        """Sample non-events for model training.
        """
        self.nonevent_idx = pd.MultiIndex.from_product([[], []])

        for deployid in self.sensors.index.unique(0):
            sensors = self.sensors.loc[deployid]
            event_idx = self.event_idx.to_series().loc[pd.IndexSlice[deployid, :]].index.get_level_values(1)

            # Valid indices for negatives
            nonevent_choices = np.array(range(self.win_size, len(sensors) - self.win_size - 1, self.win_size))
            diff_from_event = _diff_from(nonevent_choices, sensors.index.searchsorted(event_idx))
            nonevent_choices = nonevent_choices[diff_from_event > self.win_size]

            # Randomly choose nonevents
            nonevent_idx = self.rg.choice(nonevent_choices, size=len(event_idx), replace=False)
            nonevent_idx.sort()
            nonevent_idx = pd.MultiIndex.from_product([[deployid], sensors.index[nonevent_idx]])
            self.nonevent_idx = nonevent_idx.union(self.nonevent_idx)

        self.nonevents_sampled = True
    
    def _extract_nested(self, idx: pd.MultiIndex) -> pd.DataFrame:
        """Extract windows of data in nested sktime DataFrame format

        Args:
            idx (pd.MultiIndex): Indices of window centers.

        Returns:
            pd.DataFrame: Extracted windows in nested sktime DataFrame format.
        """
        data_3d_list = []
        win_size_2 = int(self.win_size / 2)
        for deployid in idx.unique(0):
            deployevents = idx.get_level_values(1)[idx.get_level_values(0) == deployid]
            deploysensors = self.sensors.loc[deployid]
            deployidx = np.array([deploysensors.index.get_loc(t) for t in deployevents])
            out_of_bounds = deployidx[np.logical_or(deployidx < win_size_2, deployidx >= len(deploysensors) - win_size_2)]
            #set_trace()
            assert len(out_of_bounds) == 0, \
                   "Index {} out of bounds in {} (and possibly more)".format(out_of_bounds[0], deployid)
            data_3d = np.empty([len(deployidx), len(deploysensors.columns), self.win_size], float)
            data_arr = deploysensors.to_numpy().transpose()
            start_idx = deployidx - win_size_2
            for i, start in enumerate(start_idx):
                data_3d[i] = data_arr[:, start:(start + self.win_size)]
            data_3d_list.append(data_3d)
        all_data_3d = np.concatenate(data_3d_list)

        # Convert 3d numpy array to nested sktime DataFrame format
        nested = from_3d_numpy_to_nested(all_data_3d)
        nested.columns = self.sensors.columns
        nested.index = idx
        
        return nested

    def _extract_all(self, deploysensors: pd.DataFrame, nth: int):
        win_size_2 = int(self.win_size / 2)
        
        idx = np.arange(win_size_2, len(deploysensors) - win_size_2, nth)
        data_3d = np.empty([len(idx), len(deploysensors.columns), self.win_size], float)
        data_arr = deploysensors.to_numpy().transpose()
        start_idx = idx - win_size_2
        for i, start in enumerate(start_idx):
            data_3d[i] = data_arr[:, start:(start + self.win_size)]
        
        nested = from_3d_numpy_to_nested(data_3d)
        nested.columns = self.sensors.columns
        nested.index = deploysensors.index[idx]
        
        return nested

    def extract_training_data(self):
        """Extract training data from longitudinal sensor data
        """
        assert self.nonevents_sampled, "Can't extract training data until nonevents sampled"
        nested_events = self._extract_nested(self.event_idx)
        nested_nonevents = self._extract_nested(self.nonevent_idx)
        self.clf_data = pd.concat([nested_events, nested_nonevents])
        self.clf_labels = [Stickleback.event] * len(nested_events) + [Stickleback.nonevent] * len(nested_nonevents)
        self.train_extracted = True

    def fit(self):
        """Fit classifier to training data
        """
        assert self.train_extracted, "Can't fit until training data extracted"
        self.clf.fit(self.clf_data, self.clf_labels)

    def _predict_one(self, deployid: str, nth: int) -> Tuple[pd.Series, pd.MultiIndex]:
        assert deployid in self.sensors.index.unique(0), "{} not found in sensors".format(deployid)
        deploysensors = self.sensors.loc[deployid]
        deploysensors.index = pd.DatetimeIndex(deploysensors.index)
        all_nested = self._extract_all(deploysensors, nth)
        event_proba = self.clf.predict_proba(all_nested)[:, 0]
        event_proba = pd.Series(event_proba, name="event_proba", index=all_nested.index) \
            .reindex(deploysensors.index) \
            .interpolate(method="cubic")
        proba_peaks = find_peaks(event_proba, height=self.proba_thr, distance=self.min_period, prominence=self.proba_prom)
        
        # TODO handle no peaks case

        event_proba.index = pd.MultiIndex.from_product([[deployid], event_proba.index])
        event_idx = event_proba.index[proba_peaks[0]]
        
        return event_proba, event_idx

    def predict_self(self, nth: int = 1) -> Tuple[pd.Series, pd.MultiIndex]:
        """Predict in-sample events

        Args:
            nth (int, optional): Predict every nth window and interpolate probabilities in between. Defaults to 1.
        """
        by_deployid = [self._predict_one(i, nth) for i in self.event_idx.unique(0)]
        event_probas = pd.concat([d[0] for d in by_deployid])
        event_idxs = reduce(lambda x, y: x.union(y), [d[1] for d in by_deployid])
        return event_probas, event_idxs

    def _assess_one(self, deployid: str, pred_idx: pd.MultiIndex, tol: pd.Timedelta) -> pd.Series:
        assert deployid in self.sensors.index.unique(0), "{} not found in sensors".format(deployid)
        
        # Find closest predicted to each actual and their distance
        # TODO handle no predicted events case
        actual = pd.DatetimeIndex(self.event_idx.get_level_values(1)[self.event_idx.get_level_values(0) == deployid])
        predicted = pd.DatetimeIndex(pred_idx.get_level_values(1))
        closest = predicted[[np.argmin(np.abs(predicted - a)) for a in actual]]
        distance = np.abs(actual - closest)
        
        # Initialize outcomes
        outcomes = pd.Series("TN", name="outcome", index=self.sensors.loc[deployid].index)
        
        # Iterate through actual events. The closest predicted event within the tolerance is a true positive. If no
        # predicted events are within the tolerance, the actual event is a false negative.
        for i, (c, d) in enumerate(zip(closest, distance)):
            if d <= tol:
                outcomes[c] = "TP" 
            else:
                outcomes[actual[i]] = "FN"
            o = "TP" if d <= tol else "FN"

        # Iterate through predicted events. Any predicted events that aren't the closest to an actual event are false
        # positives.
        for i, p in enumerate(predicted):
            if p not in closest:
                outcomes[p] = "FP" 
            
        # Sanity checks
        # TODO make these work with mask parameter
        # n_tp = np.sum(self.outcomes == "TP")
        # n_fp = np.sum(self.outcomes == "FP")
        # n_fn = np.sum(self.outcomes == "FN")
        # assert (n_tp + n_fp) == len(self.pred_events), "TP + FP != count of predicted events"
        # assert (n_tp + n_fn) == len(self.events), "TP + FN != count of actual events"
        outcomes.index = pd.MultiIndex.from_product([[deployid], outcomes.index])
        return outcomes
        
    def assess(self, pred_idx: pd.MultiIndex, tol: pd.Timedelta) -> pd.Series:
        """Assess prediction accuracy
        
        The closest predicted event to each actual event (within tolerance) is considered a true positive. Predicted 
        events that are not the closest prediction to an actual event (or farther than tolerance) are considered false
        positives. Actual events with no predicted event within the tolerance are considered false negatives.

        Args:
            deployid (str): deployment identifier
            pred_idx (pd.MultiIndex): indices of predicted events (deployid, timestamp)
            tol (int, optional): Tolerance for linking predicted and actual events (in records). Defaults to 1.
        """    
        outcomes = [self._assess_one(i, pred_idx[pred_idx.get_locs([i])], tol) for i in pred_idx.unique(0)]
        return pd.concat(outcomes).rename_axis(["deployid", "time"])

    def refit(self, new_data_idx: pd.MultiIndex, new_labels: list):
        """Refit model

        Adds data to training dataset and re-fits classifer.
        """
        self.clf_data = pd.concat([self.clf_data, self._extract_nested(new_data_idx)])
        self.clf_labels = self.clf_labels + new_labels
        self.fit()
    
    def plot_sensors_events(self, deployid, interactive=False) -> Union[plotlyFigure, matplotlibFigure]:
        """Plot longitudinal sensor data and events

        Returns:
            Figure: a plotly figure with sensor data in subplots and events marked with points.
        """
        assert(deployid in self.sensors.index)
        sensors = self.sensors.loc[deployid]
        event_sensors = self.sensors.loc[self.event_idx].loc[deployid]
        
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

    def plot_predictions(self, deployid: str, event_proba: pd.Series, pred_idx: pd.MultiIndex,
                         outcomes: pd.Series, interactive=False) -> Union[plotlyFigure, matplotlibFigure]:
        """Plot model predictions

        Returns:
            Figure: a plotly figure with sensor data and predictions probabilities in subplots. Open markers indicate actual events, blue points indicate true positive predictions, and red points indicated false positives.
        """
        assert deployid in self.sensors.index.unique(0), "{} not found in sensors".format(deployid)

        # Join sensor data with predictions
        data = pd.concat([self.sensors.loc[deployid], event_proba[deployid], outcomes[deployid]], axis=1)
        predicted_only = data.loc[pred_idx[pred_idx.get_locs([deployid])].droplevel(0)]
        actual_only = data.loc[self.event_idx[self.event_idx.get_locs([deployid])].droplevel(0)]
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
