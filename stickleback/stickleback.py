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
from typing import Union

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
    return np.array([np.min(np.abs(x - ys)) for x in xs])

class Stickleback:
    """Identify point behaviors in longitudinal sensor data.
    """

    def __init__(self, sensors: pd.DataFrame, events: pd.DatetimeIndex, win_size: int, seed: int = None, 
                 min_period: int = 1, proba_thr: float = 0.5, proba_prom: float = 0.25):
        """Instantiate a Stickleback object 

        Args:
            sensors (pd.DataFrame): Longitudinal sensor data, e.g. depth, pitch, and jerk.
            events (pd.DatetimeIndex): Times of point behavior events.
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
        self.predicted = False
        self.assessed = False

        # Classifier attributes
        self.event_idx = np.array([sensors.index.get_loc(e, method="nearest") for e in events])
        self.nonevent_idx = np.zeros(len(self.event_idx), int)
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
        self.event_proba = pd.Series(dtype=float)
        self.pred_events = pd.DatetimeIndex
        self.pred_event_idx = np.zeros([], dtype=int)
        self.outcomes = pd.Series(dtype="string")

    def sample_nonevents(self, n: int = None) -> None:
        """Sample non-events for model training.

        Args:
            n (int, optional): Number of non-events to sample. If None (default), samples an equal number of non-events as events.
        """
        if n is None:
            n = len(self.events)

        # Valid indices for negatives
        nonevent_choices = np.array(range(self.win_size, len(self.sensors.index) - self.win_size - 1, self.win_size))
        diff_from_event = _diff_from(nonevent_choices, self.event_idx)
        nonevent_choices = nonevent_choices[diff_from_event > self.win_size]

        # Randomly choose nonevents
        nonevent_idx = self.rg.choice(nonevent_choices, size=n, replace=False)
        nonevent_idx.sort()

        self.nonevent_idx = nonevent_idx
        self.nonevents_sampled = True
    
    def _extract_nested(self, idx: np.ndarray) -> pd.DataFrame:
        """Extract windows of data in nested sktime DataFrame format

        Args:
            idx (np.ndarray): Indices of window centers.

        Returns:
            pd.DataFrame: Extracted windows in nested sktime DataFrame format.
        """

        # TODO implement assert_shape and assert_dtype. See: https://medium.com/@nearlydaniel/assertion-of-arbitrary-array-shapes-in-python-3c96f6b7ccb4
        # assert_shape(idx, [None])
        # assert_dtype(idx, np.integer)
        assert idx.min() >= int(self.win_size / 2), "idx out of bounds"
        assert idx.max() < len(self.sensors) - int(self.win_size / 2), "idx out of bounds"
        
        # Create a 3d numpy array of window data
        data_3d = np.empty([len(idx), len(self.sensors.columns), self.win_size], float)
        data_arr = self.sensors.to_numpy().transpose()
        start_idx = idx - int(self.win_size / 2)
        for i, start in enumerate(start_idx):
            data_3d[i] = data_arr[:, start:(start + self.win_size)]

        # Convert 3d numpy array to nested sktime DataFrame format
        nested = from_3d_numpy_to_nested(data_3d)
        nested.columns = self.sensors.columns
        nested.index = self.sensors.index[idx]
        
        return nested

    def extract_training_data(self):
        """Extract training data from longitudinal sensor data
        """
        assert self.nonevents_sampled, "Can't extract training data until nonevents sampled"
        nested_events = self._extract_nested(self.event_idx)
        nested_nonevents = self._extract_nested(self.nonevent_idx)
        self.clf_data = pd.concat([nested_events, nested_nonevents])
        self.clf_labels = ["event"] * len(nested_events) + ["nonevent"] * len(nested_nonevents)
        self.train_extracted = True

    def fit(self):
        """Fit classifier to training data
        """
        assert self.train_extracted, "Can't fit until training data extracted"
        self.clf.fit(self.clf_data, self.clf_labels)

    def predict_self(self, nth: int = 1, mask: np.ndarray = None):
        """Predict in-sample events

        Args:
            nth (int, optional): Predict every nth window and interpolate probabilities in between. Defaults to 1.
            mask (np.ndarray, optional): Indicies to predict (i.e., for subsetting). Defaults to None.
        """
        idx = np.array(range(int(self.win_size / 2), len(self.sensors) - int(self.win_size / 2), nth))
        if mask is not None:
            assert issubclass(mask.dtype.type, np.integer), "mask must be integers"
            idx = np.intersect1d(idx, mask)
        all_nested = self._extract_nested(idx)
        event_proba = self.clf.predict_proba(all_nested)[:, 0]
        self.event_proba = pd.Series(event_proba, name="event_proba", index=all_nested.index) \
                             .reindex(self.sensors.index) \
                             .interpolate(method="cubic")
        proba_peaks = find_peaks(self.event_proba, height=self.proba_thr, distance=self.min_period, prominence=self.proba_prom)
        # TODO handle no peaks case
        self.pred_events = self.sensors.index[proba_peaks[0]]
        self.pred_event_idx = np.array([self.sensors.index.get_loc(p) for p in self.pred_events])
        self.predicted = True
        
    def assess(self, tol: int = 1):
        """Assess prediction accuracy
        
        The closest predicted event to each actual event (within tolerance) is considered a true positive. Predicted 
        events that are not the closest prediction to an actual event (or farther than tolerance) are considered false
        positives. Actual events with no predicted event within the tolerance are considered false negatives.

        Args:
            tol (int, optional): Tolerance for linking predicted and actual events (in records). Defaults to 1.
        """    
        assert self.predicted, "Cannot assess until after prediction"

        # Find closest predicted to each actual and their distance
        # TODO handle no predicted events case
        closest = np.array([np.argmin(np.abs(self.pred_event_idx - a)) for a in self.event_idx])
        distance = np.array([np.min(np.abs(self.pred_event_idx - a)) for a in self.event_idx])
        
        # Initialize outcomes
        self.outcomes = pd.Series("TN", name="outcome", index=self.sensors.index)
        
        # Iterate through actual events. The closest predicted event within the tolerance is a true positive. If no
        # predicted events are within the tolerance, the actual event is a false negative.
        for i, (c, d) in enumerate(zip(closest, distance)):
            if d <= tol:
                self.outcomes[self.pred_event_idx[c]] = "TP" 
            else:
                self.outcomes[self.event_idx[i]] = "FN"

        # Iterate through predicted events. Any predicted events that aren't the closest to an actual event are false
        # positives.
        for i, p in enumerate(self.pred_event_idx):
            if i not in closest:
                self.outcomes[p] = "FP" 
            
        # Sanity checks
        # TODO make these work with mask parameter
        # n_tp = np.sum(self.outcomes == "TP")
        # n_fp = np.sum(self.outcomes == "FP")
        # n_fn = np.sum(self.outcomes == "FN")
        # assert (n_tp + n_fp) == len(self.pred_events), "TP + FP != count of predicted events"
        # assert (n_tp + n_fn) == len(self.events), "TP + FN != count of actual events"
        
        self.assessed = True

    def refit(self):
        """Refit model

        Adds false positives to training dataset and re-fits classifer.
        """
        assert self.assessed, "Cannot refit until after assessment"

        false_pos_idx = np.array([self.sensors.index.get_loc(i) for i in self.outcomes.index[self.outcomes == "FP"]])
        if len(false_pos_idx) == 0:
            return
        
        self.clf_data = pd.concat([self.clf_data, self._extract_nested(false_pos_idx)])
        self.clf_labels = self.clf_labels + ["nonevent"] * len(false_pos_idx)
        self.fit()

        self.predicted = False
        self.assessed = False
    
    def plot_sensors_events(self, interactive=False) -> Union[plotlyFigure, matplotlibFigure]:
        """Plot longitudinal sensor data and events

        Returns:
            Figure: a plotly figure with sensor data in subplots and events marked with points.
        """
        event_sensors = self.sensors \
                            .append(pd.DataFrame(index=self.events)) \
                            .interpolate("index") \
                            .loc[self.events]
        
        if interactive:
            return self.__plot_sensors_events_interactive(event_sensors)
        else:
            return self.__plot_sensors_events_static(event_sensors)

    def __plot_sensors_events_interactive(self, event_sensors) -> plotlyFigure:
        fig = make_subplots(rows=len(self.sensors.columns), cols=1,
                            shared_xaxes=True,)
        for i, col in enumerate(self.sensors.columns):
            fig.append_trace(go.Scatter(
                x=self.sensors.index,
                y=self.sensors[col],
                mode="lines"
            ), row=i + 1, col=1)
            fig.append_trace(go.Scatter(
                x=event_sensors.index,
                y=event_sensors[col],
                mode="markers"
            ), row=i + 1, col=1)
            if col == "depth":
                fig.update_yaxes(autorange="reversed", row=i + 1, col=1)
            fig.update_yaxes(title_text=col, row=i + 1, col=1)
            
        fig.update_layout(showlegend=False)
        return fig

    def __plot_sensors_events_static(self, event_sensors) -> matplotlibFigure:
        fig, axs = plt.subplots(len(self.sensors.columns), 1)
        for i, col in enumerate(self.sensors.columns):
            # sensor data
            axs[i].plot(self.sensors.index, self.sensors[col], "-", zorder=1)
            # events
            axs[i].scatter(event_sensors.index, event_sensors[col], facecolors="none", edgecolors="r", zorder=2)
            axs[i].set_ylabel(col)
            if col == "depth":
                axs[i].invert_yaxis()
            
        return fig

    def plot_predictions(self, interactive=False) -> Union[plotlyFigure, matplotlibFigure]:
        """Plot model predictions

        Returns:
            Figure: a plotly figure with sensor data and predictions probabilities in subplots. Open markers indicate actual events, blue points indicate true positive predictions, and red points indicated false positives.
        """
        assert self.assessed, "Cannot plot predictions until predictions have been assessed"

        # Join sensor data with predictions
        data = self.sensors.join(self.event_proba).join(self.outcomes)
        predicted_only = data.iloc[self.pred_event_idx]
        actual_only = data.iloc[self.event_idx]
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
