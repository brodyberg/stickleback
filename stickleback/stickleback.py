import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from sktime.classification.compose import TimeSeriesForestClassifier, ColumnEnsembleClassifier
from sktime.utils.data_processing import from_3d_numpy_to_nested

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

    def __init__(self, sensors: pd.DataFrame, events: pd.DatetimeIndex, win_size: int, seed: int = None, 
                 proba_thr: float = 0.5, min_period: int = 1):
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
        self.proba_thr = proba_thr
        self.min_period = min_period
        self.event_proba = pd.Series(dtype=float)
        self.pred_events = pd.DatetimeIndex
        self.pred_event_idx = np.zeros([], dtype=int)
        self.outcomes = pd.Series(dtype="string")

    def sample_nonevents(self, n: int = None) -> None:
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
        """
        Extract samples from longitudinal sensor data and reformat into nested sktime DataFrame format
        
            Parameters:
                data: longitudinal sensor data
                idx: indices of sample centers
                window_size: number of records in each sample window
            
            Returns:
                Sample windows in nested sktime DataFrame format
        """
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
        assert self.nonevents_sampled, "Can't extract training data until nonevents sampled"
        nested_events = self._extract_nested(self.event_idx)
        nested_nonevents = self._extract_nested(self.nonevent_idx)
        self.clf_data = pd.concat([nested_events, nested_nonevents])
        self.clf_labels = ["event"] * len(nested_events) + ["nonevent"] * len(nested_nonevents)
        self.train_extracted = True

    def fit(self):
        self.clf.fit(self.clf_data, self.clf_labels)

    def predict_self(self, nth: int = 1, mask: np.ndarray = None):
        idx = np.array(range(int(self.win_size / 2), len(self.sensors) - int(self.win_size / 2), nth))
        if mask is not None:
            assert issubclass(mask.dtype.type, np.integer), "mask must be integers"
            idx = np.intersect1d(idx, mask)
        all_nested = self._extract_nested(idx)
        event_proba = self.clf.predict_proba(all_nested)[:, 0]
        self.event_proba = pd.Series(event_proba, name="event_proba", index=all_nested.index)
        proba_peaks = find_peaks(event_proba, height=self.proba_thr, distance=self.min_period)
        # TODO handle no peaks case
        self.pred_events = all_nested.index[proba_peaks[0]]
        self.pred_event_idx = np.array([self.sensors.index.get_loc(p) for p in self.pred_events])
        self.predicted = True
        
    def assess(self, tol: int = 1):
        """
        Assess prediction accuracy
        
        The closest predicted event to each actual event (within tolerance) is considered a true positive. Predicted 
        events that are not the closest prediction to an actual event (or farther than tolerance) are considered false
        positives. Actual events with no predicted event within the tolerance are considered false negatives.
        
            Parameters:
                predicted: datetimes of predicted events
                actual: datetimes of actual events
                data: longitudinal sensor data
                window_size: number of records per window
        """
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
    
    def plot_sensors_events(self) -> Figure:
        event_sensors = self.sensors \
                            .append(pd.DataFrame(index=self.events)) \
                            .interpolate("index") \
                            .loc[self.events]

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
        fig.show()

    def plot_predictions(self) -> Figure:
        assert self.assessed, "Cannot plot predictions until predictions have been assessed"
        # Plot sensor data and predictions
        data = self.sensors.join(self.event_proba).join(self.outcomes)
        fig = make_subplots(rows=len(data.columns), cols=1,
                            shared_xaxes=True,)
        predicted_only = data.iloc[self.pred_event_idx]
        actual_only = data.iloc[self.event_idx]

        for i, col in enumerate(data):
            # Line plot
            fig.append_trace(go.Scatter(
                x=data.index,
                y=data[col],
                mode="lines"
            ), row=i + 1, col=1)
            # Predicted events
            fig.append_trace(go.Scatter(
                x=predicted_only.index,
                y=predicted_only[col],
                marker_color=["blue" if o == "TP" else "red" for o in predicted_only["outcome"]],
                mode="markers"
            ), row=i + 1, col=1)
            # Actual events
            fig.append_trace(go.Scatter(
                x=actual_only.index,
                y=actual_only[col],
                mode="markers",
                marker_symbol="circle-open",
                marker_size=10,
                marker_color="purple",
            ), row=i + 1, col=1)
            if col == "depth":
                fig.update_yaxes(autorange="reversed", row=i + 1, col=1)
            fig.update_yaxes(title_text=col, row=i + 1, col=1)
            
        fig.update_layout(showlegend=False)
        return fig
