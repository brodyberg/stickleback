import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure
from plotly.subplots import make_subplots

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

    def __init__(self, sensors: pd.DataFrame, events: pd.DatetimeIndex, win_size: int, seed: int = None):
        self.sensors = sensors
        self.events = events
        self.win_size = win_size
        self.rg = np.random.Generator(np.random.PCG64(seed))

        self.event_idx = np.array([sensors.index.get_loc(e, method="nearest") for e in events])

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

