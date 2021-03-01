import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure
from plotly.subplots import make_subplots

class Stickleback:

    def __init__(self, sensors: pd.DataFrame, events: pd.DatetimeIndex, win_size: int):
        self.sensors = sensors
        self.events = events
        self.win_size = win_size
        self.event_index = np.array([sensors.index.get_loc(e, method="nearest") for e in events])

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