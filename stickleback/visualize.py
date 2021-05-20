from stickleback.stickleback import Stickleback
from matplotlib.figure import Figure as matplotlibFigure
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure as plotlyFigure
from plotly.subplots import make_subplots
from typing import Dict, Union

def plot_sensors_events(stkl: Stickleback, deployid, interactive=False) -> Union[plotlyFigure, matplotlibFigure]:
    assert(deployid in stkl.sensors.keys())
    sensors = stkl.sensors[deployid]
    event_sensors = sensors.loc[stkl.event_idx[deployid]]
    
    if interactive:
        return __plot_sensors_events_interactive(sensors, event_sensors)
    else:
        return __plot_sensors_events_static(sensors, event_sensors)

def __plot_sensors_events_interactive(sensors, event_sensors) -> plotlyFigure:
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

def __plot_sensors_events_static(sensors, event_sensors) -> matplotlibFigure:
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

def plot_predictions(stkl: Stickleback, deployid: str, predictions: Dict[str, pd.DataFrame], 
                        sensors: Dict[str, pd.DataFrame] = None, outcomes: Dict[str, pd.Series] = None, 
                        interactive=False) -> Union[plotlyFigure, matplotlibFigure]:
    assert (sensors is None and deployid in stkl.sensors) or (deployid in sensors)
    # Join sensor data with predictions
    if sensors is None:
        sensors = stkl.sensors
    data = stkl.sensors[deployid].join(predictions[deployid]["local_proba"])

    if outcomes is not None:
        data = data.join(outcomes[deployid])

    predicted_only = data[data["outcome"].isin(["TP", "FP"])]
    actual_only = data[data["outcome"].isin(["TP", "FN"])] if deployid in stkl.event_idx else None
    data.drop("outcome", axis="columns", inplace=True)

    if interactive:
        return __plot_predictions_interactive(data, predicted_only, actual_only)
    else:
        return __plot_predictions_static(data, predicted_only, actual_only)

def __plot_predictions_interactive(data, predicted, actual) -> plotlyFigure:
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

def __plot_predictions_static(data, predicted, actual) -> matplotlibFigure:
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
