"""Breath-counting task module.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from .io import *
from .plotting import *


__all__ = ["plot_presses"]



def plot_presses(
        df, export_filepath,
        participant="participant", cycle="cycle", press="press",
        timestamp="timestamp", response="response", accuracy="accuracy",
        task_length=10, close=True,
    ):
    """Breath-counting task individual press plot.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with a single press in each row.
    export_filepath : str or pathlib.Path instance
        Full filepath to save low resolution image. High resolution
        image will be saved in a subdirectory 'hires' within.
    participant : str
        Name of column containing the participant ID.
    cycle : str
        Name of column containing the cycle number.
    press : str
        Name of column containing the press number (within each cycle).
    timestamp : str
        Name of column containing the press timestamp. Must be in units
        of seconds with zero as the beginning of the task.
    response : str
        Name of column containing the response string (response strings
        must be either 'target', 'nontarget', or 'reset').
    accuracy : str
        Name of column containing the accuracy of each press (must be
        either 'correct', 'undershoot', 'overshoot', or 'selfcaught').
    task_length : int
        Number of minutes each participant performed the task.
    close : bool
        If True, close figure after saving. If false, return figure.

    Returns
    -------
    fig : matplotlib Figure
        Returns the Figure object with both axes.

    Notes
    -----
    Draw a detailed visualization of every press from every participant.
    Each participant is on their own row, and each of their presses is
    marked with a shape dependent on the response and a color dependent
    on the accuracy. This plot is mostly for internal use.
    """

    # Check column arguments.
    assert cycle in df, f"Column {cycle} not found in dataframe."
    assert press in df, f"Column {press} not found in dataframe."
    assert response in df, f"Column {response} not found in dataframe."
    assert accuracy in df, f"Column {accuracy} not found in dataframe."
    assert timestamp in df, f"Column {timestamp} not found in dataframe."
    assert participant in df, f"Column {participant} not found in dataframe."

    # Check content of columns.
    assert df[response].isin(["target", "nontarget", "reset"]).all()
    assert df[accuracy].isin(["correct", "undershoot", "overshoot", "selfcaught"]).all()
    assert df[timestamp].min() > 0
    assert df[timestamp].max() < 60*task_length
    assert df.groupby(participant)[timestamp].apply(lambda s: s.is_monotonic_increasing).all()

    # Load custom default settings.
    load_matplotlib_settings()

    # Set defaults.
    markers = dict(nontarget="v", target="o", reset="s")
    sizes = dict(nontarget=2, target=30, reset=10)
    palette = dict(correct="forestgreen", overshoot="indianred", undershoot="indianred", selfcaught="gray")

    scatter_kwargs = dict(alpha=.7, linewidths=0, clip_on=False)
    barh_kwargs = dict(color="white", clip_on=False, edgecolor="black", linewidth=1, height=.8)
    gridspec_kwargs = dict(top=.7, bottom=.25, left=.1, right=.9,
        width_ratios=[10, 1], wspace=.1)

    n_participants = df[participant].nunique()
    fig_width = .65 * task_length
    fig_height = .5 * n_participants
    figsize = (fig_width, fig_height)

    task_length_secs = task_length * 60
    xlimits = (0, task_length_secs)

    ylabels = df["participant_id"].unique()

    fig, (ax, axright) = plt.subplots(ncols=2, figsize=figsize,
        gridspec_kw=gridspec_kwargs, sharey=True,
        constrained_layout=False)

    for i, (_, subj_df) in enumerate(df.groupby("participant_id")):
        for (resp, acc), _df in subj_df.groupby(["response", "accuracy"]):
            xvals = _df["timestamp"].values
            yvals = np.repeat(i, xvals.size)
            ax.scatter(xvals, yvals, c=palette[acc],
                s=sizes[resp], marker=markers[resp],
                **scatter_kwargs)
        # Draw total accuracy bar on the right axis.
        subj_total_accuracy = subj_df.groupby("cycle"
            )["accuracy"].last().eq("correct").mean()
        axright.barh(i, subj_total_accuracy, **barh_kwargs)

    # Main axis aesthetics.
    ax.set_xlim(xlimits)
    msec2min = lambda x, pos: int(x/60)
    ax.xaxis.set(major_locator=plt.MaxNLocator(task_length),
        major_formatter=msec2min)
    ax.yaxis.set(major_locator=plt.MultipleLocator(1))
    ax.set_yticks(range(n_participants))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time during Breath Counting Task\n(minutes)")
    ax.set_ylabel("Participant ID")
    ax.set_ylim(-.5, n_participants-.5)
    ax.invert_yaxis()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))

    # Right axis aesthetics.
    axright.grid(False)
    axright.tick_params(which="both", top=False, left=False, right=False)
    axright.set_xlabel("Accuracy")
    axright.set_xlim(0, 1)
    axright.xaxis.set(major_locator=plt.MultipleLocator(1),
        minor_locator=plt.MultipleLocator(.5),
        major_formatter=plt.matplotlib.ticker.PercentFormatter(xmax=1))
    for side, spine in axright.spines.items():
        if side in ["top", "right"]:
            spine.set_visible(False)
        elif side == "bottom":
            spine.set_position(("outward", 5))

    # Legends. (need 2, one for the button press type and one for accuracy)
    marker_legend_handles = [ plt.matplotlib.lines.Line2D([0], [0],
            label=x, marker=m, markersize=6, color="white",
            markerfacecolor="white", markeredgecolor="black")
        for x, m in markers.items() ]
    marker_legend = ax.legend(handles=marker_legend_handles,
        loc="lower left", bbox_to_anchor=(0, 1),
        title="button press", ncol=3)

    accuracy_legend_handles = [ plt.matplotlib.patches.Patch(
            label=x, facecolor=c, edgecolor="white")
        for x, c in palette.items() ]
    accuracy_legend = ax.legend(handles=accuracy_legend_handles,
        loc="lower right", bbox_to_anchor=(1, 1),
        title="accuracy", ncol=3)

    ax.add_artist(marker_legend)
    ax.add_artist(accuracy_legend)

    fig.align_xlabels()

    fig = save_matplotlib(export_filepath, close=close)
    if not close:
        return fig