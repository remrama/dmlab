"""Plotting helper functions."""

import pathlib

import colorcet as cc
import matplotlib.pyplot as plt

from .io import *

def cmap2hex(cmap, n_intervals) -> list:
    if isinstance(cmap, str):
        if (cmap := cc.cm.get(cmap)) is None:
            try:
                cmap = plt.get_cmap(cmap)
            except ValueError as e:
                raise e
    assert isinstance(cmap, plt.matplotlib.colors.LinearSegmentedColormap)
    stops = [ 0 + x*1/(n_intervals-1) for x in range(n_intervals) ] # np.linspace
    hex_codes = []
    for s in stops:
        assert isinstance(s, float)
        rgb_floats = cmap(s)
        rgb_ints = [ round(f*255) for f in rgb_floats ]
        hex_code = "#{0:02x}{1:02x}{2:02x}".format(*rgb_ints)
        hex_codes.append(hex_code)
    return hex_codes

def save_matplotlib(lores_path, hires_extension=".pdf", close=True):
    """Saves out hi-resolution matplotlib figures.
    Assumes there is a "hires" subdirectory within the path
    of the filename passed in, which must be also be a png filename.
    """
    lores_path = ensure_path_is_pathlib(lores_path)
    hires_path = lores_path.with_suffix(hires_extension)
    hires_path = hires_path.parent / "hires" / hires_path.name
    hires_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(lores_path)
    plt.savefig(hires_path)
    if close:
        plt.close()

def set_matplotlib_style(mpl_style="technical"):
    if mpl_style == "technical":
        # plt.rcParams["figure.dpi"] = 600
        plt.rcParams["savefig.dpi"] = 600
        plt.rcParams["interactive"] = True
        plt.rcParams["figure.constrained_layout.use"] = True
        plt.rcParams["font.family"] = "Times New Roman"
        # plt.rcParams["font.sans-serif"] = "Arial"
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.rm"] = "Times New Roman"
        plt.rcParams["mathtext.cal"] = "Times New Roman"
        plt.rcParams["mathtext.it"] = "Times New Roman:italic"
        plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
        plt.rcParams["font.size"] = 8
        plt.rcParams["axes.titlesize"] = 8
        plt.rcParams["axes.labelsize"] = 8
        plt.rcParams["axes.labelsize"] = 8
        plt.rcParams["xtick.labelsize"] = 8
        plt.rcParams["ytick.labelsize"] = 8
        plt.rcParams["axes.linewidth"] = 0.8 # edge line width
        plt.rcParams["axes.axisbelow"] = True
        plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.grid.axis"] = "y"
        plt.rcParams["axes.grid.which"] = "major"
        plt.rcParams["axes.labelpad"] = 4
        plt.rcParams["xtick.top"] = True
        plt.rcParams["ytick.right"] = True
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["grid.color"] = "gainsboro"
        plt.rcParams["grid.linewidth"] = 1
        plt.rcParams["grid.alpha"] = 1
        plt.rcParams["legend.frameon"] = False
        plt.rcParams["legend.edgecolor"] = "black"
        plt.rcParams["legend.fontsize"] = 8
        plt.rcParams["legend.title_fontsize"] = 8
        plt.rcParams["legend.borderpad"] = .4
        plt.rcParams["legend.labelspacing"] = .2 # the vertical space between the legend entries
        plt.rcParams["legend.handlelength"] = 2 # the length of the legend lines
        plt.rcParams["legend.handleheight"] = .7 # the height of the legend handle
        plt.rcParams["legend.handletextpad"] = .2 # the space between the legend line and legend text
        plt.rcParams["legend.borderaxespad"] = .5 # the border between the axes and legend edge
        plt.rcParams["legend.columnspacing"] = 1 # the space between the legend line and legend text
    else:
        raise ValueError(f"matplotlib style {mpl_style} is not an option")