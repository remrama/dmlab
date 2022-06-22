"""Plotting helper functions."""

import os

# import colorcet as cc
import matplotlib.pyplot as plt

from .utils import *

def save_matplotlib(png_path, hires_extension="pdf"):
    """Saves out hi-resolution matplotlib figures.
    Assumes there is a "hires" subdirectory within the path
    of the filename passed in, which must be also be a png filename.
    """
    assert png_path.endswith(".png"), "Expected .png filename"
    png_dir, png_bname = os.path.split(png_path)
    png_bname_noext, _ = os.path.splitext(png_bname)
    hires_dir = os.path.join(png_dir, "hires")
    hires_bname = png_bname.replace(".png", f".{hires_extension}")
    hires_path = os.path.join(hires_dir, hires_bname)
    make_pathdir_if_not_exists(hires_path)
    plt.savefig(png_path)
    plt.savefig(hires_path)
    plt.close()

def load_matplotlib_settings():
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
    rcParams["axes.labelsize"] = 8
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