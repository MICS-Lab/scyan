# Copied from https://github.com/scverse/scanpy (and modified)

from cycler import cycler
from matplotlib import rcParams

from . import palettes


def set_rcParams_scanpy(fontsize):
    """Set matplotlib.rcParams to Scanpy defaults.

    Call this through `settings.set_figure_params`.
    """

    # DPI
    rcParams["figure.dpi"] = 80
    rcParams["savefig.dpi"] = 150

    # figure
    rcParams["figure.figsize"] = (4, 4)
    rcParams["figure.subplot.left"] = 0.18
    rcParams["figure.subplot.right"] = 0.96
    rcParams["figure.subplot.bottom"] = 0.15
    rcParams["figure.subplot.top"] = 0.91

    rcParams["lines.linewidth"] = 1.5  # the line width of the frame
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 1

    # font
    rcParams["font.sans-serif"] = [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ]
    fontsize = fontsize
    rcParams["font.size"] = fontsize
    rcParams["legend.fontsize"] = 0.92 * fontsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.labelsize"] = fontsize

    # legend
    rcParams["legend.numpoints"] = 1
    rcParams["legend.scatterpoints"] = 1
    rcParams["legend.handlelength"] = 0.5
    rcParams["legend.handletextpad"] = 0.4

    # color cycle
    rcParams["axes.prop_cycle"] = cycler(color=palettes.default_20)

    # lines
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.facecolor"] = "white"

    # ticks
    rcParams["xtick.color"] = "k"
    rcParams["ytick.color"] = "k"
    rcParams["xtick.labelsize"] = fontsize
    rcParams["ytick.labelsize"] = fontsize

    # axes grid
    # rcParams["axes.grid"] = True
    rcParams["grid.color"] = ".8"


def quality_settings():
    import builtins

    if getattr(builtins, "__IPYTHON__", False):
        import matplotlib_inline.backend_inline

        matplotlib_inline.backend_inline.set_matplotlib_formats("png2x")


def reset_plot_settings(fontsize: int = 10):
    quality_settings()
    set_rcParams_scanpy(fontsize)
