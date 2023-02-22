# Copied from https://github.com/scverse/scanpy (and modified)

import collections.abc as cabc
import logging
from copy import copy
from functools import partial
from itertools import combinations, product
from numbers import Integral
from typing import Collection, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from cycler import Cycler, cycler
from matplotlib import colors, patheffects
from matplotlib import pyplot as pl
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap, Normalize, is_color_like
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from pandas.api.types import is_categorical_dtype

from . import palettes

log = logging.getLogger(__name__)


def circles(
    x, y, s, ax, marker=None, c="b", vmin=None, vmax=None, scale_factor=1.0, **kwargs
):
    """
    Taken from here: https://gist.github.com/syrte/592a062c562cd2a98a83
    Make a scatter plot of circles.
    Similar to pl.scatter, but the size of circles are in data scale.
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circles.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    a = np.arange(11)
    circles(a, a, s=a*0.2, c=a, alpha=0.5, ec='none')
    pl.colorbar()
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """

    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.
    if scale_factor != 1.0:
        x = x * scale_factor
        y = y * scale_factor
    zipped = np.broadcast(x, y, s)
    patches = [Circle((x_, y_), s_) for x_, y_, s_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if isinstance(c, np.ndarray) and np.issubdtype(c.dtype, np.number):
        collection.set_array(np.ma.masked_invalid(c))
        collection.set_clim(vmin, vmax)
    else:
        collection.set_facecolor(c)

    ax.add_collection(collection)

    return collection


def check_colornorm(vmin=None, vmax=None, vcenter=None, norm=None):
    from matplotlib.colors import Normalize

    try:
        from matplotlib.colors import TwoSlopeNorm as DivNorm
    except ImportError:
        # matplotlib<3.2
        from matplotlib.colors import DivergingNorm as DivNorm

    if norm is not None:
        if (vmin is not None) or (vmax is not None) or (vcenter is not None):
            raise ValueError("Passing both norm and vmin/vmax/vcenter is not allowed.")
    else:
        if vcenter is not None:
            norm = DivNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

    return norm


def _validate_palette(adata, key):
    """
    checks if the list of colors in adata.uns[f'{key}_colors'] is valid
    and updates the color list in adata.uns[f'{key}_colors'] if needed.

    Not only valid matplotlib colors are checked but also if the color name
    is a valid R color name, in which case it will be translated to a valid name
    """

    _palette = []
    color_key = f"{key}_colors"

    for color in adata.uns[color_key]:
        if not is_color_like(color):
            # check if the color is a valid R color and translate it
            # to a valid hex color value
            if color in palettes.additional_colors:
                color = palettes.additional_colors[color]
            else:
                log.warning(
                    f"The following color value found in adata.uns['{key}_colors'] "
                    f"is not valid: '{color}'. Default colors will be used instead."
                )
                _set_default_colors_for_categorical_obs(adata, key)
                _palette = None
                break
        _palette.append(color)
    # Don't modify if nothing changed
    if _palette is not None and list(_palette) != list(adata.uns[color_key]):
        adata.uns[color_key] = _palette


def _set_colors_for_categorical_obs(adata, value_to_plot, palette):
    """
    Sets the adata.uns[value_to_plot + '_colors'] according to the given palette

    Parameters
    ----------
    adata
        annData object
    value_to_plot
        name of a valid categorical observation
    palette
        Palette should be either a valid :func:`~matplotlib.pyplot.colormaps` string,
        a sequence of colors (in a format that can be understood by matplotlib,
        eg. RGB, RGBS, hex, or a cycler object with key='color'

    Returns
    -------
    None
    """
    from matplotlib.colors import to_hex

    categories = adata.obs[value_to_plot].cat.categories
    # check is palette is a valid matplotlib colormap
    if isinstance(palette, str) and palette in pl.colormaps():
        # this creates a palette from a colormap. E.g. 'Accent, Dark2, tab20'
        cmap = pl.get_cmap(palette)
        colors_list = [to_hex(x) for x in cmap(np.linspace(0, 1, len(categories)))]
    elif isinstance(palette, cabc.Mapping):
        colors_list = [to_hex(palette[k], keep_alpha=True) for k in categories]
    else:
        # check if palette is a list and convert it to a cycler, thus
        # it doesnt matter if the list is shorter than the categories length:
        if isinstance(palette, cabc.Sequence):
            if len(palette) < len(categories):
                log.warning(
                    "Length of palette colors is smaller than the number of "
                    f"categories (palette length: {len(palette)}, "
                    f"categories length: {len(categories)}. "
                    "Some categories will have the same color."
                )
            # check that colors are valid
            _color_list = []
            for color in palette:
                if not is_color_like(color):
                    # check if the color is a valid R color and translate it
                    # to a valid hex color value
                    if color in palettes.additional_colors:
                        color = palettes.additional_colors[color]
                    else:
                        raise ValueError(
                            "The following color value of the given palette "
                            f"is not valid: {color}"
                        )
                _color_list.append(color)

            palette = cycler(color=_color_list)
        if not isinstance(palette, Cycler):
            raise ValueError(
                "Please check that the value of 'palette' is a valid "
                "matplotlib colormap string (eg. Set2), a  list of color names "
                "or a cycler with a 'color' key."
            )
        if "color" not in palette.keys:
            raise ValueError("Please set the palette key 'color'.")

        cc = palette()
        colors_list = [to_hex(next(cc)["color"]) for x in range(len(categories))]

    adata.uns[value_to_plot + "_colors"] = colors_list


def _set_default_colors_for_categorical_obs(adata, value_to_plot):
    """
    Sets the adata.uns[value_to_plot + '_colors'] using default color palettes

    Parameters
    ----------
    adata
        AnnData object
    value_to_plot
        Name of a valid categorical observation

    Returns
    -------
    None
    """
    categories = adata.obs[value_to_plot].cat.categories
    length = len(categories)

    # check if default matplotlib palette has enough colors
    if len(rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
        cc = rcParams["axes.prop_cycle"]()
        palette = [next(cc)["color"] for _ in range(length)]

    else:
        if length <= 20:
            palette = palettes.default_20
        elif length <= 28:
            palette = palettes.default_28
        elif length <= len(palettes.default_102):  # 103 colors
            palette = palettes.default_102
        else:
            palette = ["grey" for _ in range(length)]
            log.info(
                f"the obs value {value_to_plot!r} has more than 103 categories. Uniform "
                "'grey' color will be used for all categories."
            )

    _set_colors_for_categorical_obs(adata, value_to_plot, palette[:length])


def scanpy_pl_umap(
    adata: AnnData,
    color: Union[str, Sequence[str], None] = None,
    gene_symbols: Optional[str] = None,
    use_raw: Optional[bool] = None,
    sort_order: bool = True,
    groups: Optional[str] = None,
    components: Union[str, Sequence[str]] = None,
    dimensions: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None,
    layer: Optional[str] = None,
    scale_factor: Optional[float] = None,
    color_map: Union[Colormap, str, None] = None,
    cmap: Union[Colormap, str, None] = None,
    palette=None,
    na_color="lightgray",
    na_in_legend: bool = True,
    size: Union[float, Sequence[float], None] = None,
    frameon: Optional[bool] = False,
    legend_fontsize=None,
    legend_fontweight="bold",
    legend_loc: str = "right margin",
    legend_fontoutline: Optional[int] = None,
    colorbar_loc: Optional[str] = "right",
    vmax=None,
    vmin=None,
    vcenter=None,
    norm: Union[Normalize, Sequence[Normalize], None] = None,
    add_outline: Optional[bool] = False,
    outline_width: Tuple[float, float] = (0.3, 0.05),
    outline_color: Tuple[str, str] = ("black", "white"),
    ncols: int = 4,
    hspace: float = 0.25,
    wspace: Optional[float] = None,
    title: Union[str, Sequence[str], None] = None,
    show: Optional[bool] = None,
    ax: Optional[Axes] = None,
    return_fig: Optional[bool] = None,
    **kwargs,
) -> Union[Figure, Axes, None]:
    adata.strings_to_categoricals()

    basis_values = adata.obsm["X_umap"]
    dimensions = _components_to_dimensions(
        components, dimensions, total_dims=basis_values.shape[1]
    )

    # Figure out if we're using raw
    if use_raw is None:
        # check if adata.raw is set
        use_raw = layer is None and adata.raw is not None
    if use_raw and layer is not None:
        raise ValueError(
            "Cannot use both a layer and the raw representation. Was passed:"
            f"use_raw={use_raw}, layer={layer}."
        )
    if use_raw and adata.raw is None:
        raise ValueError(
            "`use_raw` is set to True but AnnData object does not have raw. "
            "Please check."
        )

    if isinstance(groups, str):
        groups = [groups]

    # Color map
    if color_map is not None:
        if cmap is not None:
            raise ValueError("Cannot specify both `color_map` and `cmap`.")
        else:
            cmap = color_map
    cmap = copy(get_cmap(cmap))
    cmap.set_bad(na_color)
    kwargs["cmap"] = cmap
    # Prevents warnings during legend creation
    na_color = colors.to_hex(na_color, keep_alpha=True)

    if "edgecolor" not in kwargs:
        # by default turn off edge color. Otherwise, for
        # very small sizes the edge will not reduce its size
        # (https://github.com/scverse/scanpy/issues/293)
        kwargs["edgecolor"] = "none"

    # Vectorized arguments

    # turn color into a python list
    color = [color] if isinstance(color, str) or color is None else list(color)
    if title is not None:
        # turn title into a python list if not None
        title = [title] if isinstance(title, str) else list(title)

    # turn vmax and vmin into a sequence
    if isinstance(vmax, str) or not isinstance(vmax, cabc.Sequence):
        vmax = [vmax]
    if isinstance(vmin, str) or not isinstance(vmin, cabc.Sequence):
        vmin = [vmin]
    if isinstance(vcenter, str) or not isinstance(vcenter, cabc.Sequence):
        vcenter = [vcenter]
    if isinstance(norm, Normalize) or not isinstance(norm, cabc.Sequence):
        norm = [norm]

    # Size
    if "s" in kwargs and size is None:
        size = kwargs.pop("s")
    if size is not None:
        # check if size is any type of sequence, and if so
        # set as ndarray
        if (
            size is not None
            and isinstance(size, (cabc.Sequence, pd.Series, np.ndarray))
            and len(size) == adata.shape[0]
        ):
            size = np.array(size, dtype=float)
    else:
        size = 120000 / adata.shape[0]

    ##########
    # Layout #
    ##########
    # Most of the code is for the case when multiple plots are required

    if wspace is None:
        #  try to set a wspace that is not too large or too small given the
        #  current figure size
        wspace = 0.75 / rcParams["figure.figsize"][0] + 0.02

    if components is not None:
        color, dimensions = list(zip(*product(color, dimensions)))

    color, dimensions = _broadcast_args(color, dimensions)

    # 'color' is a list of names that want to be plotted.
    # Eg. ['Gene1', 'louvain', 'Gene2'].
    # component_list is a list of components [[0,1], [1,2]]
    if (
        not isinstance(color, str) and isinstance(color, cabc.Sequence) and len(color) > 1
    ) or len(dimensions) > 1:
        if ax is not None:
            raise ValueError(
                "Cannot specify `ax` when plotting multiple panels "
                "(each for a given value of 'color')."
            )

        # each plot needs to be its own panel
        fig, grid = _panel_grid(hspace, wspace, ncols, len(color))
    else:
        grid = None
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111)

    ############
    # Plotting #
    ############
    axs = []

    # use itertools.product to make a plot for each color and for each component
    # For example if color=[gene1, gene2] and components=['1,2, '2,3'].
    # The plots are: [
    #     color=gene1, components=[1,2], color=gene1, components=[2,3],
    #     color=gene2, components = [1, 2], color=gene2, components=[2,3],
    # ]
    for count, (value_to_plot, dims) in enumerate(zip(color, dimensions)):
        color_source_vector = _get_color_source_vector(
            adata,
            value_to_plot,
            layer=layer,
            use_raw=use_raw,
            gene_symbols=gene_symbols,
            groups=groups,
        )
        color_vector, categorical = _color_vector(
            adata,
            value_to_plot,
            color_source_vector,
            palette=palette,
            na_color=na_color,
        )

        # Order points
        order = slice(None)
        if sort_order is True and value_to_plot is not None and categorical is False:
            # Higher values plotted on top, null values on bottom
            order = np.argsort(-color_vector, kind="stable")[::-1]
        elif sort_order and categorical:
            # Null points go on bottom
            order = np.argsort(~pd.isnull(color_source_vector), kind="stable")
        # Set orders
        if isinstance(size, np.ndarray):
            size = np.array(size)[order]
        color_source_vector = color_source_vector[order]
        color_vector = color_vector[order]
        coords = basis_values[:, dims][order, :]

        # if plotting multiple panels, get the ax from the grid spec
        # else use the ax value (either user given or created previously)
        if grid:
            ax = pl.subplot(grid[count])
            axs.append(ax)
        if frameon:
            ax.axis("off")
        if title is None:
            if value_to_plot is not None:
                ax.set_title(value_to_plot)
            else:
                ax.set_title("")
        else:
            try:
                ax.set_title(title[count])
            except IndexError:
                log.warning(
                    "The title list is shorter than the number of panels. "
                    "Using 'color' value instead for some plots."
                )
                ax.set_title(value_to_plot)

        if not categorical:
            vmin_float, vmax_float, vcenter_float, norm_obj = _get_vboundnorm(
                vmin, vmax, vcenter, norm, count, color_vector
            )
            normalize = check_colornorm(
                vmin_float,
                vmax_float,
                vcenter_float,
                norm_obj,
            )
        else:
            normalize = None

        scatter = (
            partial(ax.scatter, s=size, plotnonfinite=True)
            if scale_factor is None
            else partial(
                circles, s=size, ax=ax, scale_factor=scale_factor
            )  # size in circles is radius
        )

        if add_outline:
            # the default outline is a black edge followed by a
            # thin white edged added around connected clusters.
            # To add an outline
            # three overlapping scatter plots are drawn:
            # First black dots with slightly larger size,
            # then, white dots a bit smaller, but still larger
            # than the final dots. Then the final dots are drawn
            # with some transparency.

            bg_width, gap_width = outline_width
            point = np.sqrt(size)
            gap_size = (point + (point * gap_width) * 2) ** 2
            bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2
            # the default black and white colors can be changes using
            # the contour_config parameter
            bg_color, gap_color = outline_color

            # remove edge from kwargs if present
            # because edge needs to be set to None
            kwargs["edgecolor"] = "none"

            # remove alpha for outline
            alpha = kwargs.pop("alpha") if "alpha" in kwargs else None

            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=bg_size,
                marker=".",
                c=bg_color,
                rasterized=True,
                norm=normalize,
                **kwargs,
            )
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=gap_size,
                marker=".",
                c=gap_color,
                rasterized=True,
                norm=normalize,
                **kwargs,
            )
            # if user did not set alpha, set alpha to 0.7
            kwargs["alpha"] = 0.7 if alpha is None else alpha

        cax = scatter(
            coords[:, 0],
            coords[:, 1],
            marker=".",
            c=color_vector,
            rasterized=True,
            norm=normalize,
            **kwargs,
        )

        # remove y and x ticks
        ax.set_yticks([])
        ax.set_xticks([])

        # set default axis_labels
        axis_labels = ["UMAP" + str(d + 1) for d in dims]

        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.autoscale_view()

        if value_to_plot is None:
            # if only dots were plotted without an associated value
            # there is not need to plot a legend or a colorbar
            continue

        if legend_fontoutline is not None:
            path_effect = [
                patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")
            ]
        else:
            path_effect = None

        # Adding legends
        if categorical:
            _add_categorical_legend(
                ax,
                color_source_vector,
                palette=_get_palette(adata, value_to_plot),
                scatter_array=coords,
                legend_loc=legend_loc,
                legend_fontweight=legend_fontweight,
                legend_fontsize=legend_fontsize,
                legend_fontoutline=path_effect,
                na_color=na_color,
                na_in_legend=na_in_legend,
                multi_panel=bool(grid),
            )
        elif colorbar_loc is not None:
            pl.colorbar(
                cax, ax=ax, pad=0.01, fraction=0.08, aspect=30, location=colorbar_loc
            )

    if return_fig is True:
        return fig
    axs = axs if grid else ax
    if show is False:
        return axs


def _panel_grid(hspace, wspace, ncols, num_panels):
    from matplotlib import gridspec

    n_panels_x = min(ncols, num_panels)
    n_panels_y = np.ceil(num_panels / n_panels_x).astype(int)
    # each panel will have the size of rcParams['figure.figsize']
    fig = pl.figure(
        figsize=(
            n_panels_x * rcParams["figure.figsize"][0] * (1 + wspace),
            n_panels_y * rcParams["figure.figsize"][1],
        ),
    )
    left = 0.2 / n_panels_x
    bottom = 0.13 / n_panels_y
    gs = gridspec.GridSpec(
        nrows=n_panels_y,
        ncols=n_panels_x,
        left=left,
        right=1 - (n_panels_x - 1) * left - 0.01 / n_panels_x,
        bottom=bottom,
        top=1 - (n_panels_y - 1) * bottom - 0.1 / n_panels_y,
        hspace=hspace,
        wspace=wspace,
    )
    return fig, gs


def _get_vboundnorm(
    vmin,
    vmax,
    vcenter,
    norm: Sequence[Normalize],
    index: int,
    color_vector: Sequence[float],
) -> Tuple[Union[float, None], Union[float, None]]:
    """
    Evaluates the value of vmin, vmax and vcenter, which could be a
    str in which case is interpreted as a percentile and should
    be specified in the form 'pN' where N is the percentile.
    Eg. for a percentile of 85 the format would be 'p85'.
    Floats are accepted as p99.9

    Alternatively, vmin/vmax could be a function that is applied to
    the list of color values (`color_vector`).  E.g.

    def my_vmax(color_vector): np.percentile(color_vector, p=80)


    Parameters
    ----------
    index
        This index of the plot
    color_vector
        List or values for the plot

    Returns
    -------

    (vmin, vmax, vcenter, norm) containing None or float values for
    vmin, vmax, vcenter and matplotlib.colors.Normalize  or None for norm.

    """
    out = []
    for v_name, v in [("vmin", vmin), ("vmax", vmax), ("vcenter", vcenter)]:
        if len(v) == 1:
            # this case usually happens when the user sets eg vmax=0.9, which
            # is internally converted into list of len=1, but is expected that this
            # value applies to all plots.
            v_value = v[0]
        else:
            try:
                v_value = v[index]
            except IndexError:
                log.error(
                    f"The parameter {v_name} is not valid. If setting multiple {v_name} values,"
                    f"check that the length of the {v_name} list is equal to the number "
                    "of plots. "
                )
                v_value = None

        if v_value is not None:
            if isinstance(v_value, str) and v_value.startswith("p"):
                try:
                    float(v_value[1:])
                except ValueError:
                    log.error(
                        f"The parameter {v_name}={v_value} for plot number {index + 1} is not valid. "
                        f"Please check the correct format for percentiles."
                    )
                # interpret value of vmin/vmax as quantile with the following syntax 'p99.9'
                v_value = np.nanpercentile(color_vector, q=float(v_value[1:]))
            elif callable(v_value):
                # interpret vmin/vmax as function
                v_value = v_value(color_vector)
                if not isinstance(v_value, float):
                    log.error(
                        f"The return of the function given for {v_name} is not valid. "
                        "Please check that the function returns a number."
                    )
                    v_value = None
            else:
                try:
                    float(v_value)
                except ValueError:
                    log.error(
                        f"The given {v_name}={v_value} for plot number {index + 1} is not valid. "
                        f"Please check that the value given is a valid number, a string "
                        f"starting with 'p' for percentiles or a valid function."
                    )
                    v_value = None
        out.append(v_value)
    out.append(norm[0] if len(norm) == 1 else norm[index])
    return tuple(out)


def _components_to_dimensions(
    components: Optional[Union[str, Collection[str]]],
    dimensions: Optional[Union[Collection[int], Collection[Collection[int]]]],
    *,
    total_dims: int,
) -> List[Collection[int]]:
    """Normalize components/ dimensions args for embedding plots."""
    # TODO: Deprecate components kwarg
    ndims = 2
    if components is None and dimensions is None:
        dimensions = [tuple(i for i in range(ndims))]
    elif components is not None and dimensions is not None:
        raise ValueError("Cannot provide both dimensions and components")

    # TODO: Consider deprecating this
    # If components is not None, parse them and set dimensions
    if components == "all":
        dimensions = list(combinations(range(total_dims), ndims))
    elif components is not None:
        if isinstance(components, str):
            components = [components]
        # Components use 1 based indexing
        dimensions = [[int(dim) - 1 for dim in c.split(",")] for c in components]

    if all(isinstance(el, Integral) for el in dimensions):
        dimensions = [dimensions]
    # if all(isinstance(el, Collection) for el in dimensions):
    for dims in dimensions:
        if len(dims) != ndims or not all(isinstance(d, Integral) for d in dims):
            raise ValueError()

    return dimensions


def _add_categorical_legend(
    ax,
    color_source_vector,
    palette: dict,
    legend_loc: str,
    legend_fontweight,
    legend_fontsize,
    legend_fontoutline,
    multi_panel,
    na_color,
    na_in_legend: bool,
    scatter_array=None,
):
    """Add a legend to the passed Axes."""
    if na_in_legend and pd.isnull(color_source_vector).any():
        if "NA" in color_source_vector:
            raise NotImplementedError(
                "No fallback for null labels has been defined if NA already in categories."
            )
        color_source_vector = color_source_vector.add_categories("NA").fillna("NA")
        palette = palette.copy()
        palette["NA"] = na_color
    cats = color_source_vector.categories

    if multi_panel is True:
        # Shrink current axis by 10% to fit legend and match
        # size of plots that are not categorical
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.91, box.height])

    if legend_loc == "right margin":
        for label in cats:
            ax.scatter([], [], c=palette[label], label=label)
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=(1 if len(cats) <= 14 else 2 if len(cats) <= 30 else 3),
            fontsize=legend_fontsize,
        )
    elif legend_loc == "on data":
        # identify centroids to put labels

        all_pos = (
            pd.DataFrame(scatter_array, columns=["x", "y"])
            .groupby(color_source_vector, observed=True)
            .median()
            # Have to sort_index since if observed=True and categorical is unordered
            # the order of values in .index is undefined. Related issue:
            # https://github.com/pandas-dev/pandas/issues/25167
            .sort_index()
        )

        for label, x_pos, y_pos in all_pos.itertuples():
            ax.text(
                x_pos,
                y_pos,
                label,
                weight=legend_fontweight,
                verticalalignment="center",
                horizontalalignment="center",
                fontsize=legend_fontsize,
                path_effects=legend_fontoutline,
            )


def _get_color_source_vector(
    adata, value_to_plot, use_raw=False, gene_symbols=None, layer=None, groups=None
):
    """
    Get array from adata that colors will be based on.
    """
    if value_to_plot is None:
        # Points will be plotted with `na_color`. Ideally this would work
        # with the "bad color" in a color map but that throws a warning. Instead
        # _color_vector handles this.
        # https://github.com/matplotlib/matplotlib/issues/18294
        return np.broadcast_to(np.nan, adata.n_obs)
    if (
        gene_symbols is not None
        and value_to_plot not in adata.obs.columns
        and value_to_plot not in adata.var_names
    ):
        # We should probably just make an index for this, and share it over runs
        value_to_plot = adata.var.index[adata.var[gene_symbols] == value_to_plot][
            0
        ]  # TODO: Throw helpful error if this doesn't work
    if use_raw and value_to_plot not in adata.obs.columns:
        values = adata.raw.obs_vector(value_to_plot)
    else:
        values = adata.obs_vector(value_to_plot, layer=layer)
    if groups and is_categorical_dtype(values):
        values = values.replace(values.categories.difference(groups), np.nan)
    return values


def _get_palette(adata, values_key: str, palette=None):
    color_key = f"{values_key}_colors"
    values = pd.Categorical(adata.obs[values_key])
    if palette:
        _set_colors_for_categorical_obs(adata, values_key, palette)
    elif color_key not in adata.uns or len(adata.uns[color_key]) < len(values.categories):
        #  set a default palette in case that no colors or few colors are found
        _set_default_colors_for_categorical_obs(adata, values_key)
    else:
        _validate_palette(adata, values_key)
    return dict(zip(values.categories, adata.uns[color_key]))


def _color_vector(
    adata, values_key: str, values, palette, na_color="lightgray"
) -> Tuple[np.ndarray, bool]:
    """
    Map array of values to array of hex (plus alpha) codes.

    For categorical data, the return value is list of colors taken
    from the category palette or from the given `palette` value.

    For continuous values, the input array is returned (may change in future).
    """
    ###
    # when plotting, the color of the dots is determined for each plot
    # the data is either categorical or continuous and the data could be in
    # 'obs' or in 'var'
    to_hex = partial(colors.to_hex, keep_alpha=True)
    if values_key is None:
        return np.broadcast_to(to_hex(na_color), adata.n_obs), False
    if not is_categorical_dtype(values):
        return values, False
    else:  # is_categorical_dtype(values)
        color_map = {
            k: to_hex(v)
            for k, v in _get_palette(adata, values_key, palette=palette).items()
        }
        # If color_map does not have unique values, this can be slow as the
        # result is not categorical
        color_vector = pd.Categorical(values.map(color_map))

        # Set color to 'missing color' for all missing values
        if color_vector.isna().any():
            color_vector = color_vector.add_categories([to_hex(na_color)])
            color_vector = color_vector.fillna(to_hex(na_color))
        return color_vector, True


def _broadcast_args(*args):
    """Broadcasts arguments to a common length."""
    from itertools import repeat

    lens = [len(arg) for arg in args]
    longest = max(lens)
    if not (set(lens) == {1, longest} or set(lens) == {longest}):
        raise ValueError(f"Could not broadast together arguments with shapes: {lens}.")
    return list(
        [[arg[0] for _ in range(longest)] if len(arg) == 1 else arg for arg in args]
    )
