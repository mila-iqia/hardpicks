import torch
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import ConfusionMatrixDisplay


def plot_on_ax(
    ax,
    field,
    title,
    imshow_kwargs,
    show_colorbar=True,
    colorbar_tick_locations=None,
    colorbar_tick_labels=None,
    colorbar_label=None,
):
    """Convenience function to create an imshow with a well aligned color bar.

    This method wraps around ax.imshow. It is always tricky to remember how an array is plotted, exactly.

    Consider an example, with

        field = [[ 1,  2,  3],
                 [ 4,  5,  6],
                 [ 7,  8,  9],
                 [10, 11, 12]]

    This field has dimension (4, 3) and imshow will plot this literally (i.e., the top left corner will be 1,
    the top right corner will be 3, the bottom left corner will be 10 and the bottom right corner will be 12).
    In other words, the first dimension corresponds to the vertical axis of the plot and the second dimension
    corresponds to the horizontal axis of the plot.

    If the spatial coordinates are relevant, then we can capture them using the "extent" argument to imshow,
    which has the form:
        extent : floats (left, right, bottom, top)

    consider an example for field(x, y), where list_x = [40, 50, 60] and list_y = [-0.3, -0.2, -0.1, 0.]
    clearly, dx = 10 and dy = 0.1. To insure that the pixels in the imshow image are centered at the grid points
    implied by list_x and list_y, we must have
        left = list_x[0] - dx/2 = 35
        right = list_x[-1] + dx/2 = 65
        bottom = list_y[-1] + dy/2 = 0.05
        top = list_y[0] - dy/2 = -0.35

    and so extent = (35, 65, 0.05, -0.35).

    Args:
        ax: matplotlib ax object on which the image will be plotted
        field: a 2D numpy array containing the data that will be plotted (a field is a function of space).
        title: the ax title
        imshow_kwargs: dictionary of parameters to be passed to ax.imshow.
        show_colorbar: indicate if a colorbar should be added to the plot
        colorbar_tick_locations: (Optional) position of the ticks on the colorabar
        colorbar_tick_labels: (Optional) labels for the colorbar ticks.
        colorbar_label: (Optional) text label for the colorbar as a whole.

        NOTE that colorbar_tick_locations and colorbar_tick_labels must but be specified for the colorbar
        to be affected.

    Returns:
          [nothing]

    """
    im = ax.imshow(field, **imshow_kwargs)
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax, ax=ax, shrink=0.8)
        if colorbar_tick_locations is not None and colorbar_tick_labels is not None:
            cbar.set_ticks(colorbar_tick_locations)
            cbar.set_ticklabels(colorbar_tick_labels)
            if colorbar_label is not None:
                cbar.set_label(colorbar_label)
    ax.set_title(title, loc="center")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")


def get_vertical_and_horizontal_coordinates_from_slice_coordinates(slice_coords):
    """Utility method to get the 1 dimensional coordinates from the 3 dimensional slice_coords."""
    list_x = slice_coords[0, :, 0]
    list_y = slice_coords[:, 0, 1]
    return list_x, list_y


def get_meshgrid_spacing_from_slice_coordinates(slice_coords):
    """Get the mesh spacing in horizontal (x) and vertical (y) directions."""
    list_x, list_y = get_vertical_and_horizontal_coordinates_from_slice_coordinates(
        slice_coords
    )
    dx = list_x[1] - list_x[0]
    dy = list_y[1] - list_y[0]
    return dx, dy


def get_extent_from_slice_coordinates(slice_coords):
    """Get the extent array to be passed to imshow.

    Args:
        slice_coords: a numpy array of dimensions [Height, Width, 2] which contains the coordinates of each
                      pixel in a slice.

    Returns:
        extent : A list of the form [left, right, bottom, top], ready to be passed to imshow.
    """
    # Here we'll use "x" to mean the horizontal direction and "y" the vertical direction
    list_x, list_y = get_vertical_and_horizontal_coordinates_from_slice_coordinates(
        slice_coords
    )

    dx = list_x[1] - list_x[0]
    dy = list_y[1] - list_y[0]
    # extent: floats(left, right, bottom, top), optional
    extent = [
        list_x[0] - dx / 2,
        list_x[-1] + dx / 2,
        list_y[-1] + dy / 2,
        list_y[0] - dy / 2,
    ]

    return extent


def plot_spatial_correlations_on_ax(spatial_correlations, corr_extent, title, ax2D):
    """Plot the spatial correlations as an imshow."""
    min_value = np.min(spatial_correlations)
    max_value = np.max(spatial_correlations)
    vmax = np.max([np.abs(min_value), np.abs(max_value)])
    if min_value >= 0:
        vmin = 0.0
    else:
        vmin = -vmax
    plot_on_ax(
        ax2D,
        spatial_correlations,
        title,
        imshow_kwargs=dict(
            vmin=vmin, vmax=vmax, extent=corr_extent, cmap=plt.cm.RdBu_r
        ),
    )
    ax2D.set_xlabel("$R_h$ (m)")
    ax2D.set_ylabel("$R_v$ (m)")


def plot_vertical_and_horizontal_correlation_on_ax(
    spatial_correlations, corr_extent, title, ax1D
):
    """Plot a vertical and horizontal slice of the spatial correlation as 1D curves."""
    dim1, dim2 = spatial_correlations.shape
    vertical_slice = spatial_correlations[:, dim2 // 2]
    horizontal_slice = spatial_correlations[dim1 // 2, :]
    vertical_coordinates = np.linspace(corr_extent[0], corr_extent[1], dim1)
    horizontal_coordinates = np.linspace(corr_extent[2], corr_extent[3], dim2)
    xlim = np.max(corr_extent)
    ax1D.plot(vertical_coordinates, vertical_slice, c="blue", label="$R_h = 0$ slice")
    ax1D.plot(
        horizontal_coordinates, horizontal_slice, c="red", label="$R_v = 0$ slice"
    )
    ax1D.legend(loc=0)
    ax1D.set_xlim(-xlim, xlim)
    ax1D.set_xlabel("Offset (m)")
    ax1D.set_ylabel("Amplitude")
    ax1D.set_title(title, loc="center")


def plot_lithographic_model(
    parser_data, class_names, class_colors, extent, output_directory, show_colorbar=True
):
    """Plot the lithographic model for given data."""
    lithographic_model = parser_data["litho_label"]
    slice_name = parser_data["slice_name"].replace(".npz", "")

    colormap = mpl.colors.ListedColormap(class_colors)
    bounds = np.arange(len(class_colors) + 1)
    norm = mpl.colors.BoundaryNorm(bounds, colormap.N)

    colorbar_tick_locations = 0.5 + np.arange(len(class_colors))
    colorbar_tick_labels = [class_name.title() for class_name in class_names]

    imshow_kwargs = dict(extent=extent, cmap=colormap, norm=norm, interpolation="none")

    width_factor = 7.2
    height_factor = 4.45
    figsize = (0.8 * width_factor, height_factor)

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(111)
    plot_on_ax(
        ax1,
        lithographic_model,
        title="",
        imshow_kwargs=imshow_kwargs,
        show_colorbar=show_colorbar,
        colorbar_tick_locations=colorbar_tick_locations,
        colorbar_tick_labels=colorbar_tick_labels,
    )
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_ylim(extent[2], extent[3])
    fig.tight_layout()
    fig.savefig(output_directory.joinpath(f"litho_model_{slice_name}.png"))
    plt.close(fig)


def plot_reliability_diagram(calibration_df: pd.DataFrame):
    """Plot reliability diagram.

    Args:
        calibration_df (pd.DataFrame): dataframe assumed to contain the calibration information.

    Returns:
        fig (matplotlib figure): figure with the reliability diagrams
    """
    list_intervals = list(calibration_df.index)

    list_midpoints = []
    list_xticks = []

    first_interval = True
    for interval in list_intervals:
        if first_interval:
            list_xticks.append(interval.left)
            first_interval = False
        list_xticks.append(interval.right)
        list_midpoints.append(0.5 * (interval.left + interval.right))

    list_xticks = 100 * np.array(list_xticks)
    list_xtick_labels = []
    for i, xtick in enumerate(list_xticks):
        if i % 2 == 0:
            label = str(int(xtick))
        else:
            label = ''
        list_xtick_labels.append(label)

    list_midpoints = np.array(list_midpoints)
    list_sample_fraction = calibration_df["all count"] / calibration_df["all count"].sum()
    list_accuracy = calibration_df["correct count"] / calibration_df["all count"]
    list_mean_probability = calibration_df["mean probability"]

    ece = np.sum(np.abs(list_accuracy - list_mean_probability) * list_sample_fraction)

    fig = plt.figure(figsize=(7.2, 4.45))
    fig.suptitle("Reliability Diagram")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.bar(
        100 * list_midpoints,
        100 * list_sample_fraction,
        width=5,
        color="blue",
        edgecolor="k",
        alpha=0.5,
    )
    ax1.set_xlabel("Confidence (%)")
    ax1.set_ylabel("Fraction of Samples (%)")

    ax2.bar(
        100 * list_midpoints,
        100 * list_accuracy,
        width=5,
        color="blue",
        edgecolor="k",
        alpha=0.5,
    )

    ax2.plot([0, 100], [0, 100], lw=2, color="red", label="Perfect Calibration")
    ax2.plot(
        100 * list_mean_probability,
        100 * list_accuracy,
        "bo",
        label=f"ECE : {100*ece:2.1f} %",
    )

    ax1.set_xlim(0, 100)
    ax2.set_xlim(0, 100)
    ax2.legend(loc=0)
    ax2.set_xlabel("Confidence (%)")
    ax2.set_ylabel("Accuracy (%)")

    for ax in [ax1, ax2]:
        ax.set_xticks(list_xticks)
        ax.set_xticklabels(list_xtick_labels)

    fig.tight_layout()
    return fig


def get_confusion_matrix_figure(confusion_matrix: torch.Tensor):
    """Get confusion matrix figure.

    This function creates a normalized version of the confusion matrix where each
    row sums up to 1, representing the distribution of predicted classes for each
    true class.

    Args:
        confusion_matrix (torch.Tensor): confusion matrix to be plotted.

    Returns:
        fig (matplotlib figure): figure with the confusion matrix.
    """
    norm_confusion_matrix = confusion_matrix / confusion_matrix.sum(dim=-1, keepdim=True)
    norm_confusion_matrix = torch.nan_to_num(norm_confusion_matrix)
    confusion_matrix_numpy = norm_confusion_matrix.cpu().detach().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_numpy)
    confusion_matrix_plot = display.plot(ax=ax, values_format=".2f")
    return confusion_matrix_plot.figure_
