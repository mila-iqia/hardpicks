"""Parallel Coordinates Plotter.

This code is a more modular version of orion.plotting.backend_plotly.py. This latter
code makes it impossible to customize the graph's appearance properly.
"""

import plotly.graph_objects as go
from pandas import Categorical


class ParallelCoordinatesPlotter:
    """Class to plot parallel coordinates graphs from a pandas dataframe."""

    def __init__(self, objective_name, colorscale="YlOrRd"):
        """Initialize class.

        Arguments:
            objective_name: name of the objective that will be used to color plot. It should be a column name
                            in the pandas dataframe to be passed to the plotting method.
            colorscale: string describing the colors to be used. See plotly for details.
        """
        self.objective_name = objective_name
        self.colorscale = colorscale

    def _get_dimension(self, series):
        if type(series.values) == Categorical:
            dim_data = dict(label=series.name, values=series.values.codes)
            dim_data["tickvals"] = list(range(len(series.values.categories)))
            dim_data["ticktext"] = series.values.categories
        else:
            dim_data = dict(label=series.name, values=series.values)
            dim_data["range"] = (min(series.values), max(series.values))

        return dim_data

    def _get_all_dimensions(self, df):
        dimensions = []
        for name in df.columns:
            series = df[name]
            dimension_data = self._get_dimension(series)
            dimensions.append(dimension_data)
        return dimensions

    def get_figure(self, df):
        """Plot figure from a pandas dataframe."""
        objective = df[self.objective_name]
        omin = min(objective)
        omax = max(objective)

        dimensions = self._get_all_dimensions(df)

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=objective,
                    colorscale=self.colorscale,
                    showscale=True,
                    cmin=omin,
                    cmax=omax,
                    colorbar=dict(title=self.objective_name),
                ),
                dimensions=dimensions,
            )
        )

        return fig
