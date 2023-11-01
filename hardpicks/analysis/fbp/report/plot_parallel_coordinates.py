import pandas as pd

from hardpicks.analysis.mlflow_and_orion_analysis.orion_utils import (
    get_orion_experiment_dataframe,
)
from hardpicks.analysis.mlflow_and_orion_analysis.parallel_coordinates_plotter import (
    ParallelCoordinatesPlotter,
)
from hardpicks.analysis.mlflow_and_orion_analysis.parallel_coordinates_utils import (
    process_orion_dataframe,
    parallel_plot_layout,
)

fold = "foldC"
experiment_name = "annotation_fix"
extra_experiment_name_foldC = "annotation_fix_v2"
new_objective_name = "HitRate@1px"

if __name__ == "__main__":

    path_to_db = f"/Users/bruno/monitoring/orion/{fold}/orion_db.pkl"
    orion_df = get_orion_experiment_dataframe(path_to_db, experiment_name)
    df = process_orion_dataframe(orion_df, new_objective_name)

    if fold == "foldC":
        orion_df = get_orion_experiment_dataframe(
            path_to_db, extra_experiment_name_foldC
        )
        df2 = process_orion_dataframe(orion_df, new_objective_name)
        df = (
            pd.concat([df, df2])
            .sort_values(by=new_objective_name, ascending=False)
            .head(50)
        )

    plotter = ParallelCoordinatesPlotter(new_objective_name)
    fig = plotter.get_figure(df)
    fig.update_layout(parallel_plot_layout)
    fig.show("chrome")
