import os
import pickle

import pandas as pd

from hardpicks.analysis.fbp.report.path_constants import pickles_directory, output_directory

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


metrics_renaming_dictionary = {'Site': ('DEL1', 'Site'),
                               'Th.': ('DEL2', 'Th.'),
                               'HitRate1px': ('HR@', '1px'),
                               'HitRate3px': ('HR@', '3px'),
                               'HitRate5px': ('HR@', '5px'),
                               'HitRate7px': ('HR@', '7px'),
                               'HitRate9px': ('HR@', '9px'),
                               'GatherCoverage': ('DEL3', 'TC'),
                               'RootMeanSquaredError': ('DEL4', 'RMSE'),
                               'MeanAbsoluteError': ('DEL5', 'MAE'),
                               'MeanBiasError': ('DEL6', 'MBE')}

postprocessing_substitutions_map = dict()
for i in range(1, 7):
    postprocessing_substitutions_map[f'DEL{i}'] = '    '

filename_mapping_dict = {
    "Lalor_autopicks_th0_3.pic": ("Lalor", 0.3),
    "Lalor_autopicks_th1_0.pic": ("Lalor", 1.0),
    "Brunswick_auto_picks_for_Mila_th0_3.pic": ("Brunswick", 0.3),
    "Brunswick_auto_picks_for_Mila_th1_0.pic": ("Brunswick", 1.0),
    "MatagamiWest3D_part_autopicks_th0_3.pic": ("Matagami", 0.3),
    "MatagamiWest3D_part_autopicks_th1_0.pic": ("Matagami", 1.0),
    "Sudbury_picks_forMila_max_trough_th0_3.pic": ("Sudbury", 0.3),
    "Sudbury_picks_forMila_max_trough_th1_0.pic": ("Sudbury", 1.0),
    "Halfmile3D_auotpicks_th0_3.pic": ("Halfmile", 0.3),
    "Halfmile3D_auotpicks_th1_0.pic": ("Halfmile", 1.0),
}

output_path = os.path.join(str(output_directory), "baseline_results_table.tex")
pickle_dump_path = pickles_directory.joinpath("baseline_summaries.pkl")


if __name__ == "__main__":
    with open(pickle_dump_path, "rb") as f:
        results_dict = pickle.load(f)

    list_rows = []
    for filename, row in results_dict.items():
        site_name, threshold = filename_mapping_dict[filename]
        row["Site"] = site_name
        row["Th."] = threshold
        list_rows.append(row)

    desired_columns = list(metrics_renaming_dictionary.values())
    output_df = (
        pd.DataFrame(list_rows)
        .rename(columns=metrics_renaming_dictionary)[desired_columns]
        .reset_index(drop=True)
    )

    for col in desired_columns:
        if 'HR@' in col[0] or 'TC' in col[1]:
            output_df[col] = 100 * output_df[col]
    output_df = output_df.round(decimals=1)

    latex_output = output_df.to_latex(index=False, multicolumn=True, multicolumn_format='c')
    for key, value in postprocessing_substitutions_map.items():
        latex_output = latex_output.replace(key, value)

    with open(output_path, "w") as f:
        f.write(latex_output)
