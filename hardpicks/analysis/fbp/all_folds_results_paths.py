import glob
from collections import namedtuple
from pathlib import Path

ResultsAndSiteInfo = namedtuple(
    "ResultsAndSiteInfo",
    ["site_name", "fold", "dataset", "number_of_samples", "path_to_data_pickle"],
)

_samples_dict = dict(
    Sudbury=1001,
    Halfmile=751,
    Lalor=1501,
    Brunswick=751,
    Matagami=1001,
    Kevitsa=1001,
)
_valid_dict = dict(
    foldA="Halfmile",
    foldB="Sudbury",
    foldC="Brunswick",
    foldD="Lalor",
    foldE="Kevitsa",
    foldH="Brunswick",
    foldI="Lalor",
    foldJ="Sudbury",
    foldK="Halfmile",
)

_test_dict = dict(
    foldA="Kevitsa",
    foldB="Halfmile",
    foldC="Sudbury",
    foldD="Brunswick",
    foldE="Lalor",
    foldH="Sudbury",
    foldI="Brunswick",
    foldJ="Halfmile",
    foldK="Lalor",
)

_list_folds = [
    "foldA",
    "foldB",
    "foldC",
    "foldD",
    "foldE",
    "foldH",
    "foldI",
    "foldJ",
    "foldK",
]


def get_all_folds_results_pickle_path_and_info(predictions_base_dir: Path):
    """Get all folds results pickle path and info.

    This method handles the book-keeping of extracting the paths
    of Evaluator pickle paths for every fold, as well as returning needed
    site information. This assumes that all the pickle data is available.

    Args:
        predictions_base_dir (Path): path where all the folds results are located.

    Returns:
        list[ResultsAndSiteInfo] : all the info we need to do post-processing analysis.
    """
    list_results_and_site_info = []
    for fold in _list_folds:
        for dataset, data_dict in zip(["valid", "test"], [_valid_dict, _test_dict]):
            paths = glob.glob(str(predictions_base_dir / fold / f"*_{dataset}.pkl"))
            assert len(paths) == 1, "More than one file matches! Review data folder."
            pickle_path = paths[0]
            site_name = data_dict[fold]
            number_of_samples = _samples_dict[site_name]

            info = ResultsAndSiteInfo(
                site_name=site_name,
                fold=fold,
                dataset=dataset,
                number_of_samples=number_of_samples,
                path_to_data_pickle=pickle_path,
            )

            list_results_and_site_info.append(info)

    return list_results_and_site_info
