from pathlib import Path

ROOT_DIR = Path(__file__).parent
TOP_DIR = ROOT_DIR.parent

PATH_TO_GPU_MONITORING_SCRIPT = TOP_DIR.joinpath("monitor_gpus.py")

CONFIG_DIR = TOP_DIR.joinpath("config/")
EXAMPLES_DIR = TOP_DIR.joinpath("examples/")
DATA_ROOT_DIR = TOP_DIR.joinpath("data/")
ANALYSIS_RESULTS_DIR = DATA_ROOT_DIR.joinpath("analysis_results/")

FBP_ROOT_DATA_DIR = DATA_ROOT_DIR.joinpath("fbp/")
FBP_BAD_GATHERS_DIR = FBP_ROOT_DATA_DIR.joinpath("bad_gathers/")
FBP_ARTIFACTS_DIR = FBP_ROOT_DATA_DIR.joinpath("artifacts/")
FBP_CACHE_DIR = FBP_ROOT_DATA_DIR.joinpath("cache/")
FBP_FOLDS_DIR = FBP_ROOT_DATA_DIR.joinpath("folds/")
FBP_DATA_DIR = FBP_ROOT_DATA_DIR.joinpath("data/")

PLOTTING_DIR = ROOT_DIR.joinpath("plotting/")
PLOT_STYLE_PATH = PLOTTING_DIR.joinpath("plot_style.txt")
