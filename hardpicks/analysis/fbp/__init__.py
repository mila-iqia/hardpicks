from pathlib import Path

FBP_ANALYSIS_DIR = Path(__file__).parent
FBP_ANALYSIS_ARTIFACTS_DIR = FBP_ANALYSIS_DIR.joinpath("artifacts")
FBP_ANALYSIS_ARTIFACTS_DIR.mkdir(exist_ok=True)
