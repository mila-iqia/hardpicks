from pathlib import Path

TEST_DIR = Path(__file__).parent
SMOKE_TEST_DIR = TEST_DIR.joinpath("smoke_test")
FBP_STANDALONE_SMOKE_TEST_DIR = SMOKE_TEST_DIR.joinpath("fbp/standalone/")
