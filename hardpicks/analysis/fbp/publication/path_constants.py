from hardpicks import FBP_ARTIFACTS_DIR, PLOT_STYLE_PATH

publication_artifacts_directory = FBP_ARTIFACTS_DIR.joinpath("publication/")
publication_artifacts_directory.mkdir(exist_ok=True)

output_directory = publication_artifacts_directory.joinpath("images_and_tables/")
output_directory.mkdir(exist_ok=True)

pickles_directory = publication_artifacts_directory.joinpath("pickles/")
pickles_directory.mkdir(exist_ok=True)

style_path = str(PLOT_STYLE_PATH)
