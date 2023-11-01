from hardpicks import FBP_ARTIFACTS_DIR, PLOT_STYLE_PATH

workshop_artifacts_directory = FBP_ARTIFACTS_DIR.joinpath("workshop/")
workshop_artifacts_directory.mkdir(exist_ok=True)

output_directory = workshop_artifacts_directory.joinpath("images_and_tables/")
output_directory.mkdir(exist_ok=True)

pickles_directory = output_directory.joinpath("pickles/")
pickles_directory.mkdir(exist_ok=True)

style_path = str(PLOT_STYLE_PATH)
