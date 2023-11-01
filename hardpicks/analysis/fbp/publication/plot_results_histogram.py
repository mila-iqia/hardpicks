"""This script plots a 3 x 3 grid of histograms for foldA...foldK."""
import os
import pickle

import matplotlib.pylab as plt
import numpy as np


from hardpicks.analysis.fbp.publication.path_constants import style_path, \
    output_directory, pickles_directory

plt.style.use(style_path)

image_path = os.path.join(str(output_directory), "orion_run_results_histograms.png")

# plots can be fiddly. It's faster to have the needed data on hand.
results_pickle_path = str(pickles_directory.joinpath("results.pkl"))

early_stopping_metric_name = "valid/HitRate1px"

if __name__ == "__main__":
    with open(results_pickle_path, 'rb') as f:
        results_dict = pickle.load(f)

    alpha = 0.75

    fig = plt.figure(figsize=(7.2, 4.45))
    # fig.suptitle("Performance Distribution")

    bins = np.linspace(0, 100, 21)

    counter = 0
    for fold, fold_df in results_dict.items():
        counter += 1
        ax = fig.add_subplot(3, 3, counter)

        fold_string = fold.replace('fold', 'Fold ')
        assert len(fold_df) == 50, f"Less than 50 results present for {fold}."
        title = fold_string

        ax.set_title(title, loc='center', y=1.0)
        (100 * fold_df[early_stopping_metric_name]).hist(bins=bins, ax=ax, color='blue', alpha=alpha)
        ax.set_xlim([0, 100])

        if counter in {1, 4, 7}:
            ax.set_ylabel("count")

        if counter in {7, 8, 9}:
            ax.set_xlabel("HR@1px (%)")
        ax.grid(False)

    fig.tight_layout()

    fig.savefig(image_path)
    plt.close(fig)
