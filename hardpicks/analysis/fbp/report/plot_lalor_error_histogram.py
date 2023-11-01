import pickle
import numpy as np
import matplotlib.pylab as plt


from hardpicks.analysis.fbp.report.path_constants import style_path, output_directory

data_dump_path = "/Users/bruno/monitoring/FBP/orion/foldD/mlruns/4/" \
                 "98cded47a0844c319014630e6bd6cf27/artifacts/data_dumps/" \
                 "output_best-epoch=12-step=24894_valid.pkl"

plt.style.use(style_path)

cutoff = 1200.


image_path = output_directory.joinpath('lalor_error_distribution.png')

if __name__ == '__main__':
    with open(data_dump_path, "rb") as fd:
        attribs = pickle.load(fd)

    df = attribs['dataframe']

    error_series = df['Errors']

    cutoff_mask = error_series.abs() < cutoff

    all_errors = error_series.values
    small_errors = error_series[cutoff_mask].values

    mae = np.nanmean(np.abs(all_errors))
    rmse = np.sqrt(np.nanmean(all_errors**2))
    mbe = np.nanmean(all_errors)

    small_mae = np.mean(np.abs(small_errors))
    small_rmse = np.sqrt(np.mean(small_errors**2))
    small_mbe = np.mean(small_errors)

    label = f" RMSE = {small_rmse:2.1f}\n" \
            f"  MAE = {small_mae: 2.1f}\n" \
            f"  MBE = {small_mbe:2.1f}"

    fig = plt.figure(figsize=(7.2, 4.45))

    fig.suptitle("Lalor Site: Prediction Error Distribution")
    ax = fig.add_subplot(111)
    ax.hist(error_series, color='blue', bins=50)
    ax.set_yscale('log')
    ymin, ymax = ax.set_ylim()
    ax.vlines(cutoff, ymin, ymax, linestyles='--', color='black', label=label)

    ax.set_xlim(xmax=2000)
    ax.legend(loc=0)
    ax.set_xlabel('Error (pixels)')
    ax.set_ylabel('Count')

    fig.savefig(image_path)
    plt.close(fig)
