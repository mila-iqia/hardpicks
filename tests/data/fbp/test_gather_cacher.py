import numpy as np
import pytest

from hardpicks.data.cache_utils import (
    DatasetCacher,
    reload_dataset_from_cache,
)


@pytest.mark.slow
@pytest.mark.parametrize("real_site_name", ["Sudbury", "Halfmile"])
def test_cache_gather_dataset(tmpdir, real_site_info, real_site_path, real_dataset):
    if real_dataset is None:
        pytest.skip("missing hdf5 file for real-data test. Skipping test.")
    cacher = DatasetCacher(
        hyperparams=real_site_info,
        cache_dir_path=tmpdir,
        cache_name_prefix="gather_test",
        on_disk_file_paths=[real_site_path],
    )
    assert not cacher.is_cache_available()
    cacher.save(real_dataset)
    cached_dataset = cacher.load()
    # scroll through a bunch of samples to see if an assert triggers somewhere
    for sample_idx in range(len(real_dataset)):
        sample1, sample2 = real_dataset[sample_idx], cached_dataset[sample_idx]
        np.testing.assert_array_equal(sample1.keys(), sample2.keys())
        for key in sample1.keys():
            np.testing.assert_array_equal(sample1[key], sample2[key])
        if sample_idx > 100:
            break
    # as a final test, make sure we can reload via cache path directly
    dataset2, _ = reload_dataset_from_cache(cacher.cache_path)
    assert len(real_dataset) == len(dataset2)
