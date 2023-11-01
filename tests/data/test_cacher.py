import numpy as np

import hardpicks.data.cache_utils


def generate_fake_dataset():
    hyperparams = dict(
        shape0=np.random.randint(10, 100),
        shape1=np.random.randint(10, 100),
    )
    dataset = np.random.rand(hyperparams["shape0"], hyperparams["shape1"])
    return dataset, hyperparams


def test_simple_caching(tmpdir):
    np.random.seed(None)

    # first, create a single dataset, cache it, reload it, assert it's all the same
    dataset1, hyperparams1 = generate_fake_dataset()
    hyperparams1["dataset_name"] = "dataset1"
    cacher1 = hardpicks.data.cache_utils.DatasetCacher(
        hyperparams=hyperparams1, cache_dir_path=tmpdir, cache_name_prefix=None,
    )
    assert not cacher1.is_cache_available()
    cacher1.save(dataset1)
    assert cacher1.is_cache_available()
    dataset1_reloaded = cacher1.load()
    assert dataset1.shape == dataset1_reloaded.shape
    np.testing.assert_array_equal(dataset1, dataset1_reloaded)

    # next, create another cacher for the same dataset, but with a cache name prefix
    cacher2 = hardpicks.data.cache_utils.DatasetCacher(
        hyperparams=hyperparams1, cache_dir_path=tmpdir, cache_name_prefix="potato",
    )
    assert cacher1.hash == cacher2.hash
    assert not cacher2.is_cache_available()
    cacher2.save(dataset1)
    assert cacher2.is_cache_available()
    dataset1_reloaded = cacher2.load()
    assert dataset1.shape == dataset1_reloaded.shape
    np.testing.assert_array_equal(dataset1, dataset1_reloaded)

    # now, create another cacher with the same settings but for another dataset w/o overlap
    dataset2, hyperparams2 = generate_fake_dataset()
    hyperparams2["dataset_name"] = "dataset2"
    cacher3 = hardpicks.data.cache_utils.DatasetCacher(
        hyperparams=hyperparams2, cache_dir_path=tmpdir, cache_name_prefix=None,
    )
    assert cacher1.hash != cacher3.hash
    assert not cacher3.is_cache_available()
    cacher3.save(dataset2)
    assert cacher3.is_cache_available()
    dataset2_reloaded = cacher3.load()
    assert dataset2.shape == dataset2_reloaded.shape
    np.testing.assert_array_equal(dataset2, dataset2_reloaded)


def test_file_caching(tmpdir):
    np.random.seed(None)

    # start off with the same simple setup as in the previous test
    dataset1, hyperparams1 = generate_fake_dataset()
    hyperparams1["dataset_name"] = "dataset1"
    cacher1 = hardpicks.data.cache_utils.DatasetCacher(
        hyperparams=hyperparams1, cache_dir_path=tmpdir, cache_name_prefix=None,
    )
    assert not cacher1.is_cache_available()
    cacher1.save(dataset1)
    assert cacher1.is_cache_available()

    # next, create a second cacher with a dummy file to add to the hash
    # (for simplicity, we use the 1st dataset's cache as the dummy file)
    cacher2 = hardpicks.data.cache_utils.DatasetCacher(
        hyperparams=hyperparams1, cache_dir_path=tmpdir, cache_name_prefix=None,
        on_disk_file_paths=[cacher1.cache_path],
    )
    assert not cacher2.is_cache_available()
    assert cacher1.hash != cacher2.hash
    cacher2.save(dataset1)
    assert cacher2.is_cache_available()

    # finally, check that extra file paths do change the hash (beyond the first change)
    cacher3 = hardpicks.data.cache_utils.DatasetCacher(
        hyperparams=hyperparams1, cache_dir_path=tmpdir, cache_name_prefix=None,
        on_disk_file_paths=[cacher1.cache_path, cacher2.cache_path],
    )
    assert not cacher3.is_cache_available()
    assert cacher3.hash != cacher1.hash and cacher3.hash != cacher2.hash

    # as a bonus, test the manual cache reloading function
    dataset2, _ = hardpicks.data.cache_utils.reload_dataset_from_cache(
        cacher2.cache_path)
    np.testing.assert_array_equal(dataset1, dataset2)
