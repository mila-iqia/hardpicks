"""Dataset caching module.

The provided class (`DatasetCacher`) can be used to cache the pre-parsed
metadata in a dataset container so that it can be saved/loaded faster.
"""

import copy
import logging
import os
import pickle
import typing

import hardpicks.utils.hash_utils

logger = logging.getLogger(__name__)


class DatasetCacher:
    """Dataset cache-wrapper, compatible with the `typing.Sequence` interface.

    This object will compute a hash from a given list of dataset constructor hyperparameters
    and an optional list of on-disk files. This hash will later allow the corresponding dataset
    (instantiated externally) to be saved/loaded from disk. The file name on the cached file will
    be based on the combination of hashes from hyperparameters, on-disk files, and git commit sha1.

    NOTE THAT THIS CACHER SHOULD ONLY BE USED IF THE LOADING PROCESS FOR THE DATASET
    IS DETERMINISTIC AND ONLY DEPENDS ON THE PROVIDED HYPERPARAMETER DICTIONARY.

    Attributes:
        hyperparams: the dictionary of hyper parameters that uniquely identify the cached dataset.
        hash: the md5 checksum (hashtag) to be used as a unique identifier for the on-disk cache.
        cache_path: the location where the cache file will be saved/loaded from.
    """

    def __init__(
            self,
            hyperparams: typing.Dict[typing.AnyStr, typing.Any],
            cache_dir_path: typing.AnyStr,
            cache_name_prefix: typing.Optional[typing.AnyStr] = None,
            on_disk_file_paths: typing.Optional[typing.Iterable[typing.AnyStr]] = None,
    ):
        """Stores a local copy of the hyperparameter dictionary and computes its hashtag."""
        # note: all hyperparams should be string-representable to give proper hash values!
        self.hyperparams = copy.deepcopy(hyperparams)
        # first, create a hash-key-to-file-hash-checksum map for all dataset files (if any)
        if on_disk_file_paths is not None:
            self._append_file_hashes(self.hyperparams, on_disk_file_paths)
        # we'll also include the git sha1 of the current commit into the dict for good measure...
        self.hyperparams["__git_sha1"] = hardpicks.utils.hash_utils.get_git_hash()
        self.hash = hardpicks.utils.hash_utils.get_hash_from_params(
            **self.hyperparams)
        assert os.path.isdir(cache_dir_path), f"invalid caching directory: {cache_dir_path}"
        if cache_name_prefix is not None:
            cache_file_name = cache_name_prefix + "_" + str(self.hash) + ".pkl"
            self.cache_path = os.path.join(cache_dir_path, cache_file_name)
        else:
            self.cache_path = os.path.join(cache_dir_path, str(self.hash) + ".pkl")

    @staticmethod
    def _append_file_hashes(
            hyperparams: typing.Dict[typing.AnyStr, typing.Any],
            on_disk_file_paths: typing.Iterable[typing.AnyStr]
    ) -> None:
        """Appends the hash of the specified files to the hyperparam dictionary and returns it."""
        for file_path in on_disk_file_paths:
            assert os.path.isfile(file_path), f"invalid file path: {file_path}"
            # TODO : if this takes too long with huge datasets, we could relax it by using only file size?
            file_hash = hardpicks.utils.hash_utils.get_hash_from_path(file_path)
            assert file_path not in hyperparams, "invalid hyperparam dict (cannot overlap)"
            hyperparams[file_path] = file_hash

    def is_cache_available(self) -> bool:
        """Checks and returns whether a local cache file is available for this particular dataset."""
        return os.path.isfile(self.cache_path)

    def save(self, dataset: typing.Any) -> typing.AnyStr:
        """Saves the given dataset created using the originally provided hyperparams."""
        # note: this will FAIL if the dataset contains file descriptors or any non-dumpable object.
        with open(self.cache_path, "wb") as fd:
            pickle.dump((dataset, self.hyperparams), fd)
        return self.cache_path

    def load(self) -> typing.Any:
        """Loads the cached dataset, checks whether its internal hash still matches, and returns it."""
        assert self.is_cache_available(), f"cannot locate cache file at: {self.cache_path}"
        with open(self.cache_path, "rb") as fd:
            dataset, hyperparams = pickle.load(fd)
        hash = hardpicks.utils.hash_utils.get_hash_from_params(**hyperparams)
        assert hash == self.hash, "post-loading dataset hash validation failed"
        return dataset


def reload_dataset_from_cache(
        cache_path: typing.AnyStr,
) -> typing.Tuple[typing.Any, typing.Dict[typing.AnyStr, typing.Any]]:
    """Will reload a dataset from a given cache file directly.

    If the reloaded hyperparameters contain the path to an existing file, its hash will be verified
    to make sure the cache is still up-to-date with respect to the local copy.

    Arguments:
        cache_path: path to the cache file that contains the dataset as well as its hyperparam dict.

    Returns:
        A tuple of the reloaded dataset and its hyperparameter dictionary.
    """
    assert os.path.isfile(cache_path), f"invalid cache path: {cache_path}"
    with open(cache_path, "rb") as fd:
        dataset, hyperparams = pickle.load(fd)
    for hyperparam, val in hyperparams.items():
        if isinstance(hyperparam, str) and os.path.isfile(hyperparam):
            file_hash = hardpicks.utils.hash_utils.get_hash_from_path(
                hyperparam)
            assert file_hash == val, f"mismatch between cached and live hashes for: {hyperparam}"
    return dataset, hyperparams
