import hashlib
import re
import typing

import git
from git import InvalidGitRepositoryError


def get_git_hash():
    """Returns hash of the latest commit of the current branch.

    Returns:
        str: git hash
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha
    except (InvalidGitRepositoryError, ValueError):
        commit_hash = 'git repository not found'
    return commit_hash


def read_in_chunks(file_object, chunk_size=1024):
    """Read a file object in chunks of size chunk_size.

    Input:
        file_object: an open file
        chunk_size(int): size of chunk

    Yields:
        the read data.
    """
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


def get_hash_from_path(filepath: typing.AnyStr):
    """Compute the md5 checksum of a file.

    Input:
        filepath(Path): location of file

    Returns:
        md5 hash: the checksum.
    """
    with open(str(filepath), "rb") as f:
        file_hash = hashlib.md5()
        for chunk in read_in_chunks(f):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def get_hash_from_params(*args, **kwargs):
    """Computes the md5 checksum of a given array of parameters.

    Arguments:
        Any combination of parameters that are hashable via their string representation.

    Returns:
        The checksum of the parameters.
    """
    # by default, will use the repr of all params but remove the 'at 0x00000000' addresses
    clean_str = re.sub(r" at 0x[a-fA-F\d]+", "", str(args) + str(kwargs))
    return hashlib.sha1(clean_str.encode()).hexdigest()
