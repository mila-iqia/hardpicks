import logging
import os
import re
import subprocess
import typing
import unicodedata
from datetime import date
from pathlib import Path
import pandas as pd

from hardpicks.utils.hash_utils import (
    get_hash_from_path,
    get_git_hash,
)

LOGGER = logging.getLogger(__name__)


def rsync_folder(source, target):  # pragma: no cover
    """Uses rsync to copy the content of source into target.

    :param source: (str) path to the source folder.
    :param target: (str) path to the target folder.
    """
    if not os.path.exists(target):
        os.makedirs(target)

    LOGGER.info("rsyincing {} to {}".format(source, target))
    subprocess.check_call(["rsync", "-avzq", source, target])


def get_front_page_series(file_path: Path, script_relative_path: Path) -> pd.Series:
    """Create reproducibility table.

    Create basic information to identify the what/when of the artefact.
    Input:
        file_path: path to the file being analysed.
        script_relative_path: path to the script being executed, relative to ROOT
    """
    today = date.today()
    hash = get_hash_from_path(file_path)
    git_revision = get_git_hash()

    front_page_series = pd.Series(
        [today, git_revision, str(script_relative_path), file_path.name, hash],
        index=["Date", "git revision", "script", "file name", "file md5 checksum"],
    )

    return front_page_series


def create_header_page(pdf, front_page_series):
    """Add header page to pdf with basic info.

    Input:
        pdf: the pdf document to which to add a table
        front_page_series: table to be added
    """
    pdf.add_page()

    effective_page_width = pdf.w - 2 * pdf.l_margin

    col_width = effective_page_width / 4

    pdf.set_font("Times", "B", 14.0)
    th = pdf.font_size
    pdf.ln(4 * th)
    # Document title centered, 'B'old, 14 pt
    pdf.cell(effective_page_width, 0.0, "Report", align="C")

    pdf.set_font("Times", "", 14.0)
    th = pdf.font_size
    pdf.ln(4 * th)

    for name, value in front_page_series.items():
        pdf.cell(col_width, th, str(name), border=1)
        pdf.cell(3 * col_width, th, str(value), border=1)
        pdf.ln(th)


def slugify(
    in_str: typing.AnyStr,
    allow_unicode: bool = False,
) -> str:
    """Converts a provided string into a file-name-compatible string.

    Taken from https://github.com/django/django/blob/master/django/utils/text.py

    Will convert the input string to ASCII if 'allow_unicode' is False. Will convert spaces or
    repeated dashes to single dashes. Will remove characters that aren't alphanumerics, underscores,
    or hyphens. Will convert to lowercase. Will also strip leading and trailing whitespace, dashes,
    and underscores.
    """
    value = str(in_str)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


class WorkDirectoryContextManager:
    """Context manager class used to change and revert the current working directory."""

    def __init__(self, new_work_dir: typing.Union[typing.AnyStr, Path]):
        """Saves the new working directory we'll be landing in once initialization is complete."""
        self.new_work_dir = str(new_work_dir)

    def __enter__(self):
        """Changes the working directory to the specified one, saving the previous one for later."""
        self.old_work_dir = os.getcwd()
        os.chdir(self.new_work_dir)

    def __exit__(self, etype, value, traceback):
        """Restores the original working directory."""
        os.chdir(self.old_work_dir)
