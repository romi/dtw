# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       File author(s):
#           Christophe Godin <christophe.godin@inria.fr>
#
#       File contributor(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       File maintainer(s):
#           Christophe Godin <christophe.godin@inria.fr>
#
#       Mosaic Inria team, RDP Lab, Lyon
# ------------------------------------------------------------------------------

"""Utilities module.
"""
from os import mkdir
from os.path import abspath
from os.path import dirname
from os.path import exists
from os.path import join

_ROOT = abspath(join(dirname(__file__), '..', '..'))
DATA_DIR = abspath(join(_ROOT, 'data-analysis'))


def shared_folder(subdir=''):
    """Get the absolute path to the shared folder ``dtw/data-analysis/``.

    Parameters
    ----------
    subdir : str, optional
        Data subdirectory to use, will be created if missing.

    Returns
    -------
    str
        the absolute path to shared folder

    Examples
    --------
    >>> from dtw.util import shared_folder
    >>> shared_folder()
    >>> shared_folder('DB_eval_v1')

    """
    shared_dir = abspath(join(DATA_DIR, subdir))
    if not exists(shared_dir):
        mkdir(shared_dir)

    return shared_dir


def shared_data(filename, subdir=''):
    """Get absolute path to given ``filename`` must be a shared data.

    Append given ``filename`` to the absolute path to the shared folder ``dtw/data-analysis/``.

    Parameters
    ----------
    filename : str
        Name of a shared file found in the ``shared_folder()``.
    subdir : str, optional
        Subdirectory to use, will be created if missing.

    Returns
    -------
    str
        absolute path to the filename

    Examples
    --------
    >>> from dtw.util import shared_data
    >>> gt_csv = shared_data("groundtruth.csv", "DB_eval_v1")
    >>> pred_csv = shared_data("predicted_v0.4.csv", "DB_eval_v1")

    """
    shared_dir = shared_folder(subdir)
    return join(shared_dir, str(filename))


def default_test_db():
    """Returns the default database, ground-truth and prediction, to use for testing and demo.

    Returns
    -------
    str
        Path to the ground-truth CSV with angles and inter-nodes data.
    str
        Path to the prediction CSV with angles and inter-nodes data.

    Examples
    --------
    >>> from dtw.util import default_test_db
    >>> gt_csv, pred_csv = default_test_db()

    """
    return shared_data('groundtruth.csv', 'DB_eval_v1'), shared_data('predicted_v0.4.csv', 'DB_eval_v1')