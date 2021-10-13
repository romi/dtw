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

"""
Searching free-ends.
"""
import logging

import numpy as np
from joblib import Parallel
from joblib import delayed

from dtw.dtw import DTW
from dtw.dtw import _get_ndist


def brute_force_free_ends_search(dtw, max_value=0.4, free_ends_eps=1e-4, n_jobs=-1):
    """Explore all free-ends combinations in the prescribed limits.

    This is a brute force method to search the optimal free-ends values.
    The best model is selected using the lowest minimum normalized cost for a pair of free-ends.

    Parameters
    ----------
    max_value : float, optional
        max value for exploration of free-ends on both sequence sides. Default is ``0.4``.
    free_ends_eps : float, optional
        Minimum difference to previous minimum normalized cost to consider tested free-ends as the new best combination.
        Default is ``1e-4``.
    n_jobs : int, optional
        Number of jobs to run in parallel, by default `-1` uses all availables cores.

    Returns
    -------
    2-tuple of floats
        a length-2 tuple of left & right free-ends.
    float
        the corresponding nomalized distance

    Examples
    --------
    >>> import numpy as np
    >>> from dtw.dtw import DTW
    >>> from dtw.dtw import brute_force_free_ends_search
    >>> from dtw.dtw import mixed_dist
    >>> seq_test = np.array([[123, 169, 224, 103, 131, 143, 113, 163, 148, 11, 153, 164, 118, 139, 135, 125, 147, 174, 121, 91, 127, 124], [70, 1, 32, 15, 56, 42, 39, 46, 4, 29, 29, 10, 12, 30, 0, 14, 12, 15, 0, 0, 12, 0]]).T
    >>> seq_ref = np.array([[123, 136, 131, 143, 113, 163, 159, 153, 164, 118, 139, 135, 125, 147, 174, 121, 91, 127, 124, 152, 124, 107, 126], [70, 48, 56, 42, 39, 46, 33, 29, 10, 12, 30, 0, 14, 12, 15, 0, 0, 12, 0, 13, 16, 0, 1]]).T
    >>> max_ref = np.max(seq_ref[:, 1])
    >>> max_test = np.max(seq_test[:, 1])
    >>> dtwcomputer = DTW(seq_test,seq_ref,constraints='merge_split',ldist=mixed_dist,mixed_type=[True, False],mixed_spread=[1, max(max_ref, max_test)],mixed_weight=[0.5, 0.5])
    >>> free_ends, norm_dist = brute_force_free_ends_search(dtwcomputer)
    >>> print(free_ends)
    >>> dtwcomputer.free_ends = free_ends
    >>> ndist, path, length, ndistarray, backpointers = dtwcomputer.run()
    >>> dtwcomputer.get_results()
    >>> dtwcomputer.get_better_results()
    >>> dtwcomputer.plot_results()

    """
    assert isinstance(dtw, DTW)

    if max_value > 0.4:
        max_value = 0.4  # (max value for exploration of free-ends)
        logging.warning("Automatic free-ends capped to max 40% of min length on both sides.")

    # first find the limits of the tested free-ends
    Nmin = min(dtw.nX, dtw.nY)
    N = int(max_value * Nmin)

    kwargs = {
        "constraints": dtw.constraints,
        "delins_cost": dtw.delins_cost,
        "ldist": dtw.ldist_f,
        "mixed_type": dtw.mixed_type,
        "mixed_spread": dtw.mixed_spread,
        "mixed_weight": dtw.mixed_weight,
        "beamsize": dtw.beam_size,
        "max_stretch": dtw.max_stretch
    }
    free_ends = [(left_fe, right_fe + 1) for left_fe in range(N) for right_fe in range(N)]
    norm_dists = Parallel(n_jobs=n_jobs)(delayed(_get_ndist)(dtw.seqX, dtw.seqY, fe, **kwargs) for fe in free_ends)

    # return the free-ends for first occurrence of the min norm distance
    min_ndist = np.Infinity
    left_fe, right_fe = 0, 1
    for i, fe in enumerate(free_ends):
        ndist = norm_dists[i]
        if ndist < min_ndist - free_ends_eps:
            min_ndist = ndist
            left_fe, right_fe = fe

    return (left_fe, right_fe), min_ndist

