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
Method for sequences comparison and alignment.
"""

from dtw import DTW
from dtw.metrics import angular_dist
from dtw.metrics import euclidean_dist
from dtw.metrics import mixed_dist
from dtw.tasks.logger import get_logger
from dtw.tasks.search_free_ends import brute_force_free_ends_search

#: List of valid values for `constraint` parameter.
CONSTRAINTS = {"merge_split", "edit_distance", "asymmetric", "symmetric"}
#: Default value for `constraint` parameter.
DEF_CONSTRAINT = 'merge_split'
#: List of valid values for `dist_type` parameter.
DIST_TYPES = {"euclidean", "angular", "mixed"}
#: Default value for `dist_type` parameter.
DEF_DIST_TYPE = 'euclidean'
#: Default value for `free_ends` parameter.
DEF_FREE_ENDS = (0, 1)
#: Default value for `free_ends_eps` parameter.
DEF_FREE_ENDS_EPS = 1e-4
#: Default value for `beam_size` parameter.
DEF_BEAMSIZE = -1
#: Default value for `delins_cost` parameter.
DEF_DELINS_COST = (1., 1.)
#: Default value for `max_stretch` parameter.
DEF_MAX_STRETCH = 3


# Call this function from outside to launch the comparison of two sequences
def sequence_comparison(seq_test, seq_ref, constraint=DEF_CONSTRAINT, dist_type=DEF_DIST_TYPE,
                        free_ends=DEF_FREE_ENDS, free_ends_eps=DEF_FREE_ENDS_EPS, beam_size=DEF_BEAMSIZE,
                        delins_cost=DEF_DELINS_COST, max_stretch=DEF_MAX_STRETCH,
                        mixed_type=None, mixed_spread=None, mixed_weight=None, **kwargs):
    """Run the DTW comparison between two angles & inter-nodes sequences.

    Phylotaxis comparison by means of angles & inter-nodes sequences alignment & comparison.

    Parameters
    ----------
    seq_test, seq_ref : numpy.ndarray
        Arrays of angles & inter-nodes to compare, respectively of shape ``(N_test, 2)`` and ``(N_ref, 2)``.
    constraint : {"merge_split", "edit_distance", "asymmetric", "symmetric"}, optional
        Type of constraint to use, default "merge_split".
    dist_type : {"euclidean", "angular", "mixed"}, optional
        Type of distance to use, default "euclidean".
    free_ends : float or tuple of int, optional
        A tuple of 2 integers ``(k, l)`` that specifies relaxation bounds on the alignment of sequences endpoints:
        relaxed by ``k`` at the sequence beginning and relaxed by ``l`` at the sequence ending, default ``(0, 1)``.
        A float corresponds to a percentage of sequence length for max exploration of `free_ends` on both sides,
        and in that case ``free_ends <= 0.4``.
    free_ends_eps : float, optional
        Minimum difference to previous minimum normalized cost to consider tested free-ends as the new best combination.
        Default is ``1e-4``.
    beam_size : int, optional
        Maximum amount of distortion allowed for signal warping, default ``-1``.
    delins_cost : tuple of float, optional
        Deletion and insertion costs, default ``(1., 1.)``.
    max_stretch : bool, optional
        Maximum amount of stretching allowed for signal warping, default ``3``.
    mixed_type : list(bool), optional
        A boolean vector, of size ``2``, indicating whether the k^th component should be treated
        as an angle (``True``) or a regular scalar value (``False``).
    mixed_spread : list(float), optional
        A vector of positive scalars, of size ``2``, used to normalize the distance values computed
        for each component with their typical spread.
    mixed_weight : list(float), optional
        A vector of positive weights, of size ``2``. Does not necessarily sum to 1, but normalized if not.

    Notes
    -----
    For the `free_ends` as a 2-tuple of integers ``(k, l)``, we must have:

      - ``k + l < min(N_test, N_ref)``
      - ``k >= 0`` and ``l >= 1``

    Examples
    --------
    >>> import numpy as np
    >>> from dtw.tasks.compare_sequences import sequence_comparison
    >>> seq_test = np.array([[123, 169, 224, 103, 131, 143, 113, 163, 148, 11, 153, 164, 118, 139, 135, 125, 147, 174, 121, 91, 127, 124], [70, 1, 32, 15, 56, 42, 39, 46, 4, 29, 29, 10, 12, 30, 0, 14, 12, 15, 0, 0, 12, 0]]).T
    >>> seq_ref = np.array([[123, 136, 131, 143, 113, 163, 159, 153, 164, 118, 139, 135, 125, 147, 174, 121, 91, 127, 124, 152, 124, 107, 126], [70, 48, 56, 42, 39, 46, 33, 29, 10, 12, 30, 0, 14, 12, 15, 0, 0, 12, 0, 13, 16, 0, 1]]).T
    >>> # Get the max value for inter-nodes, used by `mixed_spread`
    >>> max_ref = np.max(seq_ref[:, 1])
    >>> max_test = np.max(seq_test[:, 1])
    >>> # Update the keyword arguments to use with this type of distance
    >>> mixed_kwargs = {'mixed_type': [True, False], 'mixed_weight': [0.5, 0.5], 'mixed_spread': [1, max(max_ref, max_test)]}
    >>> dtwcomputer = sequence_comparison(seq_test, seq_ref, dist_type='mixed', **mixed_kwargs)
    >>> df = dtwcomputer.print_results()
    >>> print(df)

    """
    logger = kwargs.pop('logger', None)
    if logger is None:
        logger = get_logger(__name__)
    else:
        logger.name = __name__.split('.')[-1]

    try:
        assert constraint in CONSTRAINTS
    except AssertionError:
        logger.info(f"Valid values for `constraint` parameter: {CONSTRAINTS}")
        raise ValueError(f"Unknown '{constraint}' value for `constraint` parameter.")
    try:
        assert dist_type in DIST_TYPES
    except AssertionError:
        logger.info(f"Valid values for `dist_type` parameter: {DIST_TYPES}")
        raise ValueError(f"Unknown '{dist_type}' value for `dist_type` parameter.")

    if dist_type == "euclidean":
        ld = euclidean_dist
    elif dist_type == "angular":
        ld = angular_dist
    else:  # mixed normalized distance
        ld = mixed_dist

    dtwcomputer = DTW(seq_test, seq_ref, constraints=constraint, ldist=ld, mixed_type=mixed_type,
                      mixed_spread=mixed_spread, mixed_weight=mixed_weight, beam_size=beam_size,
                      max_stretch=max_stretch, delins_cost=delins_cost, **kwargs)
    if type(free_ends) == tuple:
        # if `free_ends` is tuple: NOT AUTOMATIC --> uses the tuple values
        dtwcomputer.free_ends = free_ends
    else:
        # if `free_ends` is NOT tuple: AUTOMATIC --> it is assumed to be int or real <= 0.4
        # and this corresponds to a percentage of sequence length for max exploration of free-ends
        free_ends, norm_dist = brute_force_free_ends_search(dtwcomputer, max_value=free_ends,
                                                            free_ends_eps=free_ends_eps, logger=logger)
        # finally computes the DTW for free-ends found
        dtwcomputer.free_ends = free_ends

    _ = dtwcomputer.run()

    return dtwcomputer
