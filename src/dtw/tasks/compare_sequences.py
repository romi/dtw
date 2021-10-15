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
import logging

from dtw import DTW
from dtw.metrics import angular_dist
from dtw.metrics import euclidean_dist
from dtw.metrics import mixed_dist
from dtw.tasks.search_free_ends import brute_force_free_ends_search
from dtw.tasks.logger import get_logger

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
#: Default value for `beam_size` parameter.
DEF_BEAMSIZE = -1
#: Default value for `delins_cost` parameter.
DEF_DELINS_COST = (1., 1.)
#: Default value for `max_stretch` parameter.
DEF_MAX_STRETCH = 3


# Call this function from outside to launch the comparison of two sequences
def sequence_comparison(seqX, seqY, constraint=DEF_CONSTRAINT, dist_type=DEF_DIST_TYPE,
                        free_ends=DEF_FREE_ENDS, free_ends_eps=1e-4, beamsize=DEF_BEAMSIZE,
                        delins_cost=DEF_DELINS_COST, max_stretch=DEF_MAX_STRETCH,
                        mixed_type=None, mixed_spread=None, mixed_weight=None, **kwargs):
    """Run the DTW comparison between two angles & inter-nodes sequences.

    Phylotaxis comparison by means of angles & inter-nodes sequences alignment & comparison.

    Parameters
    ----------
    seqX, seqY : numpy.ndarray
        Arrays of angles & inter-nodes to compare.
    constraint : {"merge_split", "edit_distance", "asymmetric", "symmetric"}, default "merge_split"
        Type of constraint to use.
    dist_type : {"euclidean", "angular", "mixed"}, default "euclidean"
        Type of distance to use.
    free_ends : float or tuple of int, default (0, 1)
        A float corresponds to a percentage of sequence length for max exploration of `free_ends`,
        and in that case ``free_ends <= 0.4``.
        A tuple of 2 integers ``(k,l)`` that specifies relaxation bounds on
        the alignment of sequences endpoints: relaxed by ``k`` at the sequence beginning
        and relaxed by ``l`` at the sequence ending.
    free_ends_eps : float, default 1e-4
        ???.
    beamsize : int, default -1
        maximum amount of distortion allowed for signal warping.
    delins_cost : tuple of float, default (1., 1.)
        Deletion and insertion costs.
    max_stretch : bool, default 3
        ???.
    mixed_type : list(bool), optional
        A boolean vector, of size ``dim``, indicating whether the k^th component should be treated
        as an angle (``True``) or a regular scalar value (``False``).
    mixed_spread : list(float), optional
        A vector of positive scalars, of size ``dim``, used to normalize the distance values computed
        for each component with their typical spread.
    mixed_weight : list(float), optional
        A vector of positive weights, of size ``dim``.
        Does not necessarily sum to 1, but normalized if not.

    Other Parameters
    ----------------
    cum_dist_flag : bool, default True
        If ``True``, print the array of global distances.
    bp_flag : bool, default False
        If ``True``, print the back-pointers array.
    ld_flag : bool, default False
        If ``True``, print the local distance array.
    free_ends_flag : bool, default False
        If ``True``, print the sub-arrays of normalized distances on relaxed ending region and of optimal path lengths on relaxed ending region.
    optimal_path_flag : bool, default True
        If ``True``, print the optimal path.
    graphic_optimal_path_flag : bool, default True
        If ``True``, ?then?.
    graphic_seq_alignment : bool, default True
        If ``True``, generate a matplotlib figure with aligned sequences.
    verbose : bool, default True
        If ``True``, increase code verbosity.

    Notes
    -----
    For the `free_ends`, we must have:

      - ``k+l < min(nt,nr)``
      - ``k >=0`` and ``l>=1``

    """
    logger = kwargs.get('logger', None)
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

    dtwcomputer = DTW(seqX, seqY, constraints=constraint, delins_cost=delins_cost, ldist=ld,
                      mixed_type=mixed_type, mixed_spread=mixed_spread, mixed_weight=mixed_weight,
                      beam_size=beamsize, max_stretch=max_stretch)
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

    return dtwcomputer.print_results(**kwargs)
