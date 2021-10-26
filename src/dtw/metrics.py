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
Metrics used by DTW.
"""

import numpy as np

def euclidean_dist(v1, v2, **kwargs):
    """Compute the Euclidean distance between two inter-nodes lengths.

    Parameters
    ----------
    v1, v2 : float
        Inter-node lengths to compare.

    Returns
    -------
    float
        The Euclidean distance, normalized in ``[0, 1]``.

    """
    return np.linalg.norm(v1 - v2)


def angular_dist(a1, a2, **kwargs):
    """Distance between two angles as a percentage of the distance of their difference to 180 degrees.

    Parameters
    ----------
    a1, a2 : int or float
        Angles to compare, given in degrees.

    Returns
    -------
    float
        Angular distance, normalized in ``[0, 1]``.

    Notes
    -----
    The returned distance is a number between 0 and 1:

      - ``0`` means the two angles are equal up to 360 degrees.
      - ``1`` means that the two angles are separated by 180 degrees.
      - A distance of ``0.5`` corresponds to a difference of 90 degrees between the two angles.

    """
    Da = a1 - a2
    # da is the angle corresponding to the difference between
    # the original angles. 0 <= da < 360.
    da = Da % 360.
    # assert (da >= 0.)
    return 1 - np.abs(180 - da) / 180.


# to use this angular distance with numpy arrays
vangular_dist = np.vectorize(angular_dist)


def mixed_dist(v1, v2, **kwargs):
    """Distance where normal components are mixed with periodic ones (here angles).

    Parameters
    ----------
    v1, v2 : float
        Input vectors to compare (should be of same dimension ``dim``).

    Other Parameters
    ----------------
    is_angular : list(bool)
        A boolean vector, of size ``dim``, indicating whether the k^th component should be treated
        as an angle (``True``) or a regular scalar value (``False``).
    spread : list(float)
        A vector of positive scalars, of size ``dim``, used to normalize the distance values computed
        for each component with their typical spread.
    weight : list(float)
        A vector of positive weights, of size ``dim``.
        Does not necessarily sum to 1, but normalized if not.

    Returns
    -------
    float
        The distance value.

    Notes
    -----
    The resulting distance is:

    .. math::
        D(v_1,v_2) = \sqrt{\sum_{k} \text{weight} \[k\] * d{_k}^2(v_1[k],v_2[k])/ \text{spread}[k]^2)}

    where :math:`d\_k` is a distance that depends on ``type[k]``.

    """
    is_angular = kwargs.get('is_angular', None)
    spread = kwargs.get('spread', None)
    weight = kwargs.get('weight', None)

    # default values
    dim = len(v1)
    if is_angular is None:
        is_angular = np.full((dim,), False)  # by default type indicates only normal v1_coords
    if spread is None:
        spread = np.full((dim,), 1)  # spread will not modify the distance by default
    if weight is None:
        weight = np.full((dim,), 1)  # equal weights by default

    # if the array alpha is not normalized, it is is normalized first here
    weight = np.array(weight)
    sumweight = sum(weight)  # should be 1
    if not np.isclose(sumweight, 1.0):
        weight = weight / sum(weight)

    # Extract the sub-arrays corresponding to angles types and coord types resp.

    nottype = np.invert(is_angular)  # not type
    dim1 = np.count_nonzero(is_angular)  # nb of True values in type
    dim2 = np.count_nonzero(nottype)  # nb of False values in type

    v1_angles = np.extract(is_angular, v1)  # sub-array of angles only (dim1)
    v1_coords = np.extract(nottype, v1)  # sub-array of coordinates only (dim2)

    v2_angles = np.extract(is_angular, v2)  # idem for v2
    v2_coords = np.extract(nottype, v2)

    weight1 = np.extract(is_angular, weight)  # sub-array of weights for angles
    weight2 = np.extract(nottype, weight)  # sub-array of weights for coordinates

    spread1 = np.extract(is_angular, spread)  # sub-array of spread factors for angles
    spread2 = np.extract(nottype, spread)  # sub-array of spread factors for coordinates

    if not dim1 == 0:
        DD1 = vangular_dist(v1_angles, v2_angles) ** 2  # angle dist (squared)
        DD1 = DD1 / spread1 ** 2
        DD1 = DD1 * weight1  # adding weights
    else:
        DD1 = []

    # case of normal coordinates
    if not dim2 == 0:
        DD2 = (v1_coords - v2_coords) ** 2  # euclidean L2 norm (no sqrt yet)
        DD2 = DD2 / spread2 ** 2
        DD2 = DD2 * weight2  # adding weights
    else:
        DD2 = []

    DD = sum(DD1) + sum(DD2)
    return np.sqrt(DD)
