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
Generic dynamic time warping algorithms

Implementation of a generic DTW algorithm with symmetric asymmetric or classical edit
distance or split-merge constraints.

DTW techniques are based in particular on basic DTW algorithm described in :

Sakoe, H., & Chiba, S. (1978) Dynamic programming algorithm optimization for spoken
word recognition. IEEE Transactions on Acoustic, Speech, and Signal Processing,
ASSP-26(1),43-49.

and new dynamic time warping based techniques such as merge_split.

** Call:
# create a class instance of dtw to compute the distance between two sequences
dtwcomputer = dtw(seqX, seqY)
# run the algorithm with specific options
ndist, optpath, length, ndistarray, backpointers = dtwcomputer.run(l_dist = ld, free_ends = fe, beam_size = bs)
# print the results (a number of flags can be turned on/off to display various aspects of the results)
dtwcomputer.print_results(cumdistflag, bpflag, freeendsflag, optimalpathflag)

** args:
- seqX, seqY: two arrays (lists or np.arrays of scalars or of vectors with identical dimension)

** Optional args:
- Different algorithms are available and can be selected by the following flags:
symmetric, asymmetric, edit_distance, merge_split
- l_dist (func(val,val)-->positive float)= local distance used for pointwise comparison
- free_ends  ((int,int)): should be greater than (0,1). Elements are respectively
            relaxations at beginning and end of the sequences (identical for both sequences)
- beam_size  (int): maximum amount of distortion allowed for signal warping

Returned values:
- ndist: optimal computed distance between the two sequences
- optpath: optimal path recovered
- length: length of the optimal path
- ndistarray: numpy array of normalized distances
- backpointers: numpy array of back-pointers

tests: see the file example-dtw for this
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel
from joblib import delayed

FORMATTER = "%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMATTER, level=logging.DEBUG)


def euclidean_dist(v1, v2, **args):
    """Compute the Euclidean distance between `v1` and `v2`, two internodes lengths.

    Parameters
    ----------
    v1 : float
        first internode length value
    v2 : float
        second internode length value

    Returns
    -------
    float
        the Euclidean distance between the internodes lengths, normalized between [0, 1]

    """
    return np.linalg.norm(v1 - v2)


# Can only be called for scalar arguments a1 and a2
def angular_dist(a1, a2, **args):
    """Distance between two angles as a percentage of the distance of their difference to 180 degrees.

    Parameters
    ----------
    a1 : int, float
        The first angle, given in degrees.
    a2 : int, float
        The second angle, given in degrees.

    Returns
    -------
    float
        Angular distance, normalized in ``[0, 1]``.

    Notes
    -----
    The returned distance is a number between 0 and 1:

      - 0 means the two angles are equal up to 360 degrees.
      - 1 means that the two angles are separated by 180 degrees.
      - A distance of 0.5 corresponds to a difference of 90 degrees between the two angles.

    """
    Da = a1 - a2
    # da is the angle corresponding to the difference between
    # the original angles. 0 <= da < 360.
    da = Da % 360.
    # assert (da >= 0.)

    return 1 - np.abs(180 - da) / 180.


# to use this angular distance with numpy arrays
vangular_dist = np.vectorize(angular_dist)


# Can only be called for scalar arguments a1 and a2
def mixed_dist(v1, v2, is_angular=[], spread=[], weight=[]):
    """Distance where normal components are mixed with periodic ones (here angles)

    Parameters
    ----------
    v1 : float
        First input vectors to compare (should be of same dimension ``dim``).
    v2 : float
        Second input vectors to compare (should be of same dimension ``dim``).
    is_angular : list(bool), optional
        A boolean vector, of size ``dim``, indicating whether the k^th component should be treated
        as an angle (``True``) or a regular scalar value (``False``).
    spread : list(float), optional
        A vector of positive scalars, of size ``dim``, used to normalize the distance values computed
        for each component with their typical spread.
    weight : list(float), optional
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

    # default values
    dim = len(v1)
    if is_angular == []:
        is_angular = np.full((dim,), False)  # by default type indicates only normal v1_coords
    if spread == []:
        spread = np.full((dim,), 1)  # spread will not modify the distance by default
    if weight == []:
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
    v1_coords = np.extract(nottype, v1)  # sub-array of coords only (dim2)

    v2_angles = np.extract(is_angular, v2)  # idem for v2
    v2_coords = np.extract(nottype, v2)

    weight1 = np.extract(is_angular, weight)  # sub-array of weights for angles
    weight2 = np.extract(nottype, weight)  # sub-array of weights for coords

    spread1 = np.extract(is_angular, spread)  # sub-array of spread factors for angles
    spread2 = np.extract(nottype, spread)  # sub-array of spread factors for coords

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


###############################################################################
# Tools
###############################################################################
def print_matrix_bp(a):
    """Print matrix of back-pointers.

    Parameters
    ----------
    a :
        ???

    """
    print("Matrix[" + ("%d" % a.shape[0]) + "][" + ("%d" % a.shape[1]) + "]")
    rows = a.shape[0]
    cols = a.shape[1]
    for i in range(0, rows):
        for j in range(0, cols):
            print(("(%2d,%2d)" % (a[i, j][0], a[i, j][1])), end=' ')  # "%6.f" %a[i,j],
        print()
    print()


# Main DTW class: to build DTW computer objects on a pair of sequences
class DTW(object):
    """Dynamic Time Warping.

    Attributes
    ----------
    seqX : numpy.ndarray
        First Nxd array of *elements* to compare.
    seqY : numpy.ndarray
        Second array of *elements* to compare.
        constraint_type : {"edit_distance", "asymmetric, "symmetric", "merge_split"}, optional
    nX : int
        Number of elements in first sequence.
    nY : int
        Number of elements in second sequence.
    bp : numpy.ndarray
        Array of back-pointers.
    delins_cost : tuple of float, default (1., 1.)
        Deletion and insertion costs.
    cum_dist :

    editop :

    cum_dist_boundaryX :

    cum_dist_boundaryY :

    l_dist : function, default euclidean_dist
        The function to compute the local distance used to compare values of both sequences.
        Typically `euclidean_dist()`,  `angular_dist()` or `mixed_dist()`.
    free_ends : (int, int), optional
        A tuple of 2 integers ``(k,l)`` that specifies relaxation bounds on
        the alignment of sequences endpoints: relaxed by ``k`` at the sequence beginning
        and relaxed by ``l`` at the sequence ending.
    beam_size : int, optional
        maximum amount of distortion allowed for signal warping.
    constraints : {"edit_distance", "asymmetric, "symmetric", "merge_split"}, optional
        Type of constraint to use to edit local distance.
    mixed_type : list(bool), optional
        A boolean vector, of size ``dim``, indicating whether the k^th component should be treated
        as an angle (``True``) or a regular scalar value (``False``).
    mixed_spread : list(float), optional
        A vector of positive scalars, of size ``dim``, used to normalize the distance values computed
        for each component with their typical spread.
    mixed_weight : list(float), optional
        A vector of positive weights, of size ``dim``.
        Does not necessarily sum to 1, but normalized if not.
    optpath_array :

    optpathlength_array :

    optpath_normalized_cumdist_array :

    min_normalized_cost :

    non_mormalized_optcost :

    opt_index :

    opt_backtrackpath :


    """

    def __init__(self, seqX, seqY, constraints="symmetric", ldist=euclidean_dist,
                 mixed_type=[], mixed_spread=[], mixed_weight=[], free_ends=(0, 1),
                 beam_size=-1, max_stretch=3, delins_cost=(1.0, 1.0), **kwargs):
        """Constructor.

        Parameters
        ----------
        seqX : list or numpy.ndarray
            First ``N_x x n_dim`` array of *elements* to compare.
        seqY : list or numpy.ndarray
            Second ``N_y x n_dim`` array of *elements* to compare, act as reference.
        constraints : {"edit_distance", "asymmetric, "symmetric", "merge_split"}, default "merge_split"
            Type of constraint to use.
        delins_cost : tuple of float, default (1., 1.)
            Deletion and insertion costs.
        free_ends : (int, int), optional
            A tuple of 2 integers ``(k,l)`` that specifies relaxation bounds on
            the alignment of sequences endpoints: relaxed by ``k`` at the sequence beginning
            and relaxed by ``l`` at the sequence ending.
        ldist : function, default euclidean_dist
            The function to compute the local distance used to compare values of both sequences.
            Typically `euclidean_dist()`,  `angular_dist()` or `mixed_dist()`.
        mixed_type : list of bool, optional
            A boolean vector, of size ``n_dim``, indicating whether the k^th component should be treated
            as an angle (``True``) or a regular scalar value (``False``).
        mixed_spread : list of float, optional
            A vector of positive scalars, of size ``dim``, used to normalize the distance values computed
            for each component with their typical spread.
        mixed_weight : list of float, optional
            A vector of positive weights, of size ``n_dim``.
            Does not necessarily sum to 1, but normalized if not.
        beam_size : int, default -1
            maximum amount of distortion allowed for signal warping.
        max_stretch : bool, default 3
            maximum amount of stretching allowed for signal warping.

        Other Parameters
        ----------------
        names : list
            Names of the sequences, _i.e._ what they represent.
            This is used for summaries and graphical representations.

        Notes
        -----
        An *element* can be a scalar (``d=1``) or a vector.

        Example
        -------
        >>> from dtw.dtw import DTW
        >>> test_seq = [2, 3, 4, 3, 3, 4, 0, 3, 3, 2, 1, 1, 1, 3, 3, 4, 4]
        >>> ref_seq = [0, 0, 4, 3, 3, 3, 3, 3, 2, 1, 2, 1, 3, 4]
        >>> dtwcomputer = DTW(test_seq, ref_seq)
        >>> ndist, path, length, ndistarray, backpointers = dtwcomputer.run()
        >>> dtwcomputer.get_results()
        >>> dtwcomputer.get_better_results()

        """
        # Initialize empty attributes
        self.n_dim = None
        self.l_dist = None
        self.cum_dist = None
        self.cum_dist_boundaryX = None
        self.cum_dist_boundaryY = None
        self.optpath_array = None
        self.optpathlength_array = None
        self.optpath_normalized_cumdist_array = None
        self.min_normalized_cost = None
        self.non_mormalized_optcost = None
        self.opt_index = None
        self.opt_backtrack_path = None

        # seqX and seqY are expected to be two np.arrays of elements of identical dim
        # an element can be a scalar or a vector
        self.seqX = np.array(seqX)
        self.seqY = np.array(seqY)
        self.n_dim = self._check_sequences()  # verify sequences requirements
        self.nX = len(seqX)
        self.nY = len(seqY)
        self.delins_cost = delins_cost
        self._free_ends = None
        self.free_ends = free_ends
        self.max_stretch = max_stretch
        self.beam_size = beam_size
        self.constraints = constraints
        self.ldist_f = ldist

        # for mixed_mode
        self.mixed_type = mixed_type
        self.mixed_spread = mixed_spread
        self.mixed_weight = mixed_weight

        # Defines sequence names and use
        self.names = [f'sequence_{i}' for i in range(self.n_dim)]
        names = kwargs.get('names', None)
        if names is not None:
            if isinstance(names, str):
                names = [names]
            if len(names) == self.n_dim:
                self.names = names
            else:
                logging.warning("Not the same number of names and sequence dimensions, using default names!")
        else:
            logging.debug("No name given to sequence dimensions, using default names.")

    def _check_sequences(self):
        try:
            _, rn_dim = self.seqX.shape
        except ValueError:
            rn_dim = 1
        try:
            _, tn_dim = self.seqY.shape
        except ValueError:
            tn_dim = 1
        # Check we have the same number of dimensions in both sequences:
        try:
            assert tn_dim == rn_dim
        except AssertionError:
            raise ValueError("Not the same number of dimensions in the two sequences!")
        return rn_dim

    def initdtw(self):
        # initiates the arrays of backpointers, localdist and cum_dist
        assert (len(self.free_ends) == 2)
        a = self.free_ends[0]
        b = self.free_ends[1]
        assert (a + b < self.nX and a + b < self.nY)

        # initialization of backpointer array
        self.bp = np.empty((self.nX, self.nY), dtype=object)
        # initialization of cummulated distance array
        self.cum_dist = np.full((self.nX, self.nY), np.Infinity)

        # edit op
        self.editop = np.full((self.nX, self.nY), "-")

        # border array for boundary conditions on cum_dist array
        self.cum_dist_boundaryX = np.full(self.nX, np.Infinity)
        if a != 0: self.cum_dist_boundaryX[:a] = 0.0
        self.cum_dist_boundaryY = np.full(self.nY, np.Infinity)
        if a != 0: self.cum_dist_boundaryY[:a] = 0.0

        # initialization and computation of the matrix of local distances
        self.l_dist = np.full((self.nX, self.nY), np.Infinity)

        for i in range(self.nX):
            for j in range(self.nY):
                self.l_dist[i, j] = euclidean_dist(self.seqX[i], self.seqY[j])

    def find_path(self, path, editoparray, verbose=True):
        # print "Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]"
        l = len(path)
        prev_dist = 0.
        localcost = []
        for i in range(l):
            a = path[i][0]
            b = path[i][1]
            # print "[",a,",",b,"] ", editoparray[a,b]
            if verbose:
                print("%2d : [%2d,%2d]" % (i, a, b), end=' ')
                print(editoparray[a, b], end=' ')
            lc = self.cum_dist[a, b] - prev_dist
            if verbose:
                print("  cost = %6.3f" % lc)
            localcost.append(lc)
            prev_dist = self.cum_dist[a, b]

        data = {'test': path[:, 0],
                'reference': path[:, 1],
                'type': [editoparray[path[i][0], path[i][1]] for i in range(l)],
                'cost': localcost}

        return data

    @property
    def free_ends(self):
        """Get the left and right free-end values.

        Returns
        -------
        (int, int)
            a tuple with the left and right free-end values

        Example
        -------
        >>> from dtw.dtw import DTW
        >>> test_seq = [2, 3, 4, 3, 3, 4, 0, 3, 3, 2, 1, 1, 1, 3, 3, 4, 4]
        >>> ref_seq = [0, 0, 4, 3, 3, 3, 3, 3, 2, 1, 2, 1, 3, 4]
        >>> dtwcomputer = DTW(test_seq, ref_seq)
        >>> dtwcomputer.free_ends  # Get the default free-end values
        (0, 1)

        """
        return self._free_ends

    @free_ends.setter
    def free_ends(self, values):
        """Set the left and right free-end values.

        Parameters
        ----------
        values : (int, int)
            a length-2 tuple of left and right free-end values.

        Notes
        -----
        This reset the cumulative distance boundaries (``self.cumdistboundaryX`` & ``self.cumdistboundaryY``) as they depend on the left free-end.

        Example
        -------
        >>> from dtw.dtw import DTW
        >>> test_seq = [2, 3, 4, 3, 3, 4, 0, 3, 3, 2, 1, 1, 1, 3, 3, 4, 4]
        >>> ref_seq = [0, 0, 4, 3, 3, 3, 3, 3, 2, 1, 2, 1, 3, 4]
        >>> dtwcomputer = DTW(test_seq, ref_seq)
        >>> dtwcomputer.free_ends  # Get the default free-end values
        (0, 1)
        >>> dtwcomputer.free_ends = (3, 4)  # Set new free-end values
        >>> dtwcomputer.free_ends  # Get the new free-end values
        (3, 4)

        """
        try:
            assert (len(values) == 2)
        except AssertionError:
            raise ValueError(f"Parameters `free_ends` should be a length-2 iterable, got: {self.free_ends}")
        a = values[0]
        b = values[1]
        try:
            assert (a + b < self.nX and a + b < self.nY)
        except AssertionError:
            raise ValueError(f"The sum of both `free_ends` values should be inferior to the length of each sequence!")

        self._free_ends = tuple(values)
        self.initdtw()  # depends on left free-end

        return

    def get_results(self, verbose=True):
        """Return a dictionary with aligned sequences, event types and associated local costs.

        Parameters
        ----------
        verbose : bool, optional
            Also print a visual representation of this dictionary, default is ``False``.

        Returns
        -------
        dict
            the result dictionary with aligned sequences, event types and associated local costs.

        Note
        ----
        You have to call the ``run()`` method first!

        Details of 'type' list symbols for 'merge_split' constraints:
          - ``=``: match
          - ``~``: quasi-match
          - ``m``: merge
          - ``s``: split

        Details of 'type' list symbols for other methods:
          - ``m``: matching
          - ``s``: substitution
          - ``d``: deletion
          - ``i``: insertion

        Example
        -------
        >>> import numpy as np
        >>> from dtw.dtw import DTW
        >>> from dtw.dtw import mixed_dist
        >>> seq_test = np.array([[96, 163, 137, 113, 24, 170, 152, 137, 255, 148, 111, 16, 334, 160, 94, 116, 144, 132, 145], [50, 60, 48, 19, 31, 0, 37, 20, 31, 25, 7, 1, 51, 29, 26, 16, 22, 12, 23]]).T
        >>> seq_ref = np.array([[96, 163, 137, 137, 170, 152, 137, 132, 123, 148, 127, 191, 143, 160, 94, 116, 144, 132, 145], [50, 60, 48, 50, 0, 37, 20, 0, 31, 25, 8, 27, 24, 29, 26, 16, 22, 12, 23 ]]).T
        >>> max_ref = np.max(seq_ref[:, 1])
        >>> max_test = np.max(seq_test[:, 1])
        >>> dtwcomputer = DTW(seq_test, seq_ref, constraints='merge_split', ldist=mixed_dist, mixed_type=[True, False], mixed_spread=[1, max(max_ref, max_test)], mixed_weight=[1, 1], names=["angles", "inter-nodes"])
        >>> dtwcomputer.run()
        >>> dtwcomputer.get_results()

        """
        return self.find_path(self.opt_backtrack_path, self.editop, verbose)

    def get_better_results(self, start_index=1, verbose=True):
        """

        Parameters
        ----------
        start_index : {0, 1}, optional
            Use this to have the first element indexed at 0 or 1.

        Returns
        -------
        dict
            the result dictionary with aligned sequences, event types and associated local costs.

        Example
        -------
        >>> import numpy as np
        >>> from dtw.dtw import DTW
        >>> from dtw.dtw import mixed_dist

        >>> # Example #1 - Alignment of angles and inter-nodes sequences without free-ends:
        >>> seq_test = np.array([[96, 163, 137, 113, 24, 170, 152, 137, 255, 148, 111, 16, 334, 160, 94, 116, 144, 132, 145], [50, 60, 48, 19, 31, 0, 37, 20, 31, 25, 7, 1, 51, 29, 26, 16, 22, 12, 23]]).T
        >>> seq_ref = np.array([[96, 163, 137, 137, 170, 152, 137, 132, 123, 148, 127, 191, 143, 160, 94, 116, 144, 132, 145], [50, 60, 48, 50, 0, 37, 20, 0, 31, 25, 8, 27, 24, 29, 26, 16, 22, 12, 23 ]]).T
        >>> max_ref = np.max(seq_ref[:, 1])
        >>> max_test = np.max(seq_test[:, 1])
        >>> dtwcomputer = DTW(seq_test, seq_ref, constraints='merge_split', ldist=mixed_dist, mixed_type=[True, False], mixed_spread=[1, max(max_ref, max_test)], mixed_weight=[0.5, 0.5], names=["angles", "inter-nodes"])
        >>> dtwcomputer.run()
        >>> dtwcomputer.get_results()
        >>> dtwcomputer.get_better_results()

        >>> # Example #2 - Alignment of angles and inter-nodes sequences with right free-ends:
        >>> seq_test = np.array([[123, 169, 224, 103, 131, 143, 113, 163, 148, 11, 153, 164, 118, 139, 135, 125, 147, 174, 121, 91, 127, 124], [70, 1, 32, 15, 56, 42, 39, 46, 4, 29, 29, 10, 12, 30, 0, 14, 12, 15, 0, 0, 12, 0]]).T
        >>> seq_ref = np.array([[123, 136, 131, 143, 113, 163, 159, 153, 164, 118, 139, 135, 125, 147, 174, 121, 91, 127, 124, 152, 124, 107, 126], [70, 48, 56, 42, 39, 46, 33, 29, 10, 12, 30, 0, 14, 12, 15, 0, 0, 12, 0, 13, 16, 0, 1]]).T
        >>> max_ref = np.max(seq_ref[:, 1])
        >>> max_test = np.max(seq_test[:, 1])
        >>> dtwcomputer = DTW(seq_test, seq_ref, constraints='merge_split', ldist=mixed_dist, mixed_type=[True, False], mixed_spread=[1, max(max_ref, max_test)], mixed_weight=[0.5, 0.5], names=["angles", "inter-nodes"])
        >>> dtwcomputer.free_ends = (0, 5)
        >>> dtwcomputer.run()
        >>> dtwcomputer.get_results()
        >>> dtwcomputer.get_better_results()
        >>> dtwcomputer.plot_results()

        """
        idx_test, idx_ref, event_types, event_costs = self.get_results(False).values()
        event_types = np.array(event_types)
        event_costs = np.array(event_costs)
        merge_indexes = np.where(event_types == 'm')[0]
        split_indexes = np.where(event_types == 's')[0]
        logging.info(f"Found {len(merge_indexes)} merge & {len(split_indexes)} split events!")
        logging.debug(f"List of merge indexes: {merge_indexes}")
        logging.debug(f"List of split indexes: {split_indexes}")

        logging.debug(f"Updating MERGE events...")
        idx_test, idx_ref = list(idx_test), list(idx_ref)
        for i, merge_idx in enumerate(merge_indexes):
            logging.debug(f"Test sequence indexes: {idx_test}")
            logging.debug(f"Reference sequence indexes: {idx_ref}")
            # Find how much rows are missing in the 'test sequence' with this merge:
            if merge_idx == 0:
                added_test = idx_test[merge_idx] - 1
            else:
                added_test = idx_test[merge_idx] - idx_test[merge_idx - 1] - 1
            logging.debug(f"Missing {added_test} rows in the test sequence for merge index {merge_idx}")
            # Duplicate missing rows in both sequences
            idx_test = duplicate_idx(idx_test, merge_idx, added_test)
            idx_ref = duplicate_idx(idx_ref, merge_idx, added_test)
            event_types = duplicate_idx(event_types, merge_idx, added_test)
            event_costs = duplicate_idx(event_costs, merge_idx, added_test)
            logging.debug(f"Post-duplication test sequence indexes: {idx_test}")
            logging.debug(f"Post-duplication reference sequence indexes: {idx_ref}")
            # Relabel duplicated test:
            idx_test = np.array(idx_test)
            idx_test[np.where(idx_test == idx_test[merge_idx])[0]] = list(range(idx_test[merge_idx] - added_test, idx_test[merge_idx] + 1))
            idx_test = list(idx_test)
            logging.debug(f"Updated test sequence indexes: {idx_test}")
            logging.debug(f"Updated reference sequence indexes: {idx_ref}")
            # Rewrite intervals
            # Shift `merge_indexes`:
            merge_indexes[i + 1:] += added_test
            logging.debug(f"Updated merge index: {merge_indexes}")

        logging.debug(f"Updating SPLIT events...")
        split_indexes = np.where(np.array(event_types) == 's')[0]
        logging.debug(f"New split indexes after updating merge events: {split_indexes}")
        idx_test, idx_ref = list(idx_test), list(idx_ref)
        for i, split_idx in enumerate(split_indexes):
            logging.debug(f"Test sequence indexes {idx_test}")
            logging.debug(f"Reference sequence indexes {idx_ref}")
            # Find how much rows are missing in the 'reference sequence' with this split:
            if split_idx == 0:
                miss_ref = idx_ref[split_idx] - 1
            else:
                miss_ref = idx_ref[split_idx] - idx_ref[split_idx - 1] - 1
            logging.debug(f"Missing {miss_ref} rows in the reference sequence for merge index {split_idx}")
            # Duplicate missing rows in both sequences
            idx_test = duplicate_idx(idx_test, split_idx, miss_ref)
            idx_ref = duplicate_idx(idx_ref, split_idx, miss_ref)
            event_types = duplicate_idx(event_types, split_idx, miss_ref)
            event_costs = duplicate_idx(event_costs, split_idx, miss_ref)
            logging.debug(f"Post-duplication test sequence indexes {idx_test}")
            logging.debug(f"Post-duplication reference sequence indexes {idx_ref}")
            # Relabel duplicated reference:
            idx_ref = np.array(idx_ref)
            logging.debug(idx_ref[np.where(idx_ref == idx_ref[split_idx])[0]])
            logging.debug((idx_ref[split_idx] - miss_ref, idx_ref[split_idx] + 1))

            idx_ref[np.where(idx_ref == idx_ref[split_idx])[0]] = list(range(idx_ref[split_idx] - miss_ref, idx_ref[split_idx] + 1))
            idx_ref = list(idx_ref)
            logging.debug(f"Updated test sequence indexes {idx_test}")
            logging.debug(f"Updated reference sequence indexes {idx_ref}")
            # Rewrite intervals
            # Shift `split_indexes`:
            split_indexes[i + 1:] += miss_ref
            logging.debug(f"Updated split index {split_indexes}")

        # Change ids to match start index:
        idx_test = np.array(idx_test) + start_index
        idx_ref = np.array(idx_ref) + start_index
        return {'test': idx_test, 'reference': idx_ref, 'type': np.array(event_types), 'cost': np.array(event_costs)}

    def plot_results(self):
        """

        Examples
        --------
        >>> import numpy as np
        >>> from dtw.dtw import DTW
        >>> from dtw.dtw import mixed_dist
        >>> seq_test = np.array([[96, 163, 137, 113, 24, 170, 152, 137, 255, 148, 111, 16, 334, 160, 94, 116, 144, 132, 145], [50, 60, 48, 19, 31, 0, 37, 20, 31, 25, 7, 1, 51, 29, 26, 16, 22, 12, 23]]).T
        >>> seq_ref = np.array([[96, 163, 137, 137, 170, 152, 137, 132, 123, 148, 127, 191, 143, 160, 94, 116, 144, 132, 145], [50, 60, 48, 50, 0, 37, 20, 0, 31, 25, 8, 27, 24, 29, 26, 16, 22, 12, 23 ]]).T
        >>> max_ref = np.max(seq_ref[:, 1])
        >>> max_test = np.max(seq_test[:, 1])
        >>> dtwcomputer = DTW(seq_test, seq_ref, constraints='merge_split', ldist=mixed_dist, mixed_type=[True, False], mixed_spread=[1, max(max_ref, max_test)], mixed_weight=[0.5, 0.5], names=["angles", "inter-nodes"])
        >>> dtwcomputer.run()
        >>> dtwcomputer.plot_results()

        """
        ref_indexes = np.array(range(self.nY))
        results = self.get_better_results(start_index=0)
        seq_test = results['test']
        seq_ref = results['reference']
        pred_types = results['type']
        pred_types[np.where(pred_types == '=')] = ''

        fig, axs = plt.subplots(ncols=1, nrows=self.n_dim, constrained_layout=True)
        if self.n_dim == 1:
            ref_val = self.seqY
            test_val = [self.seqX[e] for e in seq_test]
            self._plot_results(axs, ref_indexes + 1, ref_val, seq_ref + 1, test_val, pred_types, self.names[0])
        else:
            for i in range(self.n_dim):
                ref_val = self.seqY[:, i]
                test_val = [self.seqX[e, i] for e in seq_test]
                self._plot_results(axs[i], ref_indexes + 1, ref_val, seq_ref + 1, test_val, pred_types, self.names[i])
        plt.suptitle(f"DTW - {self.constraints.replace('_', ' ')} alignment")
        plt.show()

    @staticmethod
    def _plot_results(ax, ref_x, ref_y, test_x, test_y, pred_types, name):
        """Display the result of the alignment.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib ``Axe`` object to populate.
        ref_x : list
            The index of the reference sequence.
        ref_y : list
            The values of the reference sequence dimension.
        test_x : list
            The index of the test sequence.
        test_y : list
            The values of the test sequence dimension.
        pred_types : list
            The type of event predicted by DTW.
        name : str
            The name of the sequence dimension (values).

        """
        ax.plot(ref_x, ref_y, marker='+', linestyle='dashed', label='Reference')
        ax.plot(test_x, test_y, marker='o', linestyle='dashed', label='Test')
        for x, y, t in zip(test_x, test_y, pred_types):
            ax.text(x, y, t.upper(), ha='center', va='center', fontfamily='monospace', fontsize='large', fontweight='medium')
        x_ticks = np.arange(0, max(ref_x) + 1, 1)
        ax.set_xticks(x_ticks)
        ax.set_title(f"{name.upper()}")
        ax.set_xlabel('index of reference sequence')
        ax.set_ylabel(name)
        ax.grid(True, which='major', axis='both', linestyle='dotted')
        ax.legend()

    def get_split_events(self):
        _, idx_ref, event_types, _ = self.get_results(False).values()
        event_types = np.array(event_types)
        split_indexes = np.where(event_types == 's')[0]
        miss_ref = []
        for i, split_idx in enumerate(split_indexes):
            # Find how much rows are missing in the 'reference sequence' with this split:
            if split_idx == 0:
                miss_ref.append(idx_ref[split_idx] - 1)
            else:
                miss_ref.append(idx_ref[split_idx] - idx_ref[split_idx - 1] - 1)

        return miss_ref

    def get_merge_events(self):
        idx_test, _, event_types, _ = self.get_results(False).values()
        event_types = np.array(event_types)
        merge_indexes = np.where(event_types == 'm')[0]
        added_test = []
        for i, merge_idx in enumerate(merge_indexes):
            # Find how much rows are missing in the 'test sequence' with this merge:
            if merge_idx == 0:
                added_test.append(idx_test[merge_idx] - 1)
            else:
                added_test.append(idx_test[merge_idx] - idx_test[merge_idx - 1] - 1)

        return added_test

    def get_aligned_reference_sequence(self):
        """

        Returns
        -------

        Example
        -------
        >>> import numpy as np
        >>> from dtw.dtw import DTW
        >>> from dtw.dtw import mixed_dist
        >>> # Alignment of angles and inter-nodes sequences with left and right free-ends:
        >>> seq_test = np.array([[166, 348, 150, 140, 294, 204, 168, 125, 125, 145, 173, 123, 127, 279, 102, 144, 136, 146, 137, 175, 103], [42, 31, 70, 55, 0, 0, 42, 27, 31, 33, 21, 23, 1, 56, 26, 18, 17, 16, 3, 0, 8]]).T
        >>> seq_ref = np.array([[150, 140, 138, 168, 125, 125, 145, 173, 123, 127, 99, 180, 102, 144, 136, 146, 137, 142, 136, 134], [70, 55, 0, 42, 27, 31, 33, 21, 23, 1, 28, 28, 26, 18, 17, 16, 3, 0, 8, 18]]).T
        >>> max_ref = np.max(seq_ref[:, 1])
        >>> max_test = np.max(seq_test[:, 1])
        >>> dtwcomputer = DTW(seq_test, seq_ref, constraints='merge_split', ldist=mixed_dist, mixed_type=[True, False], mixed_spread=[1, max(max_ref, max_test)], mixed_weight=[0.5, 0.5])
        >>> dtwcomputer.free_ends = (2, 4)
        >>> dtwcomputer.run()
        >>> dtwcomputer.get_aligned_reference_sequence()

        """
        aligned_results = self.get_better_results(0)
        return np.array([self.seqY[e] for e in aligned_results['reference']])

    def get_aligned_test_sequence(self):
        """

        Returns
        -------

        Example
        -------
        >>> import numpy as np
        >>> from dtw.dtw import DTW
        >>> from dtw.dtw import mixed_dist
        >>> # Alignment of angles and inter-nodes sequences with left and right free-ends:
        >>> seq_test = np.array([[166, 348, 150, 140, 294, 204, 168, 125, 125, 145, 173, 123, 127, 279, 102, 144, 136, 146, 137, 175, 103], [42, 31, 70, 55, 0, 0, 42, 27, 31, 33, 21, 23, 1, 56, 26, 18, 17, 16, 3, 0, 8]]).T
        >>> seq_ref = np.array([[150, 140, 138, 168, 125, 125, 145, 173, 123, 127, 99, 180, 102, 144, 136, 146, 137, 142, 136, 134], [70, 55, 0, 42, 27, 31, 33, 21, 23, 1, 28, 28, 26, 18, 17, 16, 3, 0, 8, 18]]).T
        >>> max_ref = np.max(seq_ref[:, 1])
        >>> max_test = np.max(seq_test[:, 1])
        >>> dtwcomputer = DTW(seq_test, seq_ref, constraints='merge_split', ldist=mixed_dist, mixed_type=[True, False], mixed_spread=[1, max(max_ref, max_test)], mixed_weight=[0.5, 0.5])
        >>> dtwcomputer.free_ends = (2, 4)
        >>> dtwcomputer.run()
        >>> dtwcomputer.get_aligned_test_sequence()

        """
        aligned_results = self.get_better_results(0)
        return np.array([self.seqX[e] for e in aligned_results['test']])

    def get_matching_sequences(self):
        """

        Returns
        -------

        Example
        -------
        >>> import numpy as np
        >>> from dtw.dtw import DTW
        >>> from dtw.dtw import mixed_dist
        >>> # Alignment of angles and inter-nodes sequences with left and right free-ends:
        >>> seq_test = np.array([[166, 348, 150, 140, 294, 204, 168, 125, 125, 145, 173, 123, 127, 279, 102, 144, 136, 146, 137, 175, 103], [42, 31, 70, 55, 0, 0, 42, 27, 31, 33, 21, 23, 1, 56, 26, 18, 17, 16, 3, 0, 8]]).T
        >>> seq_ref = np.array([[150, 140, 138, 168, 125, 125, 145, 173, 123, 127, 99, 180, 102, 144, 136, 146, 137, 142, 136, 134], [70, 55, 0, 42, 27, 31, 33, 21, 23, 1, 28, 28, 26, 18, 17, 16, 3, 0, 8, 18]]).T
        >>> max_ref = np.max(seq_ref[:, 1])
        >>> max_test = np.max(seq_test[:, 1])
        >>> dtwcomputer = DTW(seq_test, seq_ref, constraints='merge_split', ldist=mixed_dist, mixed_type=[True, False], mixed_spread=[1, max(max_ref, max_test)], mixed_weight=[0.5, 0.5])
        >>> dtwcomputer.free_ends = (2, 4)
        >>> dtwcomputer.run()
        >>> matched_ref, matched_test = dtwcomputer.get_matching_sequences()

        """
        aligned_results = self.get_better_results(0)
        ars = np.array([self.seqY[e] for e in aligned_results['reference']])
        ats = np.array([self.seqX[e] for e in aligned_results['test']])
        matching_indexes = [i for i, t in enumerate(aligned_results['type']) if t == '=' or t == '~']
        return ars[matching_indexes], ats[matching_indexes]

    def summarize(self):
        """Summarize alignment information.

        Example
        -------
        >>> import numpy as np
        >>> from dtw.dtw import DTW
        >>> from dtw.dtw import mixed_dist
        >>> # Alignment of angles and inter-nodes sequences with left and right free-ends:
        >>> seq_test = np.array([[166, 348, 150, 140, 294, 204, 168, 125, 125, 145, 173, 123, 127, 279, 102, 144, 136, 146, 137, 175, 103], [42, 31, 70, 55, 0, 0, 42, 27, 31, 33, 21, 23, 1, 56, 26, 18, 17, 16, 3, 0, 8]]).T
        >>> seq_ref = np.array([[150, 140, 138, 168, 125, 125, 145, 173, 123, 127, 99, 180, 102, 144, 136, 146, 137, 142, 136, 134], [70, 55, 0, 42, 27, 31, 33, 21, 23, 1, 28, 28, 26, 18, 17, 16, 3, 0, 8, 18]]).T
        >>> max_ref = np.max(seq_ref[:, 1])
        >>> max_test = np.max(seq_test[:, 1])
        >>> dtwcomputer = DTW(seq_test, seq_ref, constraints='merge_split', ldist=mixed_dist, mixed_type=[True, False], mixed_spread=[1, max(max_ref, max_test)], mixed_weight=[0.5, 0.5], names=["angles", "inter-nodes"])
        >>> dtwcomputer.free_ends = (2, 4)
        >>> dtwcomputer.run()
        >>> dtwcomputer.summarize()

         """
        results = self.get_results()
        aligned_results = self.get_better_results(0)
        merge_indexes = np.where(results['type'] == 'm')[0]
        split_indexes = np.where(results['type'] == 's')[0]
        chop_start = min(aligned_results['reference']) - 1 if min(aligned_results['reference']) > 1 else 0
        chop_end = self.nY - max(aligned_results['reference']) if max(aligned_results['reference']) > self.nY else 0
        tail_start = min(aligned_results['test']) - 1 if min(aligned_results['test']) > 1 else 0
        tail_end = self.nX - max(aligned_results['test']) if max(aligned_results['test']) > self.nX else 0

        summary = {}
        summary["reference sequence length"] = self.nY
        summary["test sequence length"] = self.nX
        summary["number chop start"] = chop_start
        summary["number chop end"] = chop_end
        summary["number tail start"] = tail_start
        summary["number tail end"] = tail_end
        summary["number of split events"] = len(split_indexes)
        summary["number of merge events"] = len(merge_indexes)
        summary["missed events per split"] = self.get_split_events()
        summary["added events per merge"] = self.get_merge_events()
        summary["total number of missed events"] = sum(self.get_split_events())
        summary["total number of added events"] = sum(self.get_merge_events())
        summary["total event errors"] = summary["number chop start"] + summary["number chop end"] + summary["total number of missed events"] \
                                        + summary["number tail start"] + summary["number tail end"] + summary["total number of added events"]
        summary["total matched events"] = len(np.where(aligned_results['type'] == '~')[0]) + len(np.where(aligned_results['type'] == '=')[0])
        matched_ref, matched_test = self.get_matching_sequences()
        for d in range(self.n_dim):
            diff = matched_ref[:, d] - matched_test[:, d]
            summary[f"{self.names[d]} mean difference"] = np.mean(diff)
            summary[f"{self.names[d]} standard deviation"] = np.std(diff)

        return summary

    def print_alignment(self, path, editoparray):
        """Print ???.

        Parameters
        ----------
        path :

        editoparray :


        """
        # print "Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]"
        print("test seq: ", end=' ')
        l = len(path)
        pi = pj = -1
        for k in range(l):
            i = path[k][0]
            j = path[k][1]
            # print ("[",a,",",b,"] ", editoparray[a,b])
            labl = editoparray[i, j]
            if labl == 'm':
                if i != pi + 1:
                    for h in range(pi + 1, i):
                        if self.seqX.ndim == 1:
                            print("%3d" % self.seqX[h], end=' ')
                        else:
                            print(self.seqX[h], end=' ')
                if self.seqX.ndim == 1:
                    print("%3d" % self.seqX[i], end=' ')
                else:
                    print(self.seqX[i], end=' ')
            elif labl == 's':
                if j != pj + 1:
                    for h in range(pj + 1, j):
                        print(" - ", end=' ')
                if self.seqX.ndim == 1:
                    print("%3d" % self.seqX[i], end=' ')
                else:
                    print(self.seqX[i], end=' ')
            elif labl == "m" or labl == "s" or labl == "i" or labl == "~" or labl == "=":
                if len(np.shape(self.seqX)) == 1:  # for scalar values
                    print("%3d" % self.seqX[i], end=' ')
                else:
                    print(self.seqX[i], end=' ')  # for vectorial values
            elif labl == "d" or labl == "s":
                print(" - ", end='s')
            pi = i
            pj = j
        print()
        print("ref seq : ", end=' ')
        pi = pj = -1
        for k in range(l):
            i = path[k][0]
            j = path[k][1]
            # print "[",a,",",b,"] ", editoparray[a,b]
            labl = editoparray[i, j]
            if labl == 'm':
                if i != pi + 1:
                    for h in range(pi + 1, i):
                        print(" - ", end=' ')
                if self.seqY.ndim == 1:
                    print("%3d" % self.seqY[j], end=' ')
                else:
                    print(self.seqY[j], end=' ')
            elif labl == 's':
                if j != pj + 1:
                    for h in range(pj + 1, j):
                        if self.seqY.ndim == 1:
                            print("%3d" % self.seqY[h], end=' ')
                        else:
                            print(self.seqY[h], end=' ')
                if self.seqY.ndim == 1:
                    print("%3d" % self.seqY[j], end=' ')
                else:
                    print(self.seqY[j], end=' ')
            elif labl == "m" or labl == "s" or labl == "d" or labl == "~" or labl == "=":
                if len(np.shape(self.seqY)) == 1:
                    print("%3d" % self.seqY[j], end=' ')
                else:
                    print(self.seqY[j], end=' ')
            elif labl == "i" or labl == "m":
                print(" - ", end=' ')
            pi = i
            pj = j
        print()

    def backtrack_path_old(self, n1, n2):
        """Returns a list containing the cells on the path ending at indexes (n1, n2).

        Parameters
        ----------
        n1 : int
            First index.
        n2 : int
            Second index.

        Returns
        -------
        list
            list containing the cells on the path ending

        """
        path = [(n1, n2)]  # initialize path to recover with last endpoint
        j = n2
        for i in range(n1):
            # go backward
            k = n1 - i
            path.append(self.bp[k, j])
            j = self.bp[k, j][1]
            if j == -1: break
            # assert(self.bp[k,j][0] == k-1) # for asymmetric constraints
        # print "Backtracked path", path
        return np.array(path)

    def backtrack_path(self, n1, n2):
        """Returns a list containing the cells on the path ending at indexes (n1, n2).

        Parameters
        ----------
        n1 : int
            First index.
        n2 : int
            Second index.

        Returns
        -------
        list
            list containing the cells on the path ending

        """
        assert (n1 != -1 or n2 != -1)
        path = [(n1, n2)]  # initialize path to recover with last endpoint
        i, j = (n1, n2)
        while i != -1 and j != -1:
            tmp_i = i
            tmp_j = j
            i = self.bp[tmp_i, tmp_j][0]
            j = self.bp[tmp_i, tmp_j][1]
            if i != -1 and j != -1:
                path.append((i, j))
            # logging.debug(i,j)

        logging.debug("Backtracked path: {path}")
        return np.array(path)

    def print_results(self, cum_dist_flag=True, bp_flag=False, ld_flag=False,
                      free_ends_flag=False, optimal_path_flag=True, graphic_optimal_path_flag=False,
                      graphic_seq_alignment=False, verbose=True, **kwargs):
        """Print results in terminal.

        Parameters
        ----------
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

        """
        import pandas as pd
        np.set_printoptions(precision=3)
        if verbose:
            print(f"{' INFOS ':*^80}")
            print(f"Sequence X length: {self.nX}")
            print(f"Sequence Y length: {self.nY}")
            print(f"Type of constraints: {self.constraints}")
            print(f"Beam size: ", (self.beam_size if self.beam_size != -1 else "None"))
            print(f"Free endings: {self.free_ends}")
            if self.constraints == 'merge_split':
                print(f"Mixed type: {self.mixed_type}")
                print(f"Mixed spread: {self.mixed_spread}")
                print(f"Mixed weight: {self.mixed_weight}")
            print(f"{' RESULTS ':*^80}")
            print(f"Alignment:")
            self.print_alignment(self.opt_backtrack_path, self.editop)
            print(f"Optimal path length: {len(self.opt_backtrack_path)}")
            print(f"Optimal normalized cost: {self.min_normalized_cost} at cell {self.opt_index} (non normalized: {self.non_mormalized_optcost}")

        if cum_dist_flag:
            print(f"Array of global distances (x downward, y rightward):\n {self.cum_dist}")

        if free_ends_flag and verbose:
            print("Sub-array of normalized distances on relaxed ending region= \n", self.optpath_normalized_cumdist_array)
            print("Sub-array of optimal path lengths on relaxed ending region= \n", self.optpathlength_array)

        data = {}
        if optimal_path_flag:
            if verbose:
                print(f"Optimal path (total norm cost = {self.min_normalized_cost}): ")
            data = self.find_path(self.opt_backtrack_path, self.editop, verbose)
        df = pd.DataFrame(data)

        # Print array of local distances
        if ld_flag and verbose:
            print("Local dist array = \n", self.l_dist)

        # Print back-pointer array
        bparray = np.empty((self.nX, self.nY), dtype=object)
        for i in range(self.nX):
            for j in range(self.nY):
                bparray[i, j] = (self.bp[i, j][0], self.bp[i, j][1])
        if bp_flag and verbose:
            print("Back-pointers array = \n", print_matrix_bp(bparray))

        # Print graphic optimal path
        if graphic_optimal_path_flag:

            plt.figure(0)
            plt.clf()
            # print bparray[:,0]
            # print bparray[:,1]
            plt.plot(self.opt_backtrack_path[:, 0], self.opt_backtrack_path[:, 1])
            plt.ylim([0, self.nY])
            plt.xlim([0, self.nX])
            plt.grid()
            plt.ion()
            plt.show()
            # plt.draw()

            ### FAIRE DES SOUS FIGS
            plt.figure(1)
            plt.clf()
            l = len(self.opt_backtrack_path)
            prev_dist = 0.
            locdist = []
            for i in range(l):
                a = self.opt_backtrack_path[i][0]
                b = self.opt_backtrack_path[i][1]
                locdist.append(self.cum_dist[a, b] - prev_dist)
                prev_dist = self.cum_dist[a, b]
            plt.hist(locdist, bins=20, range=(0, 1))
            plt.xlim([0, 1])
            plt.ion()
            plt.show()

        # Print signal alignment
        if graphic_seq_alignment:
            dim = self.seqX.ndim

            # Loop on the dimensions of test/ref vector space
            for d in range(dim):
                plt.figure(d + 2)  # 0 and 1 are already used
                plt.clf()
                # print bparray[:,0]
                # print bparray[:,1]
                if dim == 1:
                    seqX = self.seqX  # Test sequence
                    seqY = self.seqY  # Ref sequence
                else:
                    seqX = self.seqX[:, d]  # take the dth scalar sequence of the vector-sequence
                    seqY = self.seqY[:, d]

                # Find the best shift of the two sequences
                optpathlen = len(self.opt_backtrack_path)
                test_indexes = np.arange(len(seqX))
                shift = 0
                if True:  # OPTIMIZE_ALIGNMENT_DISPLAY:
                    # compute a more optimal test_index
                    minh = optpathlen
                    maxh = -optpathlen
                    # First find all the shifts that appear in the alignment from test to ref
                    for k in range(optpathlen):
                        i = self.opt_backtrack_path[k, 0]  # test
                        j = self.opt_backtrack_path[k, 1]  # ref
                        delta = j - i
                        if delta < minh:
                            minh = delta
                        if delta > maxh:
                            maxh = delta
                    score_array = np.zeros(maxh - minh + 1)
                    print("-----> minh, maxh=", minh, maxh, )
                    # Second finds a shift s that would best compensate the different shifts:
                    # the aligment would become j - (i+s)
                    for s in range(minh, maxh + 1):
                        score = 0
                        for k in range(optpathlen):
                            i = self.opt_backtrack_path[k, 0]
                            j = self.opt_backtrack_path[k, 1]
                            delta = abs(j - i - s)
                            score += delta
                        score_array[s - minh] = score
                    # shift = minh - index of minimal score
                    print("score array=", score_array)
                    shift = minh + np.argmin(score_array)  # take the first available shift
                    print(" shift = ", shift)
                    test_indexes = np.arange(shift, shift + len(seqX))
                plt.plot(test_indexes, seqX, 'b^', label='test sequence')  # test sequence + shift
                plt.plot(seqY, 'ro', label='ref sequence')  # ref sequence
                pi, pj = -1, -1  # previous i,j
                for k in range(optpathlen):
                    i = self.opt_backtrack_path[k, 0]
                    j = self.opt_backtrack_path[k, 1]
                    if i != pi + 1:
                        for h in range(pi + 1, i):
                            plt.plot([h + shift, j], [seqX[h], seqY[j]], 'g--')
                    elif j != pj + 1:
                        for h in range(pj + 1, j):
                            plt.plot([i + shift, h], [seqX[i], seqY[h]], 'g--')
                    plt.plot([i + shift, j], [seqX[i], seqY[j]], 'g--')
                    pi = i
                    pj = j
                maxval = max(max(seqX), max(seqY)) * 1.2
                plt.ylim([-1, maxval])
                plt.xlim([-1 + shift, max(self.nX, self.nY)])
                plt.xlabel('Rank')
                if dim == 1:
                    plt.ylabel('Sequence Value')
                else:
                    if self.mixed_type[d]:
                        plt.ylabel(str(d) + ' - Angle')
                    else:
                        plt.ylabel(str(d) + ' - Coord')
                plt.title('Comparison of test/ref sequences')
                plt.legend()
                plt.grid()
                plt.ion()
                plt.show()
        return df

    # These two functions add up the attributes of a sub-sequence of X (resp. of Y)
    # from i1 to i2 and compare them to the attribute at index j in sequence Y.
    # - ipair is a pair of integers (i1,i2)
    # Preconditions:
    # rmq: if i1 == i2, then norm(v_i1, v_i2 is returned)
    def ldistcumX(self, ipair, j, ldist):
        """Add up the attributes of a sub-sequence of X.

        Parameters
        ----------
        ipair : (int, int)
            Pair of ???.
        j : int
            Index in sequence Y ???.
        ldist : function, default euclidean_dist
            The function to compute the local distance used to compare values of both sequences.
            Typically `euclidean_dist()`,  `angular_dist()` or `mixed_dist()`.

        Returns
        -------
        float
            ???.

        Notes
        -----
        If ``i1 == i2``, then ``norm(v_i1, v_i2)`` is returned.

        """
        i1, i2 = ipair
        vi = self.seqX[i2]
        v2 = self.seqY[j]
        for i in range(i1, i2):
            vi = vi + self.seqX[i]
        return ldist(vi, v2, is_angular=self.mixed_type, spread=self.mixed_spread, weight=self.mixed_weight)

    def ldistcumY(self, i, jpair, ldist):
        """Add up the attributes of a sub-sequence of Y.

        Parameters
        ----------
        i : int
            Index in sequence X ???.
        jpair : (int, int)
            Pair of ???.
        ldist : function, default euclidean_dist
            The function to compute the local distance used to compare values of both sequences.
            Typically `euclidean_dist()`, `angular_dist()` or `mixed_dist()`.

        Returns
        -------
        float
            ???.

        Notes
        -----
        If ``i1 == i2``, then ``norm(v_i1, v_i2)`` is returned.

        """
        j1, j2 = jpair
        v1 = self.seqX[i]
        vj = self.seqY[j2]
        for j in range(j1, j2):
            vj = vj + self.seqY[j]
        return ldist(v1, vj, is_angular=self.mixed_type, spread=self.mixed_spread, weight=self.mixed_weight)

    def asymmetric_constraints(self, i, j, tmpcumdist, tmpcumdistindexes, ldist, max_stretch):
        """Compute asymmetric constraints.

        Implements constraints from [sakoe_chiba78]_ and [itakura75]_.
        Path may be coming from either ``(i-1,j)``, ``(i-1,j-1)``, ``(i-1,j-2)``
        ( ``(i-i,j)`` only if not coming from `j` at ``i-2``).

        Parameters
        ----------
        i : ?
            ??
        j : ?
            ??
        tmpcumdist : ?
            ??
        tmpcumdistindexes : ?
            ??
        ldist : function, default euclidean_dist
            The function to compute the local distance used to compare values of both sequences.
            Typically `euclidean_dist()`, `angular_dist()` or `mixed_dist()`.
        maxstretch : bool, default 3
            maximum amount of stretching allowed for signal warping.

        References
        ----------
        .. [sakoe_chiba78] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization for spoken word recognition," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 26, no. 1, pp. 43-49, February 1978, doi: `10.1109/TASSP.1978.1163055 <https://doi.org/10.1109/TASSP.1978.1163055>`_
        .. [itakura75] Itakura, Fumitada. "Minimum Prediction Residual Principle Applied to Speech Recognition." IEEE Transactions on Acoustics, Speech, and Signal Processing 23 , no. 1 (1975): 67-72. `10.1109/TASSP.1975.1162641 <https://doi.org/10.1109/TASSP.1975.1162641>`_

      """
        tmpcumdistindexes[0] = (i - 1, j)
        tmpcumdistindexes[1] = (i - 1, j - 1)
        tmpcumdistindexes[2] = (i - 1, j - 2)
        if i == 0:
            tmpcumdist[0] = self.cum_dist_boundaryY[j]
            tmpcumdist[1] = 0.0
            tmpcumdist[2] = np.Infinity
            if j > 0:
                tmpcumdist[1] = self.cum_dist_boundaryY[j - 1]
                tmpcumdist[2] = 0.0
            if j > 1:
                tmpcumdist[2] = self.cum_dist_boundaryY[j - 2]
            # logging.debug(tmpcumdist)
            # logging.debug(np.argmin(tmpcumdist))
        else:
            tmpcumdist[0] = self.cum_dist[i - 1, j]
            tmpcumdist[1] = self.cum_dist_boundaryX[i - 1]
            if j > 0:
                tmpcumdist[1] = self.cum_dist[i - 1, j - 1]
                tmpcumdist[2] = self.cum_dist_boundaryX[i - 1]
            if j > 1:
                tmpcumdist[2] = self.cum_dist[i - 1, j - 2]
            # decision on local optimal path:
            # logging.debug(tmpcumdist)
        if i > 0 and self.bp[i - 1, j][1] == j:  # to forbid horizontal move twice in a raw
            tmpcumdist[0] = np.Infinity
        return tmpcumdist, tmpcumdistindexes

    def symmetric_constraints(self, i, j, tmpcumdist, tmpcumdistindexes, ldist, max_stretch):
        """Compute symmetric constraints.

        Implements constraints from [sakoe_chiba78]_ and [itakura75]_.
        Paths may be coming from either ``(i-1,j)``, ``(i-1,j-1)``, ``(i,j)``
        but cannot go twice consecutively in either horizontal or vertical directions.

        Parameters
        ----------
        i : ?
            ??
        j : ?
            ??
        tmpcumdist : ?
            ??
        tmpcumdistindexes : ?
            ??
        ldist : function, default euclidean_dist
            The function to compute the local distance used to compare values of both sequences.
            Typically `euclidean_dist()`, `angular_dist()` or `mixed_dist()`.
        maxstretch : bool, default 3
            maximum amount of stretching allowed for signal warping.

        Notes
        -----
        Weighting function is not implemented!
        WARNING: Path weights have not yet been implemented (all origins have the same weights).
        This would require score normalization by the sum of weights in the end ...
        one would also have to pass the local distance to this function to make the decision here.

        References
        ----------
        .. [sakoe_chiba78] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization for spoken word recognition," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 26, no. 1, pp. 43-49, February 1978, doi: `10.1109/TASSP.1978.1163055 <https://doi.org/10.1109/TASSP.1978.1163055>`_
        .. [itakura75] Itakura, Fumitada. "Minimum Prediction Residual Principle Applied to Speech Recognition." IEEE Transactions on Acoustics, Speech, and Signal Processing 23 , no. 1 (1975): 67-72. `10.1109/TASSP.1975.1162641 <https://doi.org/10.1109/TASSP.1975.1162641>`_

      """
        tmpcumdistindexes[0] = (i - 1, j)
        tmpcumdistindexes[1] = (i - 1, j - 1)
        tmpcumdistindexes[2] = (i, j - 1)
        if i == 0:
            tmpcumdist[0] = self.cum_dist_boundaryY[j]
            tmpcumdist[1] = 0.0
            tmpcumdist[2] = self.cum_dist_boundaryX[i]
            if j > 0:
                tmpcumdist[1] = self.cum_dist_boundaryY[j - 1]
                tmpcumdist[2] = self.cum_dist[i, j - 1]
        else:
            tmpcumdist[0] = self.cum_dist[i - 1, j]
            tmpcumdist[1] = self.cum_dist_boundaryX[i - 1]
            tmpcumdist[2] = self.cum_dist_boundaryX[i]
            if j > 0:
                tmpcumdist[1] = self.cum_dist[i - 1, j - 1]
                tmpcumdist[2] = self.cum_dist[i, j - 1]

        if i > 0 and self.bp[i - 1, j][1] == j:  # to forbid horizontal move twice in a raw
            tmpcumdist[0] = np.Infinity
        if j > 0 and self.bp[i, j - 1][0] == i:  # to forbid vertical move twice in a raw
            tmpcumdist[2] = np.Infinity
        return tmpcumdist, tmpcumdistindexes

    def editdist_constraints(self, i, j, tmpcumdist, tmpcumdistindexes, ldist, max_stretch):
        """Implements edit distance constraints.

        When processing point `i`,`j`, paths may be coming from either ``(i-1,j)``, ``(i-1,j-1)``, ``(i,j-1)``
        like in the symmetric distance but and can go in principle several times consecutively
        in either horizontal or vertical directions.
        What will drive path construction is matching, insertion and deletion operations and their relative costs.

        Parameters
        ----------
        i : ?
            ??
        j : ?
            ??
        tmpcumdist : ?
            ??
        tmpcumdistindexes : ?
            ??
        ldist : function, default euclidean_dist
            The function to compute the local distance used to compare values of both sequences.
            Typically `euclidean_dist()`, `angular_dist()` or `mixed_dist()`.
        max_stretch : bool, default 3
            maximum amount of stretching allowed for signal warping.

      """
        tmpcumdistindexes[0] = (i - 1, j)
        tmpcumdistindexes[1] = (i - 1, j - 1)
        tmpcumdistindexes[2] = (i, j - 1)
        if i == 0:
            tmpcumdist[0] = self.cum_dist_boundaryY[j]
            tmpcumdist[1] = 0.0
            tmpcumdist[2] = self.cum_dist_boundaryX[i]
            if j > 0:
                tmpcumdist[1] = self.cum_dist_boundaryY[j - 1]
                tmpcumdist[2] = self.cum_dist[i, j - 1]
        else:
            tmpcumdist[0] = self.cum_dist[i - 1, j]
            tmpcumdist[1] = self.cum_dist_boundaryX[i - 1]
            tmpcumdist[2] = self.cum_dist_boundaryX[i]
            if j > 0:
                tmpcumdist[1] = self.cum_dist[i - 1, j - 1]
                tmpcumdist[2] = self.cum_dist[i, j - 1]
        return tmpcumdist, tmpcumdistindexes

    def merge_split_constraints(self, i, j, tmpcumdist, tmpcumdistindexes, ldist, max_stretch):
        """Implements merge/split edit distance constraints.

        When processing point `i`, `j`, paths may be coming from either
        ``(i-k, j)``, ``(i-1, j-1)``, ``(i, j-k)`` to ``(i, j)``.
        What will drive path construction is matching, insertion and deletion operations and their relative costs.

        Parameters
        ----------
        i : ?
            ??
        j : ?
            ??
        tmpcumdist : ?
            ??
        tmpcumdistindexes : ?
            ??
        ldist : function, default euclidean_dist
            The function to compute the local distance used to compare values of both sequences.
            Typically `euclidean_dist()`,  `angular_dist()` or `mixed_dist()`.
        maxstretch : bool, default 3
            maximum amount of stretching allowed for signal warping.

        Notes
        -----
        `tmpcumdist[0]` must contain the min of {D(i-k-1, j) + dcum((i-k)-->i, j)} over k in 1..K, K>=1
        `tmpcumdist[1]` must contain the min of {D(i-1, j) + dcum(i, j)}
        `tmpcumdist[2]` must contain the min of {D(i, j-k-1) + dcum(i, (j-k)-->j )} over k in 1..K, K>=1

        """
        if i == 0 and j == 0:
            d = self.ldistcumX((0, 0), 0, ldist)  # equivalent to ldist(seqi[0],seqj[0])
            tmpcumdist[0] = self.cum_dist_boundaryY[j] + d
            tmpcumdist[1] = d
            tmpcumdist[2] = self.cum_dist_boundaryX[i] + d
            tmpcumdistindexes[0] = (i - 1, j)
            tmpcumdistindexes[1] = (i - 1, j - 1)
            tmpcumdistindexes[2] = (i, j - 1)
        else:
            tmpcumdist[0] = np.Infinity
            tmpcumdist[1] = np.Infinity
            tmpcumdist[2] = np.Infinity

            ii = i - max_stretch  # min index to test within memory `max_stretch`
            jj = j - max_stretch  # min index to test within memory `max_stretch`
            Ki = Kj = max_stretch

            if ii < 0:  # Horizontal initialization
                Ki = max_stretch + ii  # update memory so that min index to test is at least 0
                if j == 0:
                    tmpcumdist[0] = self.ldistcumX((0, i), j, ldist)
                    tmpcumdistindexes[0] = (-1, -1)
                else:
                    tmpcumdist[0] = self.cum_dist_boundaryY[j - 1] + self.ldistcumX((0, i), j, ldist)
                    tmpcumdistindexes[0] = (-1, j - 1)

            if jj < 0:  # Vertical initialization
                Kj = max_stretch + jj  # update memory so that min index to test is at least 0
                if i == 0:
                    tmpcumdist[2] = self.ldistcumY(i, (0, j), ldist)
                    tmpcumdistindexes[2] = (-1, -1)
                else:
                    tmpcumdist[2] = self.cum_dist_boundaryX[i - 1] + self.ldistcumY(i, (0, j), ldist)
                    tmpcumdistindexes[2] = (i - 1, -1)
            # first horizontal
            for k in range(Ki):
                if k == 0: continue  # (diagonal case as j-k-1 = j-1)
                cumD0 = self.cum_dist[i - k - 1, j - 1] + self.ldistcumX((i - k, i), j, ldist)
                if cumD0 < tmpcumdist[0]:
                    tmpcumdist[0] = cumD0
                    tmpcumdistindexes[0] = (i - k - 1, j - 1)

            # Second vertical
            for k in range(Kj):
                if k == 0: continue  # (diagonal case as j-k-1 = j-1)
                cumD2 = self.cum_dist[i - 1, j - k - 1] + self.ldistcumY(i, (j - k, j), ldist)
                if cumD2 < tmpcumdist[2]:
                    tmpcumdist[2] = cumD2
                    tmpcumdistindexes[2] = (i - 1, j - k - 1)

            # Eventually, diagonal case:
            if i == 0:  # we already made sure that then j!=0
                tmpcumdist[1] = self.cum_dist_boundaryY[j - 1] + self.ldistcumY(0, (j, j), ldist)  # equivalent to l_dist(seqi[0],seqj[0])
            elif j == 0:  # we already made sure that then i!=0
                tmpcumdist[1] = self.cum_dist_boundaryX[i - 1] + self.ldistcumX((i, i), 0, ldist)  # equivalent to l_dist(seqi[0],seqj[0])
            else:
                tmpcumdist[1] = self.cum_dist[i - 1, j - 1] + self.ldistcumX((i, i), j, ldist)  # equivalent to l_dist(seqi[0],seqj[0])
            tmpcumdistindexes[1] = (i - 1, j - 1)
        return tmpcumdist, tmpcumdistindexes

    def run(self):
        """Run the DTW algorithm on both sequences.

        Returns
        -------
        float
            The minimum normalized cost.
        numpy.ndarray
            The optimal backtracked path.
        int
            The length of the optimal backtracked path.
        numpy.ndarray
            The optimal path with normalized cumulative distance ???.
        numpy.ndarray
            The back-pointer matrix.

        Notes
        -----
        For the `free_ends`, we must have:

          - ``k+l < min(nt,nr)``
          - ``k >=0`` and ``l>=1``

        See Also
        --------
        euclidean_dist
        angular_dist

        Example
        -------
        >>> import numpy as np
        >>> from dtw.dtw import DTW
        >>> from dtw.dtw import mixed_dist
        >>> seq_test = np.array([[96, 163, 137, 113, 24, 170, 152, 137, 255, 148, 111, 16, 334, 160, 94, 116, 144, 132, 145], [50, 60, 48, 19, 31, 0, 37, 20, 31, 25, 7, 1, 51, 29, 26, 16, 22, 12, 23]]).T
        >>> seq_ref = np.array([[96, 163, 137, 137, 170, 152, 137, 132, 123, 148, 127, 191, 143, 160, 94, 116, 144, 132, 145], [50, 60, 48, 50, 0, 37, 20, 0, 31, 25, 8, 27, 24, 29, 26, 16, 22, 12, 23 ]]).T
        >>> max_ref = np.max(seq_ref[:, 1])
        >>> max_test = np.max(seq_test[:, 1])
        >>> dtwcomputer = DTW(seq_test,seq_ref,constraints='merge_split',ldist=mixed_dist,mixed_type=[True, False],mixed_spread=[1, max(max_ref, max_test)],mixed_weight=[0.5, 0.5])
        >>> ndist, path, length, ndistarray, backpointers = dtwcomputer.run()

        """
        # initialize the arrays of back-pointers and cumulated distance
        self.initdtw()

        if self.constraints == "edit_distance":
            apply_constraints = self.editdist_constraints
        elif self.constraints == "asymmetric":
            apply_constraints = self.asymmetric_constraints
        elif self.constraints == "merge_split":
            apply_constraints = self.merge_split_constraints  # default is symmetric
        elif self.constraints == "symmetric":
            apply_constraints = self.symmetric_constraints  # default is symmetric
        else:
            logging.warning(f"Unknown constraint '{self.constraints}', using `symmetric` by default!")
            apply_constraints = self.symmetric_constraints  # default is symmetric

        # main dtw algorithm
        for i in range(self.nX):
            for j in range(self.nY):
                # take into account the beam size (only make computation in case indexes are not too distorted)
                if self.beam_size == -1 or np.abs(i - j) <= self.beam_size:
                    # temporary cumulated values (here 3) to make the local optimization choice
                    tmpcumdist = np.full(3, np.Infinity)
                    # temporary back-pointers
                    tmpcumdistindexes = np.full((3, 2), -1)
                    v1 = self.seqX[i]
                    v2 = self.seqY[j]
                    ld = self.ldist_f(v1, v2, is_angular=self.mixed_type, spread=self.mixed_spread, weight=self.mixed_weight)
                    # Todo: Check whether path cumcost should be compared in a normalized or non-normalized way
                    # during dtw algo. At the moment, paths are compared in a non-normalized way.
                    # However, the final optimal solution is chosen on the basis of the normalized cost
                    # (which looks a bit inconsistent)
                    tmpcumdist, tmpcumdistindexes = apply_constraints(i, j, tmpcumdist, tmpcumdistindexes, self.ldist_f, self.max_stretch)

                    # Add local distance before selecting optimum
                    # In case of merge_split, nothing must be done as the local distance was already added in apply_constraints
                    if self.constraints == "edit_distance":
                        tmpcumdist[0] = tmpcumdist[0] + self.delins_cost[0]
                        tmpcumdist[1] = tmpcumdist[1] + ld
                        tmpcumdist[2] = tmpcumdist[2] + self.delins_cost[1]
                    elif self.constraints == "asymmetric" or self.constraints == "symmetric":
                        tmpcumdist[0] = tmpcumdist[0] + ld
                        tmpcumdist[1] = tmpcumdist[1] + ld
                        tmpcumdist[2] = tmpcumdist[2] + ld

                    optindex = np.argmin(tmpcumdist)  # index of min distance

                    # case where there exist several identical cum_dist values: choose diagonal direction (index 1)
                    if optindex != 1 and np.isclose(tmpcumdist[optindex], tmpcumdist[1]):
                        optindex = 1

                    # tracks indexes on optimal path
                    # m = matching (defined by np.isclose() ), s =substitution, d = deletion, i = insertion
                    if self.constraints != "merge_split":
                        self.editop[i, j] = "m" if optindex == 1 else "d" if optindex == 2 else "i"
                        if self.editop[i, j] == "m" and not np.isclose(ld, 0):
                            self.editop[i, j] = "s"
                    else:  # case of merge_split
                        # a different strategy is used to label edit operation in case of merge_split
                        # "=" or "~" for a match or a quasi-match,
                        # "m" for a merge (several X have been aggregate to match one Y),
                        # "s" for a split (several Y have been aggregated to match one X)
                        self.editop[i, j] = "=" if optindex == 1 else "s" if optindex == 2 else "m"
                        if self.editop[i, j] == "=" and not np.isclose(ld, 0):
                            self.editop[i, j] = "~"

                    if tmpcumdist[optindex] != np.Infinity:
                        origin = tmpcumdistindexes[optindex]  # points from which optimal is coming (origin)
                        self.bp[i, j] = origin  # back-pointers can have value -1
                        self.cum_dist[i, j] = tmpcumdist[optindex]
                    else:
                        self.bp[i, j] = (-1, -1)
                        self.cum_dist[i, j] = np.Infinity
                else:
                    self.bp[i, j] = (-1, -1)
                    self.cum_dist[i, j] = np.Infinity

        # Recover the solution at the end of matrix computation by backtracking
        # For this,
        # 1. recover all paths in relaxed ending zone, their cost and their length
        # 2. then compare these paths with respect to their normalized cost.
        # 3. Rank the solutions over the relaxed ending zone
        # 4. The optimal path is the one with the minimum normalized cost (may be several)

        # 1. recover optimal paths in relaxed zone
        b = self.free_ends[1]  # size of the relaxed ending region
        self.optpath_array = np.empty((b, b), dtype=object)  # optimal path with these constraints is of size len1
        self.optpathlength_array = np.empty((b, b))  # lengths of optimal paths
        self.optpath_normalized_cumdist_array = np.empty((b, b))  # cum_dist of optimal paths

        # 2/3. Computation of normalized cost and extraction of minimal value
        self.min_normalized_cost = np.Infinity
        self.opt_index = (0, 0)
        for k in range(b):
            for l in range(b):
                logging.debug(f"Back-tracking indexes: {self.nX - k - 1, self.nY - l - 1}")
                self.optpath_array[k, l] = self.backtrack_path(self.nX - k - 1, self.nY - l - 1)
                logging.debug(f"Back-tracking path: {self.optpath_array[k, l]}")
                pathlen = len(self.optpath_array[k, l])
                logging.debug(f"Path length: {pathlen}")
                logging.debug(f"Back-tracked path: {self.optpath_array[k, l]}")
                logging.debug(f"Back-tracked path length: {len(self.optpath_array[k, l])}")
                self.optpathlength_array[k, l] = pathlen  # due to the local constraint used here
                logging.debug(f"Cumulative distances: {self.cum_dist}")
                normalized_cost = self.cum_dist[self.nX - k - 1, self.nY - l - 1] / float(pathlen)
                self.optpath_normalized_cumdist_array[k, l] = normalized_cost
                logging.debug(f"Normalized score: {normalized_cost}")
                if normalized_cost < self.min_normalized_cost:
                    self.min_normalized_cost = normalized_cost
                    index = (self.nX - k - 1, self.nY - l - 1)
                    logging.debug(f"Saved optimal index: {index}")
                    self.non_mormalized_optcost = self.cum_dist[index[0], index[1]]
                    self.opt_index = index

        # 4. Optimal solution
        k, l = self.opt_index[0], self.opt_index[1]
        logging.debug(f"Optimal solution: {k, l}")
        opt_path = self.optpath_array[self.nX - k - 1, self.nY - l - 1]  # retrieve optimal path (listed backward)
        self.opt_backtrack_path = np.flip(opt_path, 0)  # reverse the order of path to start from beginning
        optpathlength = len(self.opt_backtrack_path)
        return self.min_normalized_cost, self.opt_backtrack_path, optpathlength, self.optpath_normalized_cumdist_array, self.bp


def _get_ndist(test_seq, ref_seq, free_ends, **kwargs) -> float:
    """ Creates and run DTW algorithm for selected free-ends."""
    dtwcomputer = DTW(test_seq, ref_seq, free_ends=free_ends, **kwargs)
    _ndist, _, _, _, _ = dtwcomputer.run()
    return _ndist


def brute_force_free_ends_search(dtw, max_value=0.4, free_ends_eps=1e-4, n_jobs=-1):
    """ Explore all free-ends combinations in the prescribed limits.

    This is a brute force method to search the optimal free-ends values.
    The best model is selected using the lowest minimum normalized cost for a pair of free-ends.

    Parameters
    ----------
    max_value : float, optional
        max value for exploration of free-ends on both sequence sides. Default is ``0.4``.
    free_ends_eps : free_ends_eps, optional
        Minimum difference to previous minimum normalized cost to consider tested free-ends as the new best combination.
        Default is ``1e-4``.
    n_jobs : int, optional
        Number of jobs to run in parallel, by default `-1` uses all availables cores.

    Returns
    -------
    tuple
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
        logging.warning("Automatic free-ends capped to 40% of min length on both sides.")

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
    cumdistflag : bool, default True
        If ``True``, print the array of global distances.
    bpflag : bool, default False
        If ``True``, print the back-pointers array.
    ldflag : bool, default False
        If ``True``, print the local distance array.
    freeendsflag : bool, default False
        If ``True``, print the sub-arrays of normalized distances on relaxed ending region and of optimal path lengths on relaxed ending region.
    optimalpathflag : bool, default True
        If ``True``, print the optimal path.
    graphicoptimalpathflag : bool, default True
        If ``True``, ?then?.
    graphicseqalignment : bool, default True
        If ``True``, ?then?.
    verbose : bool, default True
        If ``True``, increase code verbosity.

    Notes
    -----
    For the `free_ends`, we must have:

      - ``k+l < min(nt,nr)``
      - ``k >=0`` and ``l>=1``

    """
    try:
        assert constraint in CONSTRAINTS
    except AssertionError:
        print(f"Valid values for `constraint` parameter: {CONSTRAINTS}")
        raise ValueError(f"Unknown '{constraint}' value for `constraint` parameter.")
    try:
        assert dist_type in DIST_TYPES
    except AssertionError:
        print(f"Valid values for `dist_type` parameter: {DIST_TYPES}")
        raise ValueError(f"Unknown '{dist_type}' value for `dist_type` parameter.")

    if dist_type == "euclidean":
        ld = euclidean_dist
    elif dist_type == "angular":
        ld = angular_dist
    else:  # mixed normalized distance
        ld = mixed_dist

    dtwcomputer = DTW(seqX, seqY, constraints=constraint, delins_cost=delins_cost, ldist=ld,
                      mixed_type=mixed_type, mixed_spread=mixed_spread, mixed_weight=mixed_weight,
                      beamsize=beamsize, max_stretch=max_stretch)
    if type(free_ends) == tuple:
        # if `free_ends` is tuple: NOT AUTOMATIC --> uses the tuple values
        dtwcomputer.free_ends = free_ends
    else:
        # if `free_ends` is NOT tuple: AUTOMATIC --> it is assumed to be int or real <= 0.4
        # and this corresponds to a percentage of sequence length for max exploration of free-ends
        logging.info(f"Starting brute force search...")
        free_ends, norm_dist = brute_force_free_ends_search(dtwcomputer, max_value=free_ends,
                                                            free_ends_eps=free_ends_eps)
        logging.info(f"Found free-ends {free_ends} at a cost of {norm_dist}.")
        # finally computes the DTW for free-ends found
        dtwcomputer.free_ends = free_ends

    _ = dtwcomputer.run()

    return dtwcomputer.print_results(**kwargs)


def duplicate_idx(l, idx, n):
    """Duplicate, n times, the index of a given list.

    Parameters
    ----------
    l : list
        the list to duplicate index
    idx : int
        the index to duplicate
    n : int
        the number of time to duplicate

    Returns
    -------
    list
        the list containing the duplicated index, n times.

    Examples
    --------
    >>> from dtw.dtw import duplicate_idx
    >>> duplicate_idx([0, 1, 2, 3], 0, 2)
    [0, 0, 0, 1, 2, 3]
    >>> duplicate_idx([0, 1, 2, 3], 1, 2)
    [0, 1, 1, 1, 2, 3]
    >>> duplicate_idx([0, 1, 2, 3], 3, 2)
    [0, 1, 2, 3, 3, 3]

    """
    if not isinstance(l, list):
        l = list(l)

    if idx == 0:
        return [l[idx]] * n + l[idx:]
    elif idx == len(l):
        return l[0:idx] + [l[idx]] * n
    else:
        return l[0:idx] + [l[idx]] * n + l[idx:]


# TODO: dtw_eval_summarize & compare_align (test_eval.R)

def dtw_eval_summarize(df, reference_df, verbose=True):
    """ Evaluate DTW predictions.

    It provides metrics to evaluate the correctness of the prediction of the several alignments of a test sequence on their respective reference sequence knowing the true alignments.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe with the column headers ["PlantID" "test_interval" "N_align" "align_score" "dtw_predict" "dtw_true" "dtw_cost" "code_eval" "dtw_eval"].
        Typically the output of ``compare_align``.
    reference_df : pandas.DataFrame
        A dataframe with the column headers ["PlantID" "reference" "modified" "dtw"].
        Typically the ``align_interval.csv`` file generated by ``Rscript test_dtw.R``.
    verbose : bool, optional
        Control the code verbosity.

    Returns
    -------
    pandas.DataFrame
        7-column dataframe containing info and metrics (6 rows), one row per PlantID, the first column is the PlantID

    """
    pass
