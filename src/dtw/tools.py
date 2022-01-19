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
Tools.
"""

def print_matrix_bp(a):
    """Print matrix of back-pointers.

    Parameters
    ----------
    a : numpy.ndarray
        The matrix of back-pointer to print

    Example
    -------
    >>> from dtw.dtw import DTW
    >>> from dtw.dtw import print_matrix_bp
    >>> test_seq = [2, 3, 4, 3, 3, 4, 0, 3, 3, 2, 1, 1, 1, 3, 3, 4, 4]
    >>> ref_seq = [0, 0, 4, 3, 3, 3, 3, 3, 2, 1, 2, 1, 3, 4]
    >>> dtwcomputer = DTW(test_seq, ref_seq)
    >>> ndist, path, length, ndistarray, backpointers = dtwcomputer.run()
    >>> print_matrix_bp(backpointers)

    """
    print("Matrix[" + ("%d" % a.shape[0]) + "][" + ("%d" % a.shape[1]) + "]")
    rows = a.shape[0]
    cols = a.shape[1]
    for i in range(0, rows):
        for j in range(0, cols):
            print(("(%2d,%2d)" % (a[i, j][0], a[i, j][1])), end=' ')  # "%6.f" %a[i,j],
        print()
    print()
