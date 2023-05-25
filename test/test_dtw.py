#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       File author(s):
#           Christophe Godin <christophe.godin@inria.fr>
#
#       File contributor(s):
#           Fabrice Besnard <fabrice.besnard@ens-lyon.fr>
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       File maintainer(s):
#           Christophe Godin <christophe.godin@inria.fr>
#
#       Mosaic Inria team, RDP Lab, Lyon
# ------------------------------------------------------------------------------

"""Need to add tests for:
- every type of constraint: "merge_split", "edit_distance", "asymmetric", "symmetric"
- mixed distances (phyllotaxis sequences?)
- brute force search of free-ends
- cost evaluation ?.
"""

import unittest
import numpy as np

from dtw.dtw import DTW


class TestDtw(unittest.TestCase):

    def setUp(self):
        self.dtwcomputer = None

    def tearDown(self):
        pass

    def test0(self):
        seq1 = [2, 3, 4, 3, 3, 4, 0, 3, 3, 2, 1, 1, 1, 3, 3, 4, 4]
        seq2 = [2, 3, 4, 3, 3, 4, 0, 3, 3, 2, 1, 1, 1, 3, 3, 4, 4]
        self.dtwcomputer = DTW(seq1, seq2)

        ndist, path, length, ndistarray, backpointers = self.dtwcomputer.run()
        # self.dtwcomputer.printresults()

        self.assertAlmostEqual(ndist, 0)

    def test1(self):
        seq1 = [2, 3, 4, 3, 3, 4, 0, 3, 3, 2, 1, 1, 1, 3, 3, 4, 4]
        seq2 = [0, 0, 4, 3, 3, 3, 3, 3, 2, 1, 2, 1, 3, 4]
        self.dtwcomputer = DTW(seq1, seq2)

        expected_path = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8], [10, 9], [11, 10], [12, 11], [13, 12], [14, 12],
                         [15, 13], [16, 13]]

        ndist, path, length, ndistarray, backpointers = self.dtwcomputer.run()

        # self.dtwcomputer.printresults()

        np.testing.assert_array_equal(path, expected_path)
