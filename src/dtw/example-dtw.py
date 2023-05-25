#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       File author(s):
#           Christophe Godin <christophe.godin@inria.fr>
#
#       File contributor(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#           Fabrice Besnard <fabrice.besnard@ens-lyon.fr>
#
#       File maintainer(s):
#           Christophe Godin <christophe.godin@inria.fr>
#
#       Mosaic Inria team, RDP Lab, Lyon
# ------------------------------------------------------------------------------

"""# Test of the generic time warping algorithm.

## Create a test
To define a test, create a class with a chosen ``testname`` and create attributes
     of this class corresponding to the desired test (see example below)

     Usually,
     - sequence 1 is considered as the test
     - sequence 2 as the reference


## Run a test
     Simply execute this file in python. It will execute test examples and
     print results of the tests (included at the end of this file):
```bash
     > python test-dtw.py
```
     or in a python shell:
```python
     python> run test-dtw.py
```

"""

import numpy as np
from dtw.dtw import DTW
from dtw.metrics import angular_dist
from dtw.dtw import euclidean_dist
from dtw.metrics import mixed_dist


def runtest(test, cum_dist_flag=True, bp_flag=False, ld_flag=False, free_ends_flag=False, optimal_path_flag=True, graphic_optimal_path_flag=True,
            graphic_seq_alignment=True):
    """Run one of the test examples below."""
    print(f"Test: {test.__name__}")
    print(f"test seq (1) = {test.seq1}")
    print(f"ref  seq (2) = {test.seq2}")

    if hasattr(test, 'constraints'):
        ct = test.constraints
    else:  # default
        ct = "symmetric"

    if hasattr(test, 'dist_type'):
        if test.dist_type == "mixed":
            stg = "mixed"
            ld = mixed_dist
        elif test.dist_type == "angular":
            stg = "angular"
            ld = angular_dist
        else:
            stg = "euclidean"
            ld = euclidean_dist
        print(stg, " distance used for local distance ...")
    else:  # Default
        print("Euclidean distance used for local distance ...")
        ld = euclidean_dist

    if hasattr(test, 'free_ends'):
        fe = test.free_ends
    else:  # Default
        fe = (0, 1)

    if hasattr(test, 'beam_size'):
        bs = test.beam_size
    else:  # Default
        bs = -1

    if hasattr(test, 'delins_cost'):
        dc = test.delins_cost
    else:  # Default
        dc = (1., 1.)

    if hasattr(test, 'mixed_type'):
        mt = test.mixed_type
    else:  # Default
        mt = []

    if hasattr(test, 'mixed_spread'):
        ms = test.mixed_spread
    else:  # Default
        ms = []

    if hasattr(test, 'mixed_weight'):
        mw = test.mixed_weight
    else:  # Default
        mw = []

    dtwcomputer = DTW(test.seq1, test.seq2, constraints=ct, ldist=ld, mixed_type=mt, mixed_spread=ms, mixed_weight=mw,
                      free_ends=fe, beam_size=bs, delins_cost=dc)

    ndist, path, length, ndistarray, backpointers = dtwcomputer.run()

    dtwcomputer.print_results(cum_dist_flag, bp_flag, ld_flag, free_ends_flag, optimal_path_flag, graphic_optimal_path_flag, graphic_seq_alignment)


# Tests definition
class test1:
    seq1 = [2, 3, 4, 3, 3, 4, 0, 3, 3, 2, 1, 1, 1, 3, 3, 4, 4]
    seq2 = [0, 0, 4, 3, 3, 3, 3, 3, 2, 1, 2, 1, 3, 4]
    constraints = "symmetric"  # by default = symmetric
    dist_type = "euclidean"  # not necessary (can be removed). option by default
    free_ends = (0, 3)


class test1_1:
    seq1 = [2, 3, 4, 3, 3, 4, 0, 3, 3, 2, 1, 1, 1, 3, 3, 4, 4]
    seq2 = [0, 0, 4, 3, 3, 3, 3, 3, 2, 1, 2, 1, 3, 4]
    constraints = "symmetric"
    dist_type = "euclidean"
    beamsize = 1  # <---  size of beam bounding distance between indexes
    free_ends = (0, 3)


class test1_2:
    seq1 = [2, 3, 4, 3, 3, 4, 0, 3, 3, 2, 1, 1, 1, 3, 3, 4, 4]
    seq2 = [0, 0, 4, 3, 3, 3, 3, 3, 2, 1, 2, 1, 3, 4]
    constraints = "symmetric"
    dist_type = "euclidean"
    free_ends = (3, 3)  # <--- add free starting point at beginning of length 3 (on both X and Y)


class test1_3:
    seq1 = [2, 3, 4, 3, 3, 4, 0, 3, 3, 2, 1, 1, 1, 3, 3, 4, 4]
    seq2 = [0, 0, 4, 3, 3, 3, 3, 3, 2, 1, 2, 1, 3, 4]
    constraints = "edit_distance"
    delinscost = (5., 5.)
    dist_type = "euclidean"
    free_ends = (3, 3)  # <--- add free starting point at beginning of length 3 (on both X and Y)


class test2:
    seq1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    seq2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    dist_type = "euclidean"
    free_ends = (0, 1)


class test2_1:
    seq1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    seq2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    dist_type = "euclidean"
    free_ends = (0, 3)


class test3:
    seq1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    seq2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    free_ends = (0, 1)


class test4:
    seq1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    seq2 = [4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5]
    free_ends = (2, 3)


class test4_1:
    seq1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    seq2 = [4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5]
    constraints = "edit_distance"
    delinscost = (5., 5.)
    free_ends = (1, 3)


class test5:
    seq1 = [136, 144, 133, 139, 171, 107, 125, 141, 159, 116, 165, 147, 104, 138, 121, 118, 129, 156, 127, 144, 177, 128, 117, 110, 133, 86, 165, 132, 138, 122,
            120, 137, 122, 145, 164, 86, 155, 116, 142, 134, 132, 167, 162]
    seq2 = [200, 103, 100, 133, 137, 171, 107, 125, 141, 159, 116, 165, 147, 104, 138, 121, 118, 129, 156, 127, 144, 177, 128, 117, 110, 133, 86, 165, 132, 138,
            122, 120, 137, 122, 145, 164, 86, 155, 116, 142, 134, 132, 167, 162]
    constraints = "edit_distance"
    delinscost = (100., 100.)
    free_ends = (3, 1)


class test5_1:
    seq1 = [136, 144, 133, 139, 171,
            107]  # ,125,141,159,116] #,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
    seq2 = [200, 103, 100, 133, 139, 171,
            107]  # ,125,141,159,116] #,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
    constraints = "edit_distance"
    delinscost = (50., 50.)
    free_ends = (3, 1)


class test5_2:
    seq1 = [136, 144, 133, 139, 171,
            107]  # ,125,141,159,116] #,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
    seq2 = [200, 103, 100, 133, 139, 171,
            107]  # ,125,141,159,116] #,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
    constraints = "edit_distance"
    delinscost = (50., 50.)
    free_ends = (0, 1)


class test5_3:
    seq1 = [136, 144, 133, 139, 171, 125, 141, 159, 116, 165, 147, 104, 138, 121, 118, 129, 156, 127, 144, 177, 128, 117, 110, 133, 86, 165, 132, 138, 122, 120,
            137, 122, 145, 164, 86, 155, 116, 142, 134, 132, 167, 162]
    seq2 = [200, 103, 100, 133, 139, 171, 125, 141, 159, 116, 165, 147, 104, 138, 121, 118, 129, 156, 127, 144, 177, 128, 117, 110, 133, 86, 165, 132, 138, 122,
            120, 137, 122, 145, 164, 86, 155, 116, 142, 134, 132, 167, 162]
    constraints = "edit_distance"
    delinscost = (50., 50.)
    free_ends = (3, 1)


class test5_4:
    seq1 = [136, 144, 133, 139, 171, 107, 125, 141, 159, 116, 165, 147, 104, 138, 121, 118, 129, 156, 127, 144, 177, 128, 117, 110, 133, 86, 165, 132, 138, 122,
            120, 137, 122, 145, 164, 86, 155, 116, 142, 134, 132, 167, 162]
    seq2 = [200, 103, 100, 133, 137, 171, 107, 125, 141, 159, 116, 165, 147, 104, 138, 121, 118, 129, 156, 127, 144, 177, 128, 117, 110, 133, 86, 165, 132, 138,
            122, 120, 137, 122, 145, 164, 86, 155, 116, 142, 134, 132, 167, 162]
    constraints = "edit_distance"
    delinscost = (50., 50.)
    free_ends = (0, 1)


class test6:
    seq1 = [136, 144, 133, 139, 171, 107, 137]
    seq2 = [136, 144, 133, 310, 107, 137]  # simulates a missing branch in the reconstruction
    constraints = "edit_distance"
    delinscost = (75., 75.)
    free_ends = (0, 1)


# Interesting: for relative angles it seems equivalent or even better to align
# the abherent angle (2 alpha, here 295) with one that is maximal also in the first
# sequence, leading to an arbitrary choice for the inserted angle (here the one
# whose alignment would cost most = 133)
class test6_1:
    seq1 = [136, 144, 133, 139, 171, 107, 137]
    seq2 = [130, 148, 138, 295, 99, 130]  # simulates a noise and missing branch in the reconstruction
    constraints = "edit_distance"
    delinscost = (75., 75.)
    free_ends = (0, 1)


# tests with absolute angles (same positions as in test6)
class test7:
    seq1 = [136, 280, 413, 552, 723, 830, 967]
    seq2 = [136, 280, 413, 723, 830, 967]  # simulates a noise and missing branch in the reconstruction
    constraints = "edit_distance"
    delinscost = (75., 75.)
    free_ends = (0, 1)


# tests with absolute angles (same positions as 6_1 - i.e. with noise)
class test7_1:
    seq1 = [136, 280, 413, 552, 723, 830, 967]
    seq2 = [130, 278, 416, 711, 810, 940]  # simulates a noise and missing branch in the reconstruction
    constraints = "edit_distance"
    delinscost = (75., 75.)
    free_ends = (0, 1)


class test10:  # test of sequences of vectors
    seq1 = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    seq2 = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    constraints = "edit_distance"
    delinscost = (5., 5.)
    free_ends = (0, 1)


class test10_1:  # Testing sequences of vectors as 2dim lists
    seq1 = [[1, 1], [1, 1], [1, 1], [2, 3], [4, 5], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    seq2 = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    constraints = "edit_distance"
    delinscost = (1., 1.)
    free_ends = (0, 1)


class test10_2:  # Testing sequences of vectors as 2D numpy arrays
    seq1 = np.array([[1, 1], [1, 1], [1, 1], [2, 3], [4, 5], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
    seq2 = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]])
    constraints = "edit_distance"
    delinscost = (1., 1.)
    free_ends = (0, 1)


##########################################
# Tests of the MERGE/SPLIT Edit operations
##########################################

class test20:  # testing a merge operation
    seq1 = np.array([1, 1, 1, 2, 2, 1, 5, 3, 2])
    seq2 = np.array([1, 1, 3, 2, 1, 5, 3, 2])
    constraints = "merge_split"
    free_ends = (0, 1)


class test20_1:  # Testing a series of 3 merge operations
    seq1 = np.array([1, 1, 1, 2, 2, 1, 5, 3, 2])
    seq2 = np.array([3, 5, 10])
    constraints = "merge_split"
    free_ends = (0, 1)


class test20_11:  # Testing a series of 3 split operations (reciprocal of test 20_1
    seq1 = np.array([3, 5, 10])
    seq2 = np.array([1, 1, 1, 2, 2, 1, 5, 3, 2])
    constraints = "merge_split"
    free_ends = (0, 1)


class test20_12:  # Testing a series of both merge and split operations
    seq1 = np.array([1, 1, 6, 5, 3, 2])
    seq2 = np.array([2, 3, 2, 1, 10])
    constraints = "merge_split"
    free_ends = (0, 1)


# tests with relative angles (same positions as in test6)
class test20_2:  # 1 merge of two angles
    seq1 = [136, 144, 133, 139, 171, 107, 137]
    seq2 = [136, 144, 133, 310, 107, 137]  # simulates a missing branch in the reconstruction
    constraints = "merge_split"
    free_ends = (0, 1)


# identical to 20_2 but with angular distance (take into account periodicity of angles)
class test20_21:
    seq1 = [136, 144, 133, 139, 171, 107, 137]
    seq2 = [136, 144, 133, 310, 107, 137]  # simulates a missing branch in the reconstruction
    constraints = "merge_split"
    dist_type = "angular"  # Instead of euclidean
    free_ends = (0, 1)


class test20_3:  # simulates a noise and missing branch in the reconstruction
    seq1 = [136, 144, 133, 139, 171, 107, 137]
    seq2 = [130, 148, 138, 295, 99, 130]
    constraints = "merge_split"
    free_ends = (0, 1)


class test20_31:  # When angular distance is taken into account cost is lower.
    seq1 = [136, 144, 133, 139, 171, 107, 137]
    seq2 = [130, 148, 138, 295, 99, 130]  # simulates a noise and missing branch in the reconstruction
    constraints = "merge_split"
    dist_type = "angular"  # Instead of euclidean
    free_ends = (0, 1)


# tests with absolute angles (same positions as in test20_2): show that this does not work
class test20_4:
    seq1 = [136, 280, 413, 552, 723, 830, 967]
    seq2 = [136, 280, 413, 723, 830, 967]  # simulates a missing branch in the reconstruction
    constraints = "merge_split"
    dist_type = "angular"  # Instead of euclidean
    free_ends = (0, 1)


# tests with absolute angles (same positions as 20_2 - i.e. with noise)
class test20_5:
    seq1 = [136, 280, 413, 552, 723, 830, 967]
    seq2 = [130, 278, 416, 711, 810, 940]  # simulates a noise and missing branch in the reconstruction
    constraints = "merge_split"
    dist_type = "angular"  # Instead of euclidean
    free_ends = (0, 1)


class test20_6:
    seq1 = [133, 101, 114, 138, 85, 143, 122, 115, 122, 124, 128, 109, 145, 129, 132, 132, 128, 137, 129, 117, 136, 146, 116, 157, 125, 119, 263, 233, 285, 120,
            110, 147, 131, 143, 268, 222, 271, 140, 133, 141, 129, 133, 138, 138, 162, 133, 145, 136, 145, 142, 129, 119, 136, 136, 148, 136, 118, 128, 138,
            121, 164, 111, 129, 148, 115]
    seq2 = np.full((65,), 137)
    dist_type = "angular"  # Instead of euclidean
    constraints = "merge_split"
    free_ends = (0, 1)


class test20_61:  # Test of a simple inversion (note that this results in a S/M with cost 0 !)
    seq1 = [137, 274, 223, 274, 137]
    seq2 = np.full((5,), 137)
    dist_type = "angular"  # Instead of euclidean
    constraints = "merge_split"
    free_ends = (0, 1)


class test20_62:  # Same thing with noise: note that the costs of the split/merge is of the order of magnitude of the noise on each angle
    seq1 = [133, 114, 138, 268, 222, 271, 140, 133, 141, 129, 133, 138]
    seq2 = np.full((12,), 137)
    dist_type = "angular"  # Instead of euclidean
    constraints = "merge_split"
    free_ends = (2, 2)


# Test of sequences of both angles and internodes
class test21:  # test of sequences of vectors
    seq1 = [[137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10]]
    seq2 = [[137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10]]
    constraints = "merge_split"
    dist_type = "mixed"  # Instead of euclidean
    mixed_type = [True, False]  # first component is of type angle, other are normal coords
    mixed_spread = [1, 10]  # will divide dists to normalize them
    mixed_weight = [1, 1]  # weight of angle dist_type compared with normal coord dist_type
    free_ends = (0, 1)


class test21_1:  # Test sequence perturbed in both angles and internodes
    seq1 = [[138, 10], [120, 2], [140, 11], [139, 10], [130, 8], [137, 12], [125, 9], [137, 5], [139, 11], [137, 10]]
    seq2 = [[137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10], [137, 10]]
    constraints = "merge_split"
    dist_type = "mixed"  # Instead of euclidean
    mixed_type = [True, False]  # first component is of type angle, other are normal coords
    mixed_spread = [1, 10]  # will divide dists to normalize them
    mixed_weight = [1, 1]  # weight of angle dist_type compared with normal coord dist_type
    free_ends = (0, 1)


class test21_2:  # Testing sequences of vectors as 2dim lists
    seq1 = [[138, 10], [120, 2], [140, 11], [139, 10], [130, 10], [137, 10], [125, 9], [137, 5], [139, 11], [137, 10]]
    seq2 = [[137, 10], [137, 10], [137, 10], [274, 10], [223, 1], [274, 20], [137, 10], [137, 10], [137, 10], [137, 10]]
    constraints = "merge_split"
    dist_type = "mixed"  # Instead of euclidean
    mixed_type = [True, False]  # first component is of type angle, other are normal coords
    mixed_spread = [1, 1]  # will divide dists to normalize them
    mixed_weight = [1, 1]  # weight of angle dist_type compared with normal coord dist_type
    free_ends = (0, 1)


# Test execution
# Can switch on/off : cumdistflag, bpflag, freeendsflag, optimalpathflag
# to control display of test results
# Just uncomment the test(s) you want to run.

runtest(test1, free_ends_flag=True, bp_flag=False, ld_flag=False, graphic_optimal_path_flag=False, graphic_seq_alignment=True)
