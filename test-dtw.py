"""
    Test of the Generic time warping algorithm

        Author: Ch. Godin, Inria
        Date 06/04/19
        Mosaic Inria team, RDP Lab, Lyon

     ** Create a test:
     To define a test, create a class with a chosen testname and create attributes
     of this class corresponding to the desired test (see example below)

     ** Run a test:
     Simply execute this file in python. It will execute test examples and
     print results of the tests (included at the end of this file):
     > python test-dtw.py

     or in a python shell:

     python> run test-dtw.py

"""

import dtw
import numpy as np

# Tests definition
class test1:
  seq1 = [2,3,4,3,3,4,0,3,3,2,1,1,1,3,3,4,4]
  seq2 = [0,0,4,3,3,3,3,3,2,1,2,1,3,4]
  constraints = "SYMMETRIC"  # by default = SYMMETRIC
  disttype = "EUCLIDEAN"    # not necessary (can be removed). option by default
  freeends=(0,3)

class test1_1:
  seq1 = [2,3,4,3,3,4,0,3,3,2,1,1,1,3,3,4,4]
  seq2 = [0,0,4,3,3,3,3,3,2,1,2,1,3,4]
  constraints = "SYMMETRIC"
  dist = "EUCLIDEAN"
  beamsize = 1      # <---  size of beam bounding distance between indexes
  freeends=(0,3)

class test1_2:
  seq1 = [2,3,4,3,3,4,0,3,3,2,1,1,1,3,3,4,4]
  seq2 = [0,0,4,3,3,3,3,3,2,1,2,1,3,4]
  constraints = "SYMMETRIC"
  dist = "EUCLIDEAN"
  freeends=(3,3)    # <--- add free starting point at beginning of length 3 (on both X and Y)

class test1_3:
  seq1 = [2,3,4,3,3,4,0,3,3,2,1,1,1,3,3,4,4]
  seq2 = [0,0,4,3,3,3,3,3,2,1,2,1,3,4]
  constraints = "EDITDISTANCE"
  delinscost = (5.,5.)
  dist = "EUCLIDEAN"
  freeends=(3,3)    # <--- add free starting point at beginning of length 3 (on both X and Y)

class test2:
  seq1 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
  seq2 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
  dist = "EUCLIDEAN"
  freeends=(0,1)

class test2_1:
  seq1 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
  seq2 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
  dist = "EUCLIDEAN"
  freeends=(0,3)

class test3:
  seq1 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
  seq2 = [1,1,1,1,1,1,1,1,1,1,1,1,1]
  freeends=(0,1)

class test4:
  seq1 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
  seq2 = [4,4,1,1,1,1,1,1,1,1,1,1,1,5,5]
  freeends=(2,3)

class test4_1:
  seq1 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
  seq2 = [4,4,1,1,1,1,1,1,1,1,1,1,1,5,5]
  constraints = "EDITDISTANCE"
  delinscost = (5.,5.)
  freeends=(1,3)

class test5:
  seq1 = [136,144,133,139,171,107,125,141,159,116,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
  seq2 = [200,103,100,133,137,171,107,125,141,159,116,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
  constraints = "EDITDISTANCE"
  delinscost = (100.,100.)
  freeends=(3,1)

class test5_1:
  seq1 = [136,144,133,139,171,107]#,125,141,159,116] #,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
  seq2 = [200,103,100,133,139,171,107]#,125,141,159,116] #,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
  constraints = "EDITDISTANCE"
  delinscost = (50.,50.)
  freeends=(3,1)

class test5_2:
  seq1 = [136,144,133,139,171,107]#,125,141,159,116] #,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
  seq2 = [200,103,100,133,139,171,107]#,125,141,159,116] #,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
  constraints = "EDITDISTANCE"
  delinscost = (50.,50.)
  freeends=(0,1)

class test5_3:
  seq1 = [136,144,133,139,171,125,141,159,116,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
  seq2 = [200,103,100,133,139,171,125,141,159,116,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
  constraints = "EDITDISTANCE"
  delinscost = (50.,50.)
  freeends=(3,1)

class test5_4:
  seq1 = [136,144,133,139,171,107,125,141,159,116,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
  seq2 = [200,103,100,133,137,171,107,125,141,159,116,165,147,104,138,121,118,129,156,127,144,177,128,117,110,133,86,165,132,138,122,120,137,122,145,164,86,155,116,142,134,132,167,162]
  constraints = "EDITDISTANCE"
  delinscost = (50.,50.)
  freeends=(0,1)

class test6:
  seq1 = [136,144,133,139,171,107,137]
  seq2 = [136,144,133,310,107,137] # simulates a missing branch in the reconstruction
  constraints = "EDITDISTANCE"
  delinscost = (75.,75.)
  freeends=(0,1)

# Interesting: for relative angles it seems equivalent or even better to align
# the abherent angle (2 alpha, here 295) with one that is maximal also in the first
# sequence, leading to an arbitrary choice for the inserted angle (here the one
# whose alignment would cost most = 133)
class test6_1:
  seq1 = [136,144,133,139,171,107,137]
  seq2 = [130,148,138,295,99,130] # simulates a noise and missing branch in the reconstruction
  constraints = "EDITDISTANCE"
  delinscost = (75.,75.)
  freeends=(0,1)

# tests with absolute angles (same positions as in test6)
class test7:
  seq1 = [136,280,413,552,723,830,967]
  seq2 = [136,280,413,723,830,967] # simulates a noise and missing branch in the reconstruction
  constraints = "EDITDISTANCE"
  delinscost = (75.,75.)
  freeends=(0,1)

# tests with absolute angles (same positions as 6_1 - i.e. with noise)
class test7_1:
  seq1 = [136,280,413,552,723,830,967]
  seq2 = [130,278,416,711,810,940] # simulates a noise and missing branch in the reconstruction
  constraints = "EDITDISTANCE"
  delinscost = (75.,75.)
  freeends=(0,1)

class test10: # test of sequences of vectors
  seq1 = [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
  seq2 = [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
  constraints = "EDITDISTANCE"
  delinscost = (5.,5.)
  freeends=(0,1)

class test10_1: # Testing sequences of vectors as 2dim lists
  seq1 = [[1,1],[1,1],[1,1],[2,3],[4,5],[1,1],[1,1],[1,1],[1,1],[1,1]]
  seq2 = [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
  constraints = "EDITDISTANCE"
  delinscost = (1.,1.)
  freeends=(0,1)

class test10_2: # Testing sequences of vectors as 2D numpy arrays
  seq1 = np.array([[1,1],[1,1],[1,1],[2,3],[4,5],[1,1],[1,1],[1,1],[1,1],[1,1]])
  seq2 = np.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])
  constraints = "EDITDISTANCE"
  delinscost = (1.,1.)
  freeends=(0,1)

# Test execution
# Can swithch on/off : cumdistflag, bpflag, freeendsflag, optimalpathflag
# to control display of test results
# Just uncomment the test(s) you want to run.

"""
dtw.runtest(test1, freeendsflag = True)
dtw.runtest(test1_1, freeendsflag = True)
dtw.runtest(test1_2, freeendsflag = True)
dtw.runtest(test1_3, freeendsflag = True)
dtw.runtest(test2, freeendsflag = True)
dtw.runtest(test2_1, freeendsflag = True)
dtw.runtest(test3, freeendsflag = True)
dtw.runtest(test4_1, freeendsflag = True, bpflag = False)
dtw.runtest(test10, freeendsflag = True)
dtw.runtest(test10_1, freeendsflag = True)
"""
dtw.runtest(test6_1, freeendsflag = True, bpflag = False, ldflag=False, graphicoptimalpathflag=True)
