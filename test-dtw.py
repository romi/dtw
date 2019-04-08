"""
    Test of the Generic time warping algorithm

        Author: Ch. Godin, Inria
        Date 06/04/19

     Create a test:
     To define a test, create a class with a chosen testname and create attributes
     of this class corresponding to the desired test (see example below)

     Run a test: simply execute this file in python. It will execute test examples and
     print results of the tests (included at the end of this file):
     > python dtw.py

     ou ds un shell python:

     python> run dtw.py

"""

import dtw

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

class test10: # test of sequences of vectors
  seq1 = [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
  seq2 = [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
  constraints = "EDITDISTANCE"
  delinscost = (5.,5.)
  freeends=(0,1)

class test10_1:
  seq1 = [[1,1],[1,1],[1,1],[2,3],[4,5],[1,1],[1,1],[1,1],[1,1],[1,1]]
  seq2 = [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
  constraints = "EDITDISTANCE"
  delinscost = (1.,1.)
  freeends=(0,1)

# Test execution
# Can swithch on/off : cumdistflag, bpflag, freeendsflag, optimalpathflag
# to control display of test results

dtw.runtest(test1, freeendsflag = True)
#dtw.runtest(test1_1, freeendsflag = True)
#dtw.runtest(test1_2, freeendsflag = True)
#dtw.runtest(test1_3, freeendsflag = True)
#dtw.runtest(test2, freeendsflag = True)
#dtw.runtest(test2_1, freeendsflag = True)
#dtw.runtest(test3, freeendsflag = True)
#dtw.runtest(test4_1, freeendsflag = True, bpflag = False)

#dtw.runtest(test10, freeendsflag = True)
#dtw.runtest(test10_1, freeendsflag = True)
