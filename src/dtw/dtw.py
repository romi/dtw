"""
    Generic time warping algorithm

        Author: Ch. Godin, Inria
        Date 06/04/19
        Mosaic Inria team, RDP Lab, Lyon

    Implementation of a generic DTW algorithm with symmetric asymmetric or edit
    distance constraints based in particular on basic DTW algorithm described in :

    Sakoe, H., & Chiba, S. (1978) Dynamic programming algorithm optimization for spoken
    word recognition. IEEE Transactions on Acoustic, Speech, and Signal Processing,
    ASSP-26(1),43-49.

    ** Call:
    # create a class instance of dtw to compute the distance between two sequences
    dtwcomputer = dtw(seq1,seq2)
    # run the algorithm with specific options
    ndist, optpath, length, ndistarray,backpointers = dtwcomputer.run(ldist = ld, freeends = fe , beamsize = bs)
    # print the results (a number of flags can be turned on/off to display various aspects of the results)
    dtwcomputer.printresults(cumdistflag, bpflag, freeendsflag, optimalpathflag)

    ** args:
    - seq1, seq2: two arrays (lists or np.arrays of scalars or of vectors with identical dimension)

    ** Optional args:
    - ldist (func(val,val)-->positive float)= local distance used for pointwise comparison
    - freeends  ((int,int)): should be greater than (0,1). Elements are respectively
                relaxations at beginning and end of the sequences (identical for both sequences)
    - beamsize  (int): maximum amount of distortion allowed for signal warping

    Returned values:
    - ndist: optimal computed distance between the two sequences
    - optpath: optimal path recovered
    - length: length of the optimal path
    - ndistarray: numpy array of normalized distances
    - backpointers: numpy array of backpointers

    tests: see the file test-dtw for this
"""

import numpy as np
import matplotlib.pyplot as plt

def euclidean_dist(v1,v2):
    return np.linalg.norm(v1-v2)

def angular_dist(a1,a2):
    """
    a1 and a2 are two angles given in degrees.
    The function compares two angles and returns their distance as a percentage of
    the distance of their difference to 180 degrees
    the returned distance is a number between 0 and 1: 0 means the two angles
    are equal up to 360 deg. 1 means that the two angles are separated by 180 degs.
    A distance of 0.5 corresponds to a difference of 90 degs between the two angles.
    """
    Da = a1-a2
    # da is the angle corresponding to the difference between
    # the original angles. 0 <= da < 360.
    da = Da % 360.
    assert(da >=0.)

    return 1 - np.abs(180-da)/180.

# tools
# Print matrix of backpointers
def printMatrixBP(a):
   print("Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]")
   rows = a.shape[0]
   cols = a.shape[1]
   for i in range(0,rows):
      for j in range(0,cols):
         print(("(%2d,%2d)" % (a[i,j][0],a[i,j][1])), end=' ') # "%6.f" %a[i,j],
      print()
   print()

# Main DTW class: to build DTW computer objects on a pair of sequences
class DTW:

  def __init__(self, seq1, seq2):
    # seq1 and seq2 should be are expected to be two np.arrays of elements of identical dim
    # an element can be a scalar or a vector
    self.seqX = np.array(seq1)
    self.seqY = np.array(seq2)
    self.N1 = len(seq1)
    self.N2 = len(seq2)

  def initdtw(self):
    #initiates the arrays of backpointers, localdist and cumdist
    assert(len(self.freeends)==2)
    a = self.freeends[0]
    b = self.freeends[1]
    assert(a+b < self.N1 and a+b < self.N2)

    # initialization of backpointer array
    self.bp = np.empty((self.N1,self.N2), dtype = object)
    # initialization of cummulated distance array
    self.cumdist = np.full((self.N1,self.N2), np.Infinity)

    # edit op
    self.editop = np.full((self.N1,self.N2), "-")

    # border array for boundary conditions on cumdist array
    self.cumdistboundaryX = np.full(self.N1, np.Infinity)
    if a != 0: self.cumdistboundaryX[:a] = 0.0
    self.cumdistboundaryY = np.full(self.N2, np.Infinity)
    if a != 0: self.cumdistboundaryY[:a] = 0.0

    # initialization and computation of the matrix of local distances
    self.ldist = np.full((self.N1,self.N2), np.Infinity)

    for i in range(self.N1):
      for j in range(self.N2):
        self.ldist[i,j] = euclidean_dist(self.seqX[i],self.seqY[j])

  def printPath(self,path, editoparray):
       #print "Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]"
       l = len(path)
       prev_dist = 0.
       for i in range(l):
          a = path[i][0]
          b = path[i][1]
          #print "[",a,",",b,"] ", editoparray[a,b]
          print("[%2d,%2d]"% (a,b), end=' ')
          print(editoparray[a,b], end=' ')
          print("  cost = ", self.cumdist[a,b]-prev_dist)
          prev_dist = self.cumdist[a,b]


  def printAlignment(self,path, editoparray):
       #print "Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]"
       l = len(path)
       for i in range(l):
          a = path[i][0]
          b = path[i][1]
          #print "[",a,",",b,"] ", editoparray[a,b]
          if editoparray[a,b] == "m" or editoparray[a,b] == "s" or editoparray[a,b] == "i":
            if len(np.shape(self.seqX)) == 1: # for scalar values
                print("%3d"% self.seqX[a], end=' ')
            else: print(self.seqX[a], end=' ')         # for vectorial values
          if editoparray[a,b] == "d":
            print("  -", end=' ')
       print()
       for i in range(l):
          a = path[i][0]
          b = path[i][1]
          #print "[",a,",",b,"] ", editoparray[a,b]
          if editoparray[a,b] == "m" or editoparray[a,b] == "s" or editoparray[a,b] == "d":
            if len(np.shape(self.seqY)) == 1:
                print("%3d"% self.seqY[b], end=' ')
            else: print(self.seqY[b], end=' ')
          if editoparray[a,b] == "i":
            print("  -", end=' ')
       print()

  # returns a list containing the cells on the path ending at indexes (n1,n2)
  def backtrack_path_old(self, n1,n2):
    path = [(n1,n2)]  #initialize path to recover with last endpoint
    j = n2
    for i in range(n1):
      # go backward
      k = n1-i
      path.append(self.bp[k,j])
      j = self.bp[k,j][1]
      if j == -1: break
      #assert(self.bp[k,j][0] == k-1) # for asymatric constraints
    #print "Backtracked path", path
    return np.array(path)

  def backtrack_path(self, n1,n2):
    assert (n1 != -1 or n2 != -1)
    path = [(n1,n2)]  #initialize path to recover with last endpoint
    i = n1
    j = n2
    while i != -1 and j != -1:
      tmpi = i
      tmpj = j
      i = self.bp[tmpi,tmpj][0]
      j = self.bp[tmpi,tmpj][1]
      if i != -1 and j != -1:
          path.append((i,j))
      #print i,j

    #print "Backtracked path", path
    return np.array(path)

  def printresults(self, cumdistflag = True, bpflag = False, ldflag = False, freeendsflag = False, optimalpathflag = True, graphicoptimalpathflag=False):
    print("**************   INFOS    ***************")
    print("len seq1 = ", self.N1, "len seq2 = ", self.N2)
    print("Type of constraints : ", self.constraints)
    print("Beam size = ", (self.beamsize if self.beamsize != -1 else "None"), ", Free endings = ", self.freeends)
    print("**************  RESULTS   ***************")
    print("Alignment = ")
    self.printAlignment(self.optbacktrackpath, self.editop)
    print("Optimal path length = ", len(self.optbacktrackpath))
    print("Optimal normalized cost = %3.f"% self.minnormalizedcost, "at cell",self.optindex, "(non normalized =", self.nonmormalizedoptcost," )")
    if cumdistflag: print("Array of global distances = (x downward, y rightward)\n", self.cumdist)
    if freeendsflag:
        print("Subarray of normalized distances on relaxed ending region= \n", self.optpathnormalizedcumdist_array)
        print("Subarray of optimal path lengths on relaxed ending region= \n", self.optpathlength_array)
    if optimalpathflag:
        print("Optimal path = ")
        self.printPath(self.optbacktrackpath, self.editop)

    # Print array of local distances
    if ldflag: print("Local dist array = \n", self.ldist)

    # Print backpointer array
    bparray = np.empty( (self.N1,self.N2),dtype=object)
    for i in range(self.N1):
      for j in range(self.N2):
        bparray[i,j]= (self.bp[i,j][0],self.bp[i,j][1])
    if bpflag: print("Backpointers array = \n", printMatrixBP(bparray))

    # Print graphic optimal path
    if True :
        plt.figure(0)
        plt.clf()
        #print bparray[:,0]
        #print bparray[:,1]
        plt.plot(self.optbacktrackpath[:,0],self.optbacktrackpath[:,1])
        plt.ylim([0, self.N2])
        plt.xlim([0, self.N1])
        plt.grid()
        plt.show()
        plt.draw()


  def asymmetric_constraints(self,i,j,tmpcumdist,tmpcumdistindexes):
      """
      Implements constraints from (see Sakoe-Chiba 73 and Itakura 1975):
      path may be coming from either (i-1,j), (i-1,j-1), (i-1,j-2)
      ( (i-i,j) only if not coming from j at i-2).
      """
      tmpcumdistindexes[0]=(i-1,j)
      tmpcumdistindexes[1]=(i-1,j-1)
      tmpcumdistindexes[2]=(i-1,j-2)
      if i == 0:
        tmpcumdist[0] = self.cumdistboundaryY[j]
        tmpcumdist[1] = 0.0
        tmpcumdist[2] = np.Infinity
        if j > 0:
          tmpcumdist[1] = self.cumdistboundaryY[j-1]
          tmpcumdist[2] = 0.0
        if j > 1:
          tmpcumdist[2] = self.cumdistboundaryY[j-2]
        #print tmpcumdist
        #print np.argmin(tmpcumdist)
      else:
        tmpcumdist[0] = self.cumdist[i-1,j]
        tmpcumdist[1] = self.cumdistboundaryX[i-1]
        if j > 0:
          tmpcumdist[1] = self.cumdist[i-1,j-1]
          tmpcumdist[2] = self.cumdistboundaryX[i-1]
        if j > 1:
          tmpcumdist[2] = self.cumdist[i-1,j-2]
        # decision on local optimal path:
        #print tmpcumdist
      if i > 0 and self.bp[i-1,j][1] == j: # to forbid horizontal move twice in a raw
        tmpcumdist[0] = np.Infinity

  def symmetric_constraints(self,i,j,tmpcumdist,tmpcumdistindexes):
      """
      Implements constraints from (see Sakoe and Chiba 1773 and 1978):
      paths may be coming from either (i-1,j), (i-1,j-1), (i,j)
      but cannot go twice consecutivelly in either horizontal or vertical
      directions. Weighting function is not implemented

      WARNING: Path weights have not yet been implemented (all origins have the same weights).
      This would require score normalization by the sum of weights in the end ...
      one would also have to pass the local distance to this function to make the decision here.
      """
      tmpcumdistindexes[0]=(i-1,j)
      tmpcumdistindexes[1]=(i-1,j-1)
      tmpcumdistindexes[2]=(i,j-1)
      if i == 0:
        tmpcumdist[0] = self.cumdistboundaryY[j]
        tmpcumdist[1] = 0.0
        tmpcumdist[2] = self.cumdistboundaryX[i]
        if j > 0:
          tmpcumdist[1] = self.cumdistboundaryY[j-1]
          tmpcumdist[2] = self.cumdist[i,j-1]
      else:
        tmpcumdist[0] = self.cumdist[i-1,j]
        tmpcumdist[1] = self.cumdistboundaryX[i-1]
        tmpcumdist[2] = self.cumdistboundaryX[i]
        if j > 0:
          tmpcumdist[1] = self.cumdist[i-1,j-1]
          tmpcumdist[2] = self.cumdist[i,j-1]

      if i > 0 and self.bp[i-1,j][1] == j : # to forbid horizontal move twice in a raw
        tmpcumdist[0] = np.Infinity
      if j > 0 and self.bp[i,j-1][0] == i : # to forbid vertical move twice in a raw
        tmpcumdist[2] = np.Infinity


  def editdist_constraints(self,i,j,tmpcumdist,tmpcumdistindexes):
      """
      Implements edit distance constraints. Paths may be coming from either
      (i-1,j), (i-1,j-1), (i,j) like in the symmetric distance
      but and can go in principle several times consecutivelly in either horizontal or vertical
      directions. What will drive path construction is matching, insertion and deletion
      operations and their relative costs.
      """
      tmpcumdistindexes[0]=(i-1,j)
      tmpcumdistindexes[1]=(i-1,j-1)
      tmpcumdistindexes[2]=(i,j-1)
      if i == 0:
        tmpcumdist[0] = self.cumdistboundaryY[j]
        tmpcumdist[1] = 0.0
        tmpcumdist[2] = self.cumdistboundaryX[i]
        if j > 0:
          tmpcumdist[1] = self.cumdistboundaryY[j-1]
          tmpcumdist[2] = self.cumdist[i,j-1]
      else:
        tmpcumdist[0] = self.cumdist[i-1,j]
        tmpcumdist[1] = self.cumdistboundaryX[i-1]
        tmpcumdist[2] = self.cumdistboundaryX[i]
        if j > 0:
          tmpcumdist[1] = self.cumdist[i-1,j-1]
          tmpcumdist[2] = self.cumdist[i,j-1]

  def run(self, ldist = euclidean_dist, constraints = "SYMMETRIC", delinscost=(1.0,1.0), freeends = (0,1), beamsize = -1 ):
    """
    Carries out the DTW algorithm on both sequences
    - ldist is the local distance used to compare values of both sequences
    - freeends is a tuple of 2 integers (k,l) that specifies relaxation bounds on
    the alignement of sequences endpoints: relaxed by k at the sequence beginning
    and relaxed by l at the sequence ending. Note that k+l must be < min(N1,N2).
    Note that we must have: k >=0 and l>=1
    """
    self.freeends = freeends
    self.beamsize = beamsize
    self.constraints = constraints
    # initialize the arrays of backpointers and cumulated distance
    self.initdtw()

    if constraints == "EDITDISTANCE":  apply_constraints = self.editdist_constraints
    elif constraints == "ASYMMETRIC": apply_constraints = self.asymmetric_constraints
    else: apply_constraints = self.symmetric_constraints # default is SYMMETRIC

    # main dtw algorithm
    for i in range(self.N1):
      for j in range(self.N2):
        #take into account the beam size (only make computation in case indexes are not too distorted)
        if self.beamsize == -1 or np.abs(i-j) <= self.beamsize :
          # temporary cumulated values (here 3) to make the local optimization choice
          tmpcumdist = np.full(3,np.Infinity)
          # temporary backpointers
          tmpcumdistindexes = np.full((3,2),-1)
          v1 = self.seqX[i]
          v2 = self.seqY[j]
          ld = ldist(v1,v2)
          # Todo: Check whether path cumcost should be compared in a normalized or non-normalized way
          # during dtw algo. At the moment, paths are compared in a non-normalized way.
          # However, the final optimal solution is chosen on the basis of the normalized cost
          # (which looks a bit inconsistent)
          apply_constraints(i,j,tmpcumdist,tmpcumdistindexes)

          # Add local distance before selecting optimum

          if constraints == "EDITDISTANCE":
            tmpcumdist[0] = tmpcumdist[0] + delinscost[0]
            tmpcumdist[1] = tmpcumdist[1] + ld
            tmpcumdist[2] = tmpcumdist[2] + delinscost[1]
          else:
            tmpcumdist[0] = tmpcumdist[0] + ld
            tmpcumdist[1] = tmpcumdist[1] + ld
            tmpcumdist[2] = tmpcumdist[2] + ld

          optindex = np.argmin(tmpcumdist) # index of min distance

          # case where there exist several identical cumdist values: choose diagonal direction (index 1)
          if optindex != 1 and np.isclose(tmpcumdist[optindex], tmpcumdist[1]):
            optindex = 1

          # tracks indexes on optimal path
          # m = matching (defined by np.isclose() ), s =substitution, d = deletion, i = insertion
          self.editop[i,j] = "m" if optindex == 1 else "d" if optindex == 2 else "i"
          if self.editop[i,j] == "m" and not np.isclose(ld, 0):
              self.editop[i,j] = "s"

          if tmpcumdist[optindex] != np.Infinity:
            origin = tmpcumdistindexes[optindex] # points from which optimal is coming (origin)
            self.bp[i,j] = origin # backpointers can have value -1
            self.cumdist[i,j] = tmpcumdist[optindex]
          else:
            self.bp[i,j] = (-1,-1)
            self.cumdist[i,j] = np.Infinity
        else:
          self.bp[i,j] = (-1,-1)
          self.cumdist[i,j] = np.Infinity

    # Recover the solution at the end of matrix computation by backtracking
    # For this,
    # 1. recover all paths in relaxed ending zone, their cost and their length
    # 2. then compare these paths with respect to their normalized cost.
    # 3. Rank the solutions over the relaxed ending zone
    # 4. The optimal path is the one with the minimum normalized cost (may be several)

    # 1. recover optimal paths in relaxed zone
    b = freeends[1] # size of the relaxed ending region
    self.optpath_array = np.empty((b,b), dtype = object) # optimal path with these constraints is of size len1
    self.optpathlength_array = np.empty((b,b)) # lengths of optimal paths
    self.optpathnormalizedcumdist_array = np.empty((b,b)) # cumdist of optimal paths

     # 2/3. Computation of normalized cost and extraction of minimal value
    self.minnormalizedcost = np.Infinity
    self.optindex = (0,0)
    for k in range(b):
      for l in range(b):
        #print "Backtracking indexes: ", self.N1-k-1, self.N2-l-1
        #print self.backtrack_path(self.N1-k-1, self.N2-l-1)
        self.optpath_array[k,l] = self.backtrack_path(self.N1-k-1, self.N2-l-1)
        pathlen = len(self.optpath_array[k,l])
        #print "pathlen = ", pathlen
        #print "Backtracked path", self.optpath_array[k,l]
        #print "pathlen =", pathlenpath
        self.optpathlength_array[k,l] = pathlen # due to the local constraint used here
        normalizedcost = self.cumdist[self.N1-k-1, self.N2-l-1]/float(pathlen)
        self.optpathnormalizedcumdist_array[k,l] = normalizedcost
        #print "Normalized score = ", normalizedcost
        if normalizedcost < self.minnormalizedcost:
          self.minnormalizedcost = normalizedcost
          index = (self.N1-k-1,self.N2-l-1)
          #print "---> Saved optindex = ", index
          self.nonmormalizedoptcost = self.cumdist[index[0],index[1]]
          self.optindex = index


    # 4. Optimal solution
    k,l = self.optindex[0],self.optindex[1]
    #print k,l
    optpath = self.optpath_array[self.N1-k-1,self.N2-l-1] # retreive opt path (listed backward)
    self.optbacktrackpath = np.flip(optpath, 0) # reverse the order of path to start from beginning
    optpathlength = len(self.optbacktrackpath)
    return self.minnormalizedcost, self.optbacktrackpath, optpathlength, self.optpathnormalizedcumdist_array,self.bp

######### FOR TESTING THE MODULE ##########
def runtest(test, cumdistflag = True, bpflag = False, ldflag = False, freeendsflag = False, optimalpathflag = True, graphicoptimalpathflag= False):
    print("Test: ", test.__name__)
    print("seq1 = ", test.seq1)
    print("seq2 = ", test.seq2)
    dtwcomputer = DTW(test.seq1, test.seq2)
    if hasattr(test, 'constraints'): ct = test.constraints
    else: ct = "SYMMETRIC"
    if hasattr(test, 'disttype'):
        if test.disttype == "EUCLIDEAN":
            stg = "EUCLIDEAN"
            ld = euclidean_dist
        elif test.disttype == "ANGULAR":
            stg = "ANGULAR"
            ld = angular_dist
        else:
            stg = "EUCLIDEAN"
            ld = euclidean_dist
        print(stg, " distance used for local distance ...")
    else:
        print("EUCLIDEAN distance used for local distance ...")
        ld = euclidean_dist
    if hasattr(test, 'freeends'): fe = test.freeends
    else: fe = (0,1)
    if hasattr(test, 'beamsize'): bs = test.beamsize
    else: bs = -1
    if hasattr(test, 'delinscost'): dc = test.delinscost
    else: dc = (1.,1.)

    ndist, path, length, ndistarray,backpointers = dtwcomputer.run(ldist = ld, constraints = ct, delinscost = dc, freeends = fe , beamsize = bs)
    dtwcomputer.printresults(cumdistflag, bpflag, ldflag, freeendsflag, optimalpathflag, graphicoptimalpathflag)
