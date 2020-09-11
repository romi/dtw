"""
    Generic dynamic time warping algorithms

        Author: Ch. Godin, Inria
        Date 06/04/19 - July 2020
        Mosaic Inria team, RDP Lab, Lyon

    Implementation of a generic DTW algorithm with symmetric asymmetric or classical edit
    distance or split-merge constraints.

    DTW techniques are based in particular on basic DTW algorithm described in :

    Sakoe, H., & Chiba, S. (1978) Dynamic programming algorithm optimization for spoken
    word recognition. IEEE Transactions on Acoustic, Speech, and Signal Processing,
    ASSP-26(1),43-49.

    and new dynamic time warping based techniques such as MERGE_SPLIT.

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
    - Different algorithms are available and can be selected by the following flags:
    SYMMETRIC, ASYMMETRIC, EDITDISTANCE, MERGE_SPLIT
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

    tests: see the file example-dtw for this
"""

import numpy as np
import matplotlib.pyplot as plt


def euclidean_dist(v1, v2, **args):
    return np.linalg.norm(v1 - v2)

# Can only be called for scalar arguments a1 and a2
def angular_dist(a1, a2, **args):
    """
    a1 and a2 are two angles given in degrees.
    The function compares two angles and returns their distance as a percentage of
    the distance of their difference to 180 degrees
    the returned distance is a number between 0 and 1: 0 means the two angles
    are equal up to 360 deg. 1 means that the two angles are separated by 180 degs.
    A distance of 0.5 corresponds to a difference of 90 degs between the two angles.
    """
    Da = a1 - a2
    # da is the angle corresponding to the difference between
    # the original angles. 0 <= da < 360.
    da = Da % 360.
    #assert (da >= 0.)

    return 1 - np.abs(180 - da) / 180.

# to use this angular distance with numpy arrays
vangular_dist = np.vectorize(angular_dist)

# Can only be called for scalar arguments a1 and a2
def mixed_dist(v1, v2, type=[], spread=[], weight=[]):
    """
    computes a distance where normal components are mixed with periodic ones (here angles)
    - v1 and v2 are the input vectors to compared (of same dimension = dim)
    - type is a boolean (dim) vector indicating by a boolean value whether the kth component should be treated as an angle (True) or a regular scalar value
    - spread is (dim) vector of positive scalars used to normalize the dist values computed for each component with their typical spread
    - weight is a (dim) vector of positive weights (does not necessarily sum to 1 - but normalized if not - )

    The resulting distance is:

    D(v1,v2) = sqrt ( sum_k { weight[k] * d_k^2(v1[k],v2[k])/ spread[k]^2})

    where d_k is a distance that depends on type[k]
    """

    # default values
    dim = len(v1)
    if type == []:
        type = np.full((dim,), False)  # by default type indicates only normal v1_coords
    if spread == []:
        spread = np.full((dim,), 1)    # spread will not modify the distance by default
    if weight == []:
        weight = np.full((dim,), 1)    # equal weights by default

    # if the array alpha is not normalized, it is is normalized first here
    weight = np.array(weight)
    sumweight = sum(weight) # should be 1
    if not np.isclose(sumweight,1.0):
        weight = weight/sum(weight)

    # Extract the subarrays corresponding to angles types and coord types resp.

    nottype = np.invert(type)          # not type
    dim1 = np.count_nonzero(type)      # nb of True values in type
    dim2 = np.count_nonzero(nottype)   # nb of False values in type

    v1_angles = np.extract(type,v1)    # subarray of angles only (dim1)
    v1_coords = np.extract(nottype,v1) # subarray of coords only (dim2)

    v2_angles = np.extract(type,v2)    # idem for v2
    v2_coords = np.extract(nottype,v2)

    weight1 = np.extract(type,weight)    # subarray of weights for angles
    weight2 = np.extract(nottype,weight) # subarray of weights for coords

    spread1 = np.extract(type,spread)    # subarray of spread factors for angles
    spread2 = np.extract(nottype,spread) # subarray of spread factors for coords

    if not dim1 == 0:
        DD1 = vangular_dist(v1_angles,v2_angles)**2 # angle dist (squared)
        DD1 = DD1 / spread1**2
        DD1 = DD1 * weight1 # adding weights
    else:
        DD1 = []

    # case of normal coordinates
    if not dim2 == 0:
        DD2 = (v1_coords-v2_coords)**2  # euclidean L2 norm (no sqrt yet)
        DD2 = DD2 / spread2**2
        DD2 = DD2 * weight2 # adding weights
    else:
        DD2 = []

    DD = sum(DD1) + sum(DD2)
    return np.sqrt(DD)


# tools
# Print matrix of backpointers
def printMatrixBP(a):
    print("Matrix[" + ("%d" % a.shape[0]) + "][" + ("%d" % a.shape[1]) + "]")
    rows = a.shape[0]
    cols = a.shape[1]
    for i in range(0, rows):
        for j in range(0, cols):
            print(("(%2d,%2d)" % (a[i, j][0], a[i, j][1])), end=' ')  # "%6.f" %a[i,j],
        print()
    print()


# Main DTW class: to build DTW computer objects on a pair of sequences
class Dtw:

    def __init__(self, seq1, seq2):
        # seq1 and seq2 are expected to be two np.arrays of elements of identical dim
        # an element can be a scalar or a vector
        self.seqX = np.array(seq1)
        self.seqY = np.array(seq2)
        self.N1 = len(seq1)
        self.N2 = len(seq2)

    def initdtw(self):
        # initiates the arrays of backpointers, localdist and cumdist
        assert (len(self.freeends) == 2)
        a = self.freeends[0]
        b = self.freeends[1]
        assert (a + b < self.N1 and a + b < self.N2)

        # initialization of backpointer array
        self.bp = np.empty((self.N1, self.N2), dtype=object)
        # initialization of cummulated distance array
        self.cumdist = np.full((self.N1, self.N2), np.Infinity)

        # edit op
        self.editop = np.full((self.N1, self.N2), "-")

        # border array for boundary conditions on cumdist array
        self.cumdistboundaryX = np.full(self.N1, np.Infinity)
        if a != 0: self.cumdistboundaryX[:a] = 0.0
        self.cumdistboundaryY = np.full(self.N2, np.Infinity)
        if a != 0: self.cumdistboundaryY[:a] = 0.0

        # initialization and computation of the matrix of local distances
        self.ldist = np.full((self.N1, self.N2), np.Infinity)

        for i in range(self.N1):
            for j in range(self.N2):
                self.ldist[i, j] = euclidean_dist(self.seqX[i], self.seqY[j])

    def printPath(self, path, editoparray):
        # print "Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]"
        l = len(path)
        prev_dist = 0.
        for i in range(l):
            a = path[i][0]
            b = path[i][1]
            # print "[",a,",",b,"] ", editoparray[a,b]
            print("%2d : [%2d,%2d]" % (i, a, b), end=' ')
            print(editoparray[a, b], end=' ')
            print("  cost = %6.3f"% (self.cumdist[a, b] - prev_dist))
            prev_dist = self.cumdist[a, b]

    def printAlignment(self, path, editoparray):
        # print "Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]"
        print("test seq: ", end=' ')
        l = len(path)
        pi = pj = -1
        for k in range(l):
            i = path[k][0]
            j = path[k][1]
            #print ("[",a,",",b,"] ", editoparray[a,b])
            labl =  editoparray[i,j]
            if labl == 'M' :
                if i != pi+1:
                    for h in range(pi+1,i):
                        if self.seqX.ndim == 1:
                            print("%3d" % self.seqX[h], end=' ')
                        else:
                            print(self.seqX[h], end=' ')
                if self.seqX.ndim == 1:
                    print("%3d" % self.seqX[i], end=' ')
                else:
                    print(self.seqX[i], end=' ')
            elif labl == 'S':
                if j != pj+1:
                    for h in range(pj+1,j):
                        print(" - ", end=' ')
                if self.seqX.ndim == 1:
                    print("%3d" % self.seqX[i], end=' ')
                else:
                    print(self.seqX[i], end=' ')
            elif labl == "m" or labl == "s" or labl == "i" or labl == "M" or labl == "~" or labl == "=":
                if len(np.shape(self.seqX)) == 1:  # for scalar values
                    print("%3d" % self.seqX[i], end=' ')
                else:
                    print(self.seqX[i], end=' ')  # for vectorial values
            elif labl == "d" or labl == "S":
                print(" - ", end='S')
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
            if labl == 'M' :
                if i != pi+1:
                    for h in range(pi+1,i):
                        print(" - ", end=' ')
                if self.seqY.ndim == 1:
                    print("%3d" % self.seqY[j], end=' ')
                else:
                    print(self.seqY[j], end=' ')
            elif labl == 'S':
                if j != pj+1:
                    for h in range(pj+1,j):
                        if self.seqY.ndim == 1:
                            print("%3d" % self.seqY[h], end=' ')
                        else:
                            print(self.seqY[h], end=' ')
                if self.seqY.ndim == 1:
                    print("%3d" % self.seqY[j], end=' ')
                else:
                    print(self.seqY[j], end=' ')
            elif labl == "m" or labl == "s" or labl == "d" or labl == "S" or labl == "~" or labl == "=":
                if len(np.shape(self.seqY)) == 1:
                    print("%3d" % self.seqY[j], end=' ')
                else:
                    print(self.seqY[j], end=' ')
            elif labl == "i" or labl == "M":
                print(" - ", end=' ')
            pi = i
            pj = j
        print()

    # returns a list containing the cells on the path ending at indexes (n1,n2)
    def backtrack_path_old(self, n1, n2):
        path = [(n1, n2)]  # initialize path to recover with last endpoint
        j = n2
        for i in range(n1):
            # go backward
            k = n1 - i
            path.append(self.bp[k, j])
            j = self.bp[k, j][1]
            if j == -1: break
            # assert(self.bp[k,j][0] == k-1) # for asymatric constraints
        # print "Backtracked path", path
        return np.array(path)

    def backtrack_path(self, n1, n2):
        assert (n1 != -1 or n2 != -1)
        path = [(n1, n2)]  # initialize path to recover with last endpoint
        i = n1
        j = n2
        while i != -1 and j != -1:
            tmpi = i
            tmpj = j
            i = self.bp[tmpi, tmpj][0]
            j = self.bp[tmpi, tmpj][1]
            if i != -1 and j != -1:
                path.append((i, j))
            # print i,j

        # print "Backtracked path", path
        return np.array(path)

    def printresults(self, cumdistflag=True, bpflag=False, ldflag=False, freeendsflag=False, optimalpathflag=True,graphicoptimalpathflag=True,graphicseqalignment=True):
        print("**************   INFOS    ***************")
        print("len seq test (1) = ", self.N1, ", len seq ref (2) = ", self.N2)
        print("Type of constraints : ", self.constraints)
        print("Beam size = ", (self.beamsize if self.beamsize != -1 else "None"), ", Free endings = ", self.freeends)
        if self.constraints == 'MERGE_SPLIT':
            print("Mixed_type = ", self.mixed_type, ", Mixed_spread = ", self.mixed_spread, ", Mixed_weight = ", self.mixed_weight)
        print("**************  RESULTS   ***************")
        print("Alignment = ")
        self.printAlignment(self.optbacktrackpath, self.editop)
        print("Optimal path length = ", len(self.optbacktrackpath))
        print("Optimal normalized cost = %6.3f" % self.minnormalizedcost, "at cell", self.optindex, "(non normalized = %6.3f" % self.nonmormalizedoptcost, " )")
        np.set_printoptions(precision=3)
        if cumdistflag: print("Array of global distances = (x downward, y rightward)\n", self.cumdist)
        if freeendsflag:
            print("Subarray of normalized distances on relaxed ending region= \n", self.optpathnormalizedcumdist_array)
            print("Subarray of optimal path lengths on relaxed ending region= \n", self.optpathlength_array)
        if optimalpathflag:
            print("Optimal path (total norm cost = %6.3f)= "% self.minnormalizedcost)
            self.printPath(self.optbacktrackpath, self.editop)

        # Print array of local distances
        if ldflag: print("Local dist array = \n", self.ldist)

        # Print backpointer array
        bparray = np.empty((self.N1, self.N2), dtype=object)
        for i in range(self.N1):
            for j in range(self.N2):
                bparray[i, j] = (self.bp[i, j][0], self.bp[i, j][1])
        if bpflag: print("Backpointers array = \n", printMatrixBP(bparray))

        # Print graphic optimal path
        if graphicoptimalpathflag:

            plt.figure(0)
            plt.clf()
            # print bparray[:,0]
            # print bparray[:,1]
            plt.plot(self.optbacktrackpath[:, 0], self.optbacktrackpath[:, 1])
            plt.ylim([0, self.N2])
            plt.xlim([0, self.N1])
            plt.grid()
            plt.ion()
            plt.show()
            #plt.draw()

            ### FAIRE DES SOUS FIGS
            plt.figure(1)
            plt.clf()
            l = len(self.optbacktrackpath)
            prev_dist = 0.
            locdist = []
            for i in range(l):
                a = self.optbacktrackpath[i][0]
                b = self.optbacktrackpath[i][1]
                locdist.append(self.cumdist[a, b] - prev_dist)
                prev_dist = self.cumdist[a, b]
            plt.hist(locdist, bins = 10)
            plt.xlim([0, 1])
            plt.ion()
            plt.show()

        # Print signal alignemment
        if graphicseqalignment:
            dim = self.seqX.ndim

            # Loop on the dimensions of test/ref vector space
            for d in range(dim):
                plt.figure(d+2) # 0 and 1 are already used
                plt.clf()
                # print bparray[:,0]
                # print bparray[:,1]
                if dim == 1:
                    seqX = self.seqX # Test sequencce
                    seqY = self.seqY # Ref sequence
                else:
                    seqX = self.seqX[:,d] # take the dth scalar sequence of the vector-sequence
                    seqY = self.seqY[:,d]

                # Find the best shift of the two sequences
                optpathlen = len(self.optbacktrackpath)
                test_indexes = np.arange(len(seqX))
                shift = 0
                if True: #OPTIMIZE_ALIGNMENT_DISPLAY:
                    # compute a more optimal test_index
                    minh = optpathlen
                    maxh = -optpathlen
                    # First find all the shifts that appear in the aligment from test to ref
                    for k in range(optpathlen):
                        i = self.optbacktrackpath[k, 0] # test
                        j = self.optbacktrackpath[k, 1] # ref
                        delta = j-i
                        if delta < minh:
                            minh = delta
                        if delta > maxh:
                            maxh = delta
                    scorearray = np.zeros(maxh-minh+1)
                    print("-----> minh,maxh=",minh,maxh,)
                    # Second finds a shift s that would best compensate the different shifts:
                    # the aligment would become j - (i+s)
                    for s in range(minh,maxh+1):
                        score = 0
                        for k in range(optpathlen):
                            i = self.optbacktrackpath[k, 0]
                            j = self.optbacktrackpath[k, 1]
                            delta = abs(j-i-s)
                            score += delta
                        scorearray[s-minh]=score
                    # shift = minh - index of minimal score
                    print("score array=", scorearray)
                    shift = minh + np.argmin(scorearray) # take the first available shift
                    print(" shift = ", shift)
                    test_indexes = np.arange(shift,shift+len(seqX))
                plt.plot(test_indexes,seqX, 'b^', label='test sequence') # test sequence + shift
                plt.plot(seqY, 'ro', label='ref sequence')  # ref sequence
                pi,pj = -1,-1 # previous i,j
                for k in range(optpathlen):
                    i = self.optbacktrackpath[k, 0]
                    j = self.optbacktrackpath[k, 1]
                    if i != pi+1:
                        for h in range(pi+1,i):
                            plt.plot([h+shift,j],[seqX[h],seqY[j]], 'g--')
                    elif j != pj+1:
                        for h in range(pj+1,j):
                            plt.plot([i+shift,h],[seqX[i],seqY[h]], 'g--')
                    plt.plot([i+shift,j],[seqX[i],seqY[j]], 'g--')
                    pi = i
                    pj = j
                maxval = max(max(seqX),max(seqY))*1.2
                plt.ylim([-1, maxval])
                plt.xlim([-1+shift, max(self.N1,self.N2)])
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



    # These two functions add up the attributes of a sub-sequence of X (resp. of Y)
    # from i1 to i2 and compare them to the attribute at index j in sequence Y.
    # - ipair is a pair of integers (i1,i2)
    # Preconditions:
    # rmq: if i1 == i2, then norm(v_i1, v_i2 is returned)
    def ldistcumX(self,ipair,j,ldist):
        i1, i2 = ipair
        vi = self.seqX[i2]
        v2 = self.seqY[j]
        for i in range(i1,i2):
            vi = vi + self.seqX[i]
        return ldist(vi,v2,type = self.mixed_type,spread = self.mixed_spread, weight = self.mixed_weight)

    def ldistcumY(self,i,jpair,ldist):
        j1, j2 = jpair
        v1 = self.seqX[i]
        vj = self.seqY[j2]
        for j in range(j1,j2):
            vj = vj + self.seqY[j]
        return ldist(v1,vj,type = self.mixed_type,spread = self.mixed_spread, weight = self.mixed_weight)

    def asymmetric_constraints(self, i, j, tmpcumdist, tmpcumdistindexes, ldist, max_stretch):
        """
      Implements constraints from (see Sakoe-Chiba 73 and Itakura 1975):
      path may be coming from either (i-1,j), (i-1,j-1), (i-1,j-2)
      ( (i-i,j) only if not coming from j at i-2).
      """
        tmpcumdistindexes[0] = (i - 1, j)
        tmpcumdistindexes[1] = (i - 1, j - 1)
        tmpcumdistindexes[2] = (i - 1, j - 2)
        if i == 0:
            tmpcumdist[0] = self.cumdistboundaryY[j]
            tmpcumdist[1] = 0.0
            tmpcumdist[2] = np.Infinity
            if j > 0:
                tmpcumdist[1] = self.cumdistboundaryY[j - 1]
                tmpcumdist[2] = 0.0
            if j > 1:
                tmpcumdist[2] = self.cumdistboundaryY[j - 2]
            # print tmpcumdist
            # print np.argmin(tmpcumdist)
        else:
            tmpcumdist[0] = self.cumdist[i - 1, j]
            tmpcumdist[1] = self.cumdistboundaryX[i - 1]
            if j > 0:
                tmpcumdist[1] = self.cumdist[i - 1, j - 1]
                tmpcumdist[2] = self.cumdistboundaryX[i - 1]
            if j > 1:
                tmpcumdist[2] = self.cumdist[i - 1, j - 2]
            # decision on local optimal path:
            # print tmpcumdist
        if i > 0 and self.bp[i - 1, j][1] == j:  # to forbid horizontal move twice in a raw
            tmpcumdist[0] = np.Infinity

    def symmetric_constraints(self, i, j, tmpcumdist, tmpcumdistindexes, ldist, max_stretch):
        """
      Implements constraints from (see Sakoe and Chiba 1773 and 1978):
      paths may be coming from either (i-1,j), (i-1,j-1), (i,j)
      but cannot go twice consecutivelly in either horizontal or vertical
      directions. Weighting function is not implemented

      WARNING: Path weights have not yet been implemented (all origins have the same weights).
      This would require score normalization by the sum of weights in the end ...
      one would also have to pass the local distance to this function to make the decision here.
      """
        tmpcumdistindexes[0] = (i - 1, j)
        tmpcumdistindexes[1] = (i - 1, j - 1)
        tmpcumdistindexes[2] = (i, j - 1)
        if i == 0:
            tmpcumdist[0] = self.cumdistboundaryY[j]
            tmpcumdist[1] = 0.0
            tmpcumdist[2] = self.cumdistboundaryX[i]
            if j > 0:
                tmpcumdist[1] = self.cumdistboundaryY[j - 1]
                tmpcumdist[2] = self.cumdist[i, j - 1]
        else:
            tmpcumdist[0] = self.cumdist[i - 1, j]
            tmpcumdist[1] = self.cumdistboundaryX[i - 1]
            tmpcumdist[2] = self.cumdistboundaryX[i]
            if j > 0:
                tmpcumdist[1] = self.cumdist[i - 1, j - 1]
                tmpcumdist[2] = self.cumdist[i, j - 1]

        if i > 0 and self.bp[i - 1, j][1] == j:  # to forbid horizontal move twice in a raw
            tmpcumdist[0] = np.Infinity
        if j > 0 and self.bp[i, j - 1][0] == i:  # to forbid vertical move twice in a raw
            tmpcumdist[2] = np.Infinity

    def editdist_constraints(self, i, j, tmpcumdist, tmpcumdistindexes, ldist, max_stretch):
        """
      Implements edit distance constraints.
      When processing point i,j, paths may be coming from either
      (i-1,j), (i-1,j-1), (i,j-1) like in the symmetric distance
      but and can go in principle several times consecutivelly in either horizontal or vertical
      directions. What will drive path construction is matching, insertion and deletion
      operations and their relative costs.
      """
        tmpcumdistindexes[0] = (i - 1, j)
        tmpcumdistindexes[1] = (i - 1, j - 1)
        tmpcumdistindexes[2] = (i, j - 1)
        if i == 0:
            tmpcumdist[0] = self.cumdistboundaryY[j]
            tmpcumdist[1] = 0.0
            tmpcumdist[2] = self.cumdistboundaryX[i]
            if j > 0:
                tmpcumdist[1] = self.cumdistboundaryY[j - 1]
                tmpcumdist[2] = self.cumdist[i, j - 1]
        else:
            tmpcumdist[0] = self.cumdist[i - 1, j]
            tmpcumdist[1] = self.cumdistboundaryX[i - 1]
            tmpcumdist[2] = self.cumdistboundaryX[i]
            if j > 0:
                tmpcumdist[1] = self.cumdist[i - 1, j - 1]
                tmpcumdist[2] = self.cumdist[i, j - 1]

    def merge_split_constraints(self, i, j, tmpcumdist, tmpcumdistindexes, ldist, max_stretch):
        """
      Implements merge/split edit distance constraints.
      When processing point i,j, paths may be coming from either
      (i-k,j), (i-1,j-1), (i,j-k) to (i,j). What will drive path construction is matching, insertion and deletion
      operations and their relative costs.

      tmpcumdist[0] must contain the min of {D(i-k-1,j)+dcum( (i-k)-->i , j)} over k in 1..K, K>=1
      tmpcumdist[1] must contain the min of {D(i-1,j)+dcum(i,j)}
      tmpcumdist[2] must contain the min of {D(i,j-k-1)+dcum( i , (j-k)-->j )} over k in 1..K, K>=1
      """
        K = max_stretch
        if i == 0 and j == 0:
            d = self.ldistcumX([0,0],0,ldist) # equivalent to ldist(seqi[0],seqj[0])
            tmpcumdist[0] = self.cumdistboundaryY[j] + d
            tmpcumdist[1] = d
            tmpcumdist[2] = self.cumdistboundaryX[i] + d
            tmpcumdistindexes[0] = (i - 1, j)
            tmpcumdistindexes[1] = (i - 1, j - 1)
            tmpcumdistindexes[2] = (i, j - 1)
        else:
            tmpcumdist[0] = np.Infinity
            tmpcumdist[1] = np.Infinity
            tmpcumdist[2] = np.Infinity

            ii = i - K # min index to test within memory K
            jj = j - K # min index to test within memory K
            Ki = Kj = K

            if ii < 0: # Horizontal initialization
                Ki = K + ii # update memory so that min index to test is at least 0
                if j == 0:
                    tmpcumdist[0] = self.ldistcumX([0,i],j,ldist)
                    tmpcumdistindexes[0] = (-1, -1)
                else:
                    tmpcumdist[0] = self.cumdistboundaryY[j - 1] + self.ldistcumX([0,i],j,ldist)
                    tmpcumdistindexes[0] = (-1, j - 1)

            if jj < 0: # Vertical initialization
                Kj = K + jj # update memory so that min index to test is at least 0
                if i == 0:
                    tmpcumdist[2] = self.ldistcumY(i,[0,j],ldist)
                    tmpcumdistindexes[2] = (-1, -1)
                else:
                    tmpcumdist[2] = self.cumdistboundaryX[i - 1] + self.ldistcumY(i,[0,j],ldist)
                    tmpcumdistindexes[2] = (i - 1, -1)
            # first horizontal
            for k in range(Ki):
                if k == 0: continue # (diagonal case as j-k-1 = j-1)
                cumD0 =  self.cumdist[i - k - 1, j - 1] + self.ldistcumX([i - k,i],j,ldist)
                if cumD0 < tmpcumdist[0]:
                    tmpcumdist[0] = cumD0
                    tmpcumdistindexes[0] = (i - k - 1, j - 1)

            # Second vertical
            for k in range(Kj):
                if k == 0: continue # (diagonal case as j-k-1 = j-1)
                cumD2 =  self.cumdist[i - 1, j - k - 1] + self.ldistcumY(i,[j - k,j],ldist)
                if cumD2 < tmpcumdist[2]:
                    tmpcumdist[2] = cumD2
                    tmpcumdistindexes[2] = (i - 1, j - k - 1)

            # Eventually, diagonal case:
            if i == 0: # we already made sure that then j!=0
                tmpcumdist[1] =  self.cumdistboundaryY[j - 1] + self.ldistcumY(0,[j,j],ldist) # equivalent to ldist(seqi[0],seqj[0])
            elif j == 0: # we already made sure that then i!=0
                tmpcumdist[1] =  self.cumdistboundaryX[i - 1] + self.ldistcumX([i,i],0,ldist) # equivalent to ldist(seqi[0],seqj[0])
            else:
                tmpcumdist[1] =  self.cumdist[i - 1, j - 1] + self.ldistcumX([i,i],j,ldist) # equivalent to ldist(seqi[0],seqj[0])
            tmpcumdistindexes[1] = (i - 1, j - 1)



    def run(self, ldist=euclidean_dist, constraints="SYMMETRIC", delinscost=(1.0, 1.0), mixed_type = [], mixed_spread = [], mixed_weight = [], freeends=(0, 1), beamsize=-1, max_stretch=3):
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

        # for mixed_mode
        self.mixed_type = mixed_type
        self.mixed_spread = mixed_spread
        self.mixed_weight = mixed_weight

        # initialize the arrays of backpointers and cumulated distance
        self.initdtw()

        if constraints == "EDITDISTANCE":
            apply_constraints = self.editdist_constraints
        elif constraints == "ASYMMETRIC":
            apply_constraints = self.asymmetric_constraints
        elif constraints == "MERGE_SPLIT":
            apply_constraints = self.merge_split_constraints  # default is SYMMETRIC
        else:
            apply_constraints = self.symmetric_constraints  # default is SYMMETRIC

        # main dtw algorithm
        for i in range(self.N1):
            for j in range(self.N2):
                # take into account the beam size (only make computation in case indexes are not too distorted)
                if self.beamsize == -1 or np.abs(i - j) <= self.beamsize:
                    # temporary cumulated values (here 3) to make the local optimization choice
                    tmpcumdist = np.full(3, np.Infinity)
                    # temporary backpointers
                    tmpcumdistindexes = np.full((3, 2), -1)
                    v1 = self.seqX[i]
                    v2 = self.seqY[j]
                    ld = ldist(v1, v2, type = self.mixed_type,spread = self.mixed_spread, weight = self.mixed_weight)
                    # Todo: Check whether path cumcost should be compared in a normalized or non-normalized way
                    # during dtw algo. At the moment, paths are compared in a non-normalized way.
                    # However, the final optimal solution is chosen on the basis of the normalized cost
                    # (which looks a bit inconsistent)
                    apply_constraints(i, j, tmpcumdist, tmpcumdistindexes, ldist, max_stretch)

                    # Add local distance before selecting optimum
                    # In case of MERGE_SPLIT, nothing must be done as the local distance was already added in apply_constraints
                    if constraints == "EDITDISTANCE":
                        tmpcumdist[0] = tmpcumdist[0] + delinscost[0]
                        tmpcumdist[1] = tmpcumdist[1] + ld
                        tmpcumdist[2] = tmpcumdist[2] + delinscost[1]
                    elif constraints == "ASYMMETRIC" or constraints == "SYMMETRIC":
                        tmpcumdist[0] = tmpcumdist[0] + ld
                        tmpcumdist[1] = tmpcumdist[1] + ld
                        tmpcumdist[2] = tmpcumdist[2] + ld

                    optindex = np.argmin(tmpcumdist)  # index of min distance

                    # case where there exist several identical cumdist values: choose diagonal direction (index 1)
                    if optindex != 1 and np.isclose(tmpcumdist[optindex], tmpcumdist[1]):
                        optindex = 1

                    # tracks indexes on optimal path
                    # m = matching (defined by np.isclose() ), s =substitution, d = deletion, i = insertion
                    if constraints != "MERGE_SPLIT":
                        self.editop[i, j] = "m" if optindex == 1 else "d" if optindex == 2 else "i"
                        if self.editop[i, j] == "m" and not np.isclose(ld, 0):
                            self.editop[i, j] = "s"
                    else: # case of MERGE_SPLIT
                        # a different strategy is used to label edit operation in case of MERGE_SPLIT
                        # "=" or "~" for a match or a quasi-match,
                        # "M" for a merge (several X have been aggregate to match one Y),
                        # "S" for a split (several Y have been aggregated to match one X)
                        self.editop[i, j] = "=" if optindex == 1 else "S" if optindex == 2 else "M"
                        if self.editop[i, j] == "=" and not np.isclose(ld, 0):
                            self.editop[i, j] = "~"

                    if tmpcumdist[optindex] != np.Infinity:
                        origin = tmpcumdistindexes[optindex]  # points from which optimal is coming (origin)
                        self.bp[i, j] = origin  # backpointers can have value -1
                        self.cumdist[i, j] = tmpcumdist[optindex]
                    else:
                        self.bp[i, j] = (-1, -1)
                        self.cumdist[i, j] = np.Infinity
                else:
                    self.bp[i, j] = (-1, -1)
                    self.cumdist[i, j] = np.Infinity

        # Recover the solution at the end of matrix computation by backtracking
        # For this,
        # 1. recover all paths in relaxed ending zone, their cost and their length
        # 2. then compare these paths with respect to their normalized cost.
        # 3. Rank the solutions over the relaxed ending zone
        # 4. The optimal path is the one with the minimum normalized cost (may be several)

        # 1. recover optimal paths in relaxed zone
        b = freeends[1]  # size of the relaxed ending region
        self.optpath_array = np.empty((b, b), dtype=object)  # optimal path with these constraints is of size len1
        self.optpathlength_array = np.empty((b, b))  # lengths of optimal paths
        self.optpathnormalizedcumdist_array = np.empty((b, b))  # cumdist of optimal paths

        # 2/3. Computation of normalized cost and extraction of minimal value
        self.minnormalizedcost = np.Infinity
        self.optindex = (0, 0)
        for k in range(b):
            for l in range(b):
                # print "Backtracking indexes: ", self.N1-k-1, self.N2-l-1
                # print self.backtrack_path(self.N1-k-1, self.N2-l-1)
                self.optpath_array[k, l] = self.backtrack_path(self.N1 - k - 1, self.N2 - l - 1)
                pathlen = len(self.optpath_array[k, l])
                # print "pathlen = ", pathlen
                # print "Backtracked path", self.optpath_array[k,l]
                # print "pathlen =", pathlenpath
                self.optpathlength_array[k, l] = pathlen  # due to the local constraint used here
                normalizedcost = self.cumdist[self.N1 - k - 1, self.N2 - l - 1] / float(pathlen)
                self.optpathnormalizedcumdist_array[k, l] = normalizedcost
                # print "Normalized score = ", normalizedcost
                if normalizedcost < self.minnormalizedcost:
                    self.minnormalizedcost = normalizedcost
                    index = (self.N1 - k - 1, self.N2 - l - 1)
                    # print "---> Saved optindex = ", index
                    self.nonmormalizedoptcost = self.cumdist[index[0], index[1]]
                    self.optindex = index

        # 4. Optimal solution
        k, l = self.optindex[0], self.optindex[1]
        # print k,l
        optpath = self.optpath_array[self.N1 - k - 1, self.N2 - l - 1]  # retreive opt path (listed backward)
        self.optbacktrackpath = np.flip(optpath, 0)  # reverse the order of path to start from beginning
        optpathlength = len(self.optbacktrackpath)
        return self.minnormalizedcost, self.optbacktrackpath, optpathlength, self.optpathnormalizedcumdist_array, self.bp


# Call this function from outside to launch the comparison of two sequences
def runCompare(seq1,seq2,
               constraint_type = 'MERGE_SPLIT', dist_type = 'EUCLIDEAN',
               mixed_type = [], mixed_spread = [], mixed_weight = [],
               freeends = (0,1), beamsize = -1, delinscost = (1.,1.),
               cumdistflag=False, bpflag=False, ldflag=False, freeendsflag=False,
               optimalpathflag=True, graphicoptimalpathflag=True):

    dtwcomputer = Dtw(seq1, seq2)
    ct = constraint_type
    if dist_type == "EUCLIDEAN":
        stg = "EUCLIDEAN"
        ld = euclidean_dist
    elif dist_type == "ANGULAR":
        stg = "ANGULAR"
        ld = angular_dist
    else: # mixed and normalize distance
        stg = "MIXED"
        ld = mixed_dist
    fe = freeends
    bs = beamsize
    dc = delinscost
    ndist, path, length, ndistarray, backpointers = dtwcomputer.run(ldist=ld, constraints=ct, delinscost=dc,mixed_type = mixed_type, mixed_spread = mixed_spread, mixed_weight = mixed_weight, freeends=fe, beamsize=bs)
    dtwcomputer.printresults(cumdistflag, bpflag, ldflag, freeendsflag, optimalpathflag, graphicoptimalpathflag)




######### FOR TESTING THE MODULE ##########
def runtest(test, cumdistflag=True, bpflag=False, ldflag=False, freeendsflag=False, optimalpathflag=True, graphicoptimalpathflag=True,graphicseqalignment=True):
    print("Test: ", test.__name__)
    print("test seq (1) = ", test.seq1)
    print("ref  seq (2) = ", test.seq2)
    dtwcomputer = Dtw(test.seq1, test.seq2)
    if hasattr(test, 'constraints'):
        ct = test.constraints
    else: # default
        ct = "SYMMETRIC"
    if hasattr(test, 'disttype'):
        if test.disttype == "MIXED":
            stg = "MIXED"
            ld = mixed_dist
        elif test.disttype == "ANGULAR":
            stg = "ANGULAR"
            ld = angular_dist
        else:
            stg = "EUCLIDEAN"
            ld = euclidean_dist
        print(stg, " distance used for local distance ...")
    else:  # Default
        print("EUCLIDEAN distance used for local distance ...")
        ld = euclidean_dist
    if hasattr(test, 'freeends'):
        fe = test.freeends
    else:  # Default
        fe = (0, 1)
    if hasattr(test, 'beamsize'):
        bs = test.beamsize
    else:  # Default
        bs = -1
    if hasattr(test, 'delinscost'):
        dc = test.delinscost
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


    ndist, path, length, ndistarray, backpointers = dtwcomputer.run(ldist=ld, constraints=ct, delinscost=dc,
    mixed_type = mt, mixed_spread = ms, mixed_weight = mw,freeends=fe, beamsize=bs)

    dtwcomputer.printresults(cumdistflag, bpflag, ldflag, freeendsflag, optimalpathflag, graphicoptimalpathflag,graphicseqalignment)
