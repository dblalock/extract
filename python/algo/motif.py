#!/usr/env/python

import time
import copy
import numpy as np
from collections import namedtuple

from ..datasets import synthetic as synth
from ..utils import evaluate
from ..utils import sliding_window as window
from ..utils.arrays import zNormalizeRows, zNormalize, isScalar
from ..utils.subseq import distsToRows, optimalAlignment
from ..utils.sequence import isListOrTuple

from dist import distsToRandomVects

from joblib import Memory
memory = Memory('./output', verbose=0)

from numba import jit

# ================================================================
# Data Structures
# ================================================================

_infoFields = [
	'subseq1',
	'subseq2',
	'dist',
	'idx1',
	'idx2',
	'length',
	'fromSeq',
	'Xnorm'
]
MotifInfo = namedtuple('MotifInfo', _infoFields)

# ================================================================
# Functions
# ================================================================

def _normByLength(m, d):
	return d / m


def moenLowerBound(oldLen, newLen, oldDist, normByLength=True):
	# see "Enumeration of Time Series Motifs of All Lengths"
	if oldLen > newLen:
		raise ValueError("MOEN lower bound valid only for increasing lengths!")
	if oldLen == newLen:
		return oldDist

	frac = 1.
	m = int(oldLen)
	while m < int(newLen):
		fltM = float(m) # don't get hosed by integer division

		# the magic 25 is 5^2, where 5 is 5 sigma above the mean when znormed;
		# see "Time Series Join on Subsequence Correlation"
		frac *= 1. / (fltM / (1+m) + fltM / ((1+m)*(1+m)) * 25.)
		m += 1

	if normByLength:
		frac *= oldLen / float(newLen)

	return oldDist * frac


def findMotif(seqs, lengths, lengthNormFunc=None, abandonLengths=True, **kwargs):
	tstart = time.clock()

	if isScalar(lengths):
		lengths = [lengths]
	if not isListOrTuple(seqs):
		seqs = [seqs]

	minDist = np.inf
	maxBitSave = -np.inf

	useMDL = kwargs.get('threshAlgo') == 'mdl' # TODO this is a hack
	mdlAbandon = kwargs.get('mdlAbandonCriterion') # TODO this is also a hack
	# print "findMotif(): threshAlgo, useMDL = ", kwargs.get('threshAlgo'), useMDL

	if not useMDL:
		assert(not lengthNormFunc) # our early abandoning requires _normByLength

	lengthNormFunc = lengthNormFunc or _normByLength

	oldDist = 0.
	oldM = lengths[0]
	bestMotif = None
	for i, m in enumerate(lengths):
		if useMDL:
			motif = findMotifOfLengthFast(seqs[:], m, **kwargs)
			ptr = [-1]
			idx1, idx2 = motif.idx1, motif.idx2
			Xnorm, fromSeq = motif.Xnorm, motif.fromSeq
			s1, s2 = Xnorm[idx1], Xnorm[idx2]
			extractInstanceIdxsOfMotif(s1, s2, Xnorm, m, fromSeq,
				bitSaveContainer=ptr, idx1=idx1, idx2=idx2, **kwargs)
			bitsave = ptr[0]
			if bitsave > maxBitSave:
				print "findMotif(): new best bitsave {} at length {}".format(bitsave, m)
				maxBitSave = bitsave
				bestMotif = copy.copy(motif)
			if bitsave < 0. and maxBitSave > 0. and mdlAbandon == 'allNegative':
				break

		else:
			# early abandoning using MOEN lower bound
			if abandonLengths and i > 0 and oldDist < np.inf:
				# XXX this assumes that we're using znormed euclidean dist
				lowerBound = moenLowerBound(oldM, m, oldDist)
				if lowerBound >= minDist:
					print("early abandoning length {}".format(m))
					continue # don't have to evaluate this length

			# determine whether to compute the exact distance or just determine
			# whether to check for distances < the current best (and return the
			# exact distance if one is better). This is mutually exclusive with
			# abandoning entire lengths, since comparing to the best dist so
			# far can only tell us that the lowest dist is > this dist, not
			# what that lowest dist is, and therefore can't give us a bound
			# on the lowest dist for a slightly larger length.
			dMin = minDist * m
			if abandonLengths:
				dMin = np.inf
			motif = findMotifOfLengthFast(seqs[:], m, dMin=dMin, **kwargs)
			# motif = findMotifOfLengthFast(seqs[:], m, verbose=True, **kwargs)
			print("---- best pair at length {} = {}, {}".format(m, motif.idx1, motif.idx2))

			oldM = m
			oldDist = motif.dist # store raw euclidean dist^2, for MOEN LB

			if lengthNormFunc: # divide by length, sqrt(length), or whatever
				motif = motif._replace(dist=lengthNormFunc(m, motif.dist))

			if motif.dist < minDist:
				minDist = motif.dist
				bestMotif = copy.copy(motif)

	tElapsed = time.clock() - tstart
	print '------------------------ Found best motif in {}s; length = {}'.format(
		tElapsed, bestMotif.length)
	return bestMotif


@jit
def findMotifOfLength(seqs, m):
	"""find the closest non-overlapping pair of subsequences of length m"""
	t0 = time.time()

	origSeqs = seqs[:]
	origM = m

	# flatten Nd input seqs
	origDims = len(seqs[0].shape)
	if origDims > 1:
		m *= np.prod(seqs[0].shape[1:])
		for i, seq in enumerate(seqs):
			seqs[i] = seq.flatten()
		# m *= origDims # TODO don't enforce stepping in only one direction

	print("Searching for motif of length %d (flattened to %d)..." % (origM, m))

	# get a list whose elements are 2D arrays, each row of which is a subseq
	# stride = np.zeros(seqs[0].shape)
	allSubseqs = window.sliding_windows_of_elements(seqs, m, 1)
	allSubseqs = map(zNormalizeRows, allSubseqs)

	# print map(lambda subseqs: subseqs.shape, allSubseqs)
	# return

	minDist = np.inf
	minSeq1Idx, minSubseq1Idx = -1, -1
	minSeq2Idx, minSubseq2Idx = -1, -1

	# for each original sequence
	for seqNum, subseqs in enumerate(allSubseqs):
		print(subseqs.shape)
		print("------------------------")
		# for each subseq in the original seq
		for subseqNum, subseq in enumerate(subseqs):
			# for each seq before this one
			# note that there's no need to also check seqs after this one
			# if we're using a symmetric distance measure (which L2 is)
			for otherSeqNum, otherSubseqs in enumerate(allSubseqs[:seqNum]):
				dists = distsToRows(otherSubseqs, subseq)
				minVal = np.min(dists)
				if minVal < minDist:
					# new closest pair; record which sequence each element
					# of the pair was in and its position within this sequence
					minSeq1Idx, minSubseq1Idx = seqNum, subseqNum
					minSeq2Idx, minSubseq2Idx = otherSeqNum, np.argmin(dists)

			# for each non-overlapping subseq in the current seq
			dists = distsToRows(seqs[seqNum], subseq)
			# disallow overlapping subseqs
			startIdx = max(0, subseqNum - m + 1)
			endIdx = min(len(dists), (subseqNum + m - 1) + 1)
			dists[startIdx:endIdx] = np.inf
			minVal = np.min(dists)
			if minVal < minDist:
				# new closest pair; record which sequence each element
				# of the pair was in and its position within this sequence
				minDist = minVal
				minSeq1Idx, minSubseq1Idx = seqNum, subseqNum
				minSeq2Idx, minSubseq2Idx = seqNum, np.argmin(dists)

	# print map(lambda x: x.shape, allSubseqs)
	# print "final subseq idxs:"
	# print minSeq1Idx, minSubseq1Idx
	# print minSeq2Idx, minSubseq2Idx
	minSeq1Idx, minSubseq1Idx = minSeq1Idx / origDims, minSubseq1Idx / origDims
	minSeq2Idx, minSubseq2Idx = minSeq2Idx / origDims, minSubseq2Idx / origDims
	# print minSeq1Idx, minSubseq1Idx
	# print minSeq2Idx, minSubseq2Idx

	# extract subseqs
	# seq1 = allSubseqs[minSeq1Idx]
	seq1 = origSeqs[minSeq1Idx]
	seq2 = origSeqs[minSeq2Idx]
	subseq1 = seq1[minSubseq1Idx:minSubseq1Idx+origM]
	subseq2 = seq2[minSubseq2Idx:minSubseq2Idx+origM]

	# # plt.plot(seq1)
	# subseq1 = seq1[minSubseq1Idx]
	# # plt.plot(subseq1)
	# seq2 = allSubseqs[minSeq2Idx]
	# # plt.plot(seq2)
	# subseq2 = seq2[minSubseq2Idx]
	# plt.plot(subseq2)
	# plt.show()

	# print m
	# print seq1.shape
	# print seq2.shape
	# print subseq1.shape
	# print subseq2.shape
	print("Found motif with dist %g in %gs" % (minDist, time.time() - t0))
	return subseq1, subseq2, minDist, minSeq1Idx, minSeq2Idx


@jit
def _mkLoop(Xsort, unsortIdxs, fromSeqSort, projDistsSort, subseqLen, dMin=np.inf,
	allFromSameSeq=False):

	# print("got initial dMin {} ({})".format(dMin, dMin*dMin / subseqLen))

	# find best motif--we only have to check stuff earlier in the orderline
	# since L2 dist is symmetric
	start1, start2 = -1, -1
	nSubseqs = Xsort.shape[0]

	# inner loop when everything is from the same seq
	if allFromSameSeq:
		for i in range(nSubseqs):
			origIdx1 = unsortIdxs[i]
			fromSeq1 = fromSeqSort[i]
			j = i - 1
			while j >= 0:
				# ignore overlapping subseqs of the same seq
				origIdx2 = unsortIdxs[j]
				if int(np.abs(origIdx2 - origIdx1)) < subseqLen:
					j -= 1
					continue

				# check lower bounds to try avoiding distance computation; we
				# can abandon early if we reach something in the orderline
				# with too great a distance
				diffs = np.abs(projDistsSort[i] - projDistsSort[j])
				if diffs[0] >= dMin: # ordered by this value
					break
				elif np.any(diffs >= dMin):
					j -= 1
					continue

				# compute full distance and check if new motif
				diff = Xsort[i] - Xsort[j]
				d = np.dot(diff, diff)
				d = np.sqrt(d)
				if d < dMin:
					dMin = d
					start1, start2 = j, i

				j -= 1

		return start1, start2, dMin

	# inner loop when subseqs can be from different seqs
	for i in range(nSubseqs):
		# if i % 100 == 0:
		# 	print("Testing subseq {0}".format(i))
		origIdx1 = unsortIdxs[i]
		fromSeq1 = fromSeqSort[i]
		j = i - 1
		while j >= 0:
			# ignore overlapping subseqs of the same seq
			fromSeq2 = fromSeqSort[j]
			if fromSeq1 == fromSeq2:
				origIdx2 = unsortIdxs[j]
				if int(np.abs(origIdx2 - origIdx1)) < subseqLen:
					j -= 1
					continue

			# check lower bounds to try avoiding distance computation; we
			# can abandon early if we reach something in the orderline
			# with too great a distance
			diffs = np.abs(projDistsSort[i] - projDistsSort[j])
			if diffs[0] >= dMin: # ordered by this value
				break
			elif np.any(diffs >= dMin):
				j -= 1
				continue

			# compute full distance and check if new motif
			diff = Xsort[i] - Xsort[j]
			d = np.dot(diff, diff)
			d = np.sqrt(d)
			if d < dMin:
				dMin = d
				start1, start2 = j, i

			j -= 1

	print("returning dMin {} ({})".format(dMin, dMin*dMin / subseqLen))

	return start1, start2, dMin


# @autojit # can't jit or autojit this for no discernible reason
# def findMotifOfLengthFast(seqs, m, nprojections=10, norm='each', returnNormedSeqs=True):
def findMotifOfLengthFast(seqs, m, nprojections=10, norm='each',
	returnNormedSeqs=False, verbose=False, dMin=-1, **kwargs):
	"""find the closest non-overlapping pair of subsequences of length m
	using the MK algorithm"""
	# TODO split into smaller funcs, remove dup code and asserts, etc

	if dMin <= 0.:
		dMin = np.inf

	t0 = time.clock()

	origM = m

	# X, allSubseqs = window.flattened_subseqs_of_length(seqs, origM)
	# # X = np.asarray(allSubseqs, dtype=np.float).reshape((-1, m)) # -1 = compute it
	# n, m = X.shape
	# Xnorm, X, allSubseqs = window.flattened_subseqs_of_length(seqs, origM, norm=norm)
	Xnorm, X, allSubseqs, fromSeq = window.flattened_subseqs_of_length(seqs, origM,
		norm=norm, return_from_seq=True)
	# X = np.asarray(allSubseqs, dtype=np.float).reshape((-1, m)) # -1 = compute it
	n, m = Xnorm.shape
	if verbose:
		print("Searching for motif of length %d (flattened to %d)..." % (origM, m))

	allFromSameSeq = len(np.unique(fromSeq)) == 1
	if allFromSameSeq and len(Xnorm) <= origM:
		raise ValueError("Length {} too long for time series with {} subsequences!".format(
			origM, len(Xnorm)))

	# Xnorm = X
	# if norm == 'each':
	# 	# normalize each dimension in each subseq
	# 	Xnorm = np.empty(X.shape)
	# 	for i, subseq in enumerate(X):
	# 		s = subseq.reshape((origM, -1))
	# 		s = zNormalizeCols(s, removeZeros=False)
	# 		Xnorm[i] = s.flatten()

	# 		# so cols of the above are definitely normalized
	# 		# right when we plot them
	# 		# plt.figure()
	# 		# plt.plot(s)
	# 		# # plt.savefig('figs/motif/shapes/shape-subseqs_%d.png' % i)
	# 		# plt.savefig('figs/motif/msrc/subseqs_%d.png' % i)
	# 		# plt.close()

	# elif norm == 'all':
	# 	# normalize concatenation of all dims
	# 	Xnorm = zNormalizeRows(X)

	# figure out seq boundaries so we can check for overlapping matches
	# print "Computing seq boundaries..."
	# seqLens = np.array(map(lambda subseqs: subseqs.shape[0], allSubseqs))
	# # seqLens = np.array((s.shape[0] for s in allSubseqs))
	# startIdxs = np.r_[0, np.cumsum(seqLens)[:-1]]
	# endIdxs = np.r_[startIdxs[1:], n]
	# fromSeq = np.zeros(n)
	# for i in range(len(startIdxs)):
	# 	startIdx, endIdx = startIdxs[i], endIdxs[i]
	# 	fromSeq[startIdx:endIdx] = i

	# # project each subsequence onto k random unit vectors--we'll use
	# # the resulting distances
	# # print "Computing random projections..."
	# # np.random.seed() # undo any previous seeding for debugging # TODO remove
	# projVects = np.random.randn(m, nprojections)
	# # projVectIdxs = np.random.choice(np.arange(n), nprojections)
	# # projVects = X[projVectIdxs].T.copy()
	# projVects = zNormalizeCols(projVects)

	# # dotProds = np.dot(X, projVects)
	# # projDists = 2 * (m - dotProds) # ||a||^2 + ||b||^2 - 2a.b = m + m - 2a.b
	# projDists = np.empty((n, nprojections))
	# for i, row in enumerate(Xnorm):
	# 	for j, col in enumerate(projVects.T):
	# 		diff = row - col
	# 		projDists[i, j] = np.dot(diff, diff)
	# projDists = np.sqrt(projDists) # triangle inequality holds for norm, not norm^2

	# print "Validating projection distances..."
	# for i, row in enumerate(X): # so this always passes...
	# 	for j, col in enumerate(projVects.T):
	# 		diff = row - col
	# 		d = np.sum(diff*diff)
	# 		d = np.sqrt(d)
	# 		if np.abs(d - projDists[i, j]) > 1e-10:
	# 			print d, projDists[i,j]
	# 			assert(0)

	# print "Validating triangle inequality..."
	# for i, row1 in enumerate(X): # always passes now
	# 	if np.max(row1) == 0:
	# 		continue
	# 	dists1 = projDists[i]
	# 	for j, row2 in enumerate(X):
	# 		if np.max(row2) == 0:
	# 			continue
	# 		dists2 = projDists[j]
	# 		# d = 2*(m - np.dot(row1, row2))
	# 		d = np.sum((row1-row2)*(row1-row2))
	# 		assert(d >= 0)
	# 		d = np.sqrt(d)
	# 		diffs = np.abs(dists1 - dists2)
	# 		if np.any(d < diffs):
	# 			print("lower bound failed at %d, %d!" % (i, j))
	# 			print d, diffs
	# 			plt.plot(row1)
	# 			plt.plot(row2)
	# 			plt.show()
	# 			assert(0)

	# # figure out std deviations of dists to different projections and
	# # sort projections by decreasing std
	# # print "Sorting random projections..."
	# projStds = np.std(projDists, axis=0)
	# projSortIdxs = np.argsort(projStds)[::-1]
	# projDists = projDists[:, projSortIdxs]

	projDists, projVects = distsToRandomVects(Xnorm, nprojections)

	# order seqs by 1st projection
	# print "Sorting subsequences..."
	sortIdxs = np.argsort(projDists[:, 0])
	unsortIdxs = np.arange(n)[sortIdxs] # unsortIdxs: projected idx -> orig idx
	Xsort = Xnorm[sortIdxs, :]
	fromSeqSort = fromSeq[sortIdxs]
	projDistsSort = projDists[sortIdxs, :]

	# for i, row in enumerate(Xsort): # this always passes too...
	# 	for j, col in enumerate(projVects.T):
	# 		diff = row - col
	# 		d = np.sum(diff*diff)
	# 		if np.abs(d - projDistsSort[i, j]) > 1e-10:
	# 			print d, projDistsSort[i,j]
	# 			assert(0)

	# print "Searching for motif..."
	start1, start2, dist = _mkLoop(Xsort, unsortIdxs, fromSeqSort,
		projDistsSort, origM, np.sqrt(dMin), allFromSameSeq)
	dist = dist * dist # mk uses using actual L2, not L2^2

	if verbose:
		print("Best motif: %d, %d" % (start1, start2))
		print("Best motif (orig idxs): %d, %d" % (unsortIdxs[start1], unsortIdxs[start2]))
		print("Found motif with dist %g in %gs" % (dist, time.clock() - t0))

	if start1 < 0 or start2 < 0: # _mkLoop didn't find anything better than dMin
		return MotifInfo(None, None, np.inf, -1, -1, origM, fromSeq, Xnorm)

	# return seqs corresponding to best motif
	origIdx1 = unsortIdxs[start1]
	origIdx2 = unsortIdxs[start2]
	if returnNormedSeqs:
		s1 = Xsort[start1].reshape((origM, -1))
		s2 = Xsort[start2].reshape((origM, -1))
	else:
		s1 = X[origIdx1].reshape((origM, -1))
		s2 = X[origIdx2].reshape((origM, -1))

	# return instances in order
	if origIdx2 < origIdx1:
		origIdx1, origIdx2 = origIdx2, origIdx1
		s1, s2 = s2, s1

	return MotifInfo(s1, s2, dist, origIdx1, origIdx2, origM, fromSeq, Xnorm)


def nonOverlappingMinima(minDists, m, fromSeq=None):
	"""
	Returns the indices i such that (minDist[i] <= minDists[j] or
	fromSeq[i] != fromSeq[j]) for all j in [i, i + m - 1]. If fromSeq
	is not provided, the latter test always fails (i.e., both dists are
	assumed to be from the same sequence, and thus able to overlap).

	>>> nonOverlappingMinima([1,3,2], 1)
	array([0, 1, 2])
	>>> nonOverlappingMinima([1,3,2,4], 3)
	array([0, 3])
	>>> nonOverlappingMinima([2,3,1,4], 3)
	array([2])
	>>> nonOverlappingMinima([1,3,2,4,5,6], 3)
	array([0, 3])
	>>> nonOverlappingMinima([1,3,2,4,3,2], 3)
	array([0, 5])
	"""
	minDists = np.asarray(minDists)

	if fromSeq is not None:
		assert(len(fromSeq) == len(minDists))

	idxs = []
	candidateIdx = 0
	candidateDist = minDists[0]
	for idx in range(1, len(minDists)):
		dist = minDists[idx]
		# overlaps if within m of each other and from the same original seq
		overlaps = np.abs(idx - candidateIdx) < m
		if fromSeq is not None:
			overlaps = overlaps and (fromSeq[candidateIdx] == fromSeq[idx])

		if overlaps: # overlaps
			if dist < candidateDist:
				# replace current candidate
				candidateIdx = idx
				candidateDist = dist
		else:
			# no overlap, so safe to flush candidate
			idxs.append(candidateIdx)

			# set this point as new candidate
			candidateIdx = idx
			candidateDist = dist

	# invariant: candidate idx has not been added to idxs;
	# as a result, the final candidate idx hasn't been added; since
	# no better overlaps with this candidate are possible, we should
	# add it when the loop terminates
	idxs.append(candidateIdx)

	# print "nonOverlappingMinima(): minDists shape = ", minDists.shape
	# print "nonOverlappingMinima(): m = ", m
	# print "nonoverlapping minima idxs:", idxs

	return np.array(idxs)


def nonOverlappingMaxima(v, *args, **kwargs):
	"""see nonOverlappingMinima()"""
	return nonOverlappingMinima(-v, *args, **kwargs)


def nonOverlappingGreedy(minDists, m, thresh, fromSeq=None):
	"""Greedily iterates thru minDists looking for idxs i such that
	minDists[i] <= thresh; when such an index is found, the search jumps
	ahead to i+m so that overlaps are not considered.

	If fromSeq is provided, the jump is instead until either i has advanced m
	points, or the value of the fromSeq[i] changes. Consequently, all elements
	from a given sequence (as reflected by fromSeq) must be contiguous.
	"""

	minDists = np.asarray(minDists)
	if fromSeq is not None:
		assert(len(fromSeq) == len(minDists))

	idxs = []
	i = 0
	while i < len(minDists):
		if minDists[i] <= thresh:
			idxs.append(i)
			# everything from the same sequence--skip ahead by m
			if fromSeq is None:
				i += m
			# dists from different sequences--skip ahead up to m
			# positions, or until the start of a new sequence
			else:
				end = i + m
				while i < end:
					i += 1
					if fromSeq[i] != fromSeq[i-1]:
						break
		else:
			i += 1

	return np.array(idxs)


def findCornerMinnen(dists):
	"""Using technique in Minnen et al, Improving Activity Discovery
	with Automatic Neighborhood Estimation

	Returns cutoffDist in sorted dists (ascending)
	"""
	# get lowest 10% of dists and sort them
	dists = np.sort(dists)
	dists = dists[:len(dists)/10]

	# compute 1st derivative
	deriv = dists[1:] - dists[:-1]

	# weighted vote for best dist
	deriv /= np.sum(deriv)
	idxs = np.arange(len(deriv)) + .5  		# halfway between i, i+1
	cutoffIdx = int(np.sum(deriv * idxs)) 	# weighted sum of idxs
	cutoffDist = (dists[cutoffIdx] + dists[cutoffIdx+1]) / 2

	return cutoffDist


def findCornerMaxSep(dists):
	dists = np.sort(dists)
	deriv = dists[1:] - dists[:-1]
	maxSepIdx = np.argmax(deriv)
	return (dists[maxSepIdx] + dists[maxSepIdx+1]) / 2


def findThreshold(dists, algorithm=None):
	if algorithm is None:
		return np.inf
	elif algorithm.lower() == "minnen":
		return findCornerMinnen(dists)
	elif algorithm.lower() == "maxsep":
		return findCornerMaxSep(dists)
	elif isScalar(algorithm):
		return algorithm
	else:
		raise ValueError("Received invalid algorithm name %s" % algorithm)


def entropy(A, axis=None):
	"""Computes the Shannon entropy of the elements of A. Assumes A is
	an array-like of nonnegative ints whose max value is approximately
	the number of unique values present.

	>>> a = [0, 1]
	>>> entropy(a)
	1.0
	>>> A = np.c_[a, a]
	>>> entropy(A)
	1.0
	>>> A 					# doctest: +NORMALIZE_WHITESPACE
	array([[0, 0], [1, 1]])
	>>> entropy(A, axis=0) 	# doctest: +NORMALIZE_WHITESPACE
	array([ 1., 1.])
	>>> entropy(A, axis=1) 	# doctest: +NORMALIZE_WHITESPACE
	array([[ 0.], [ 0.]])
	>>> entropy([0, 0, 0])
	0.0
	>>> entropy([])
	0.0
	>>> entropy([5])
	0.0
	>>> entropy([-2, -1])
	1.0
	"""
	if A is None or len(A) < 2:
		return 0.

	A = np.asarray(A).astype(np.int)
	A = np.copy(A) - np.min(A) # ensure all nonnegative

	if axis is None:
		A = A.flatten()
		counts = np.bincount(A) # needs small, non-negative ints
		counts = counts[counts > 0]
		if len(counts) == 1:
			return 0. # avoid returning -0.0 to prevent weird doctests
		probs = counts / float(A.size)
		return -np.sum(probs * np.log2(probs))
	elif axis == 0:
		entropies = map(lambda col: entropy(col), A.T)
		return np.array(entropies)
	elif axis == 1:
		entropies = map(lambda row: entropy(row), A)
		return np.array(entropies).reshape((-1, 1))
	else:
		raise ValueError("unsupported axis: {}".format(axis))


@jit # uncomment for doctests to run
def minValueExcludingIdxs(x, disallowedIdxs):
	"""
	Find the minimum value x[i] such that i is not within disallowedIdxs;
	note that disallowedIdxs must be sorted in ascending order.

	Returns
	-------
	val, idx -- the minimum value and the first index at which it occurs

	>>> x = [2, 0, 3, 1]
	>>> minValueExcludingIdxs(x, [])
	(0, 1)
	>>> minValueExcludingIdxs(x, [1])
	(1, 3)
	>>> minValueExcludingIdxs(x, range(len(x)))
	(inf, -1)
	>>> minValueExcludingIdxs(x, [0, 1, 3])
	(3, 2)
	"""
	if not len(disallowedIdxs):
		minIdx = np.argmin(x)
		return x[minIdx], minIdx

	minIdx = -1
	minVal = np.inf
	j = 0
	numDisallowed = len(disallowedIdxs)
	for i, val in enumerate(x):
		jInbounds = j < numDisallowed
		while jInbounds and i > disallowedIdxs[j]:
			j += 1
		if jInbounds and i == disallowedIdxs[j]:
			j += 1
			continue

		if val < minVal:
			minVal = val
			minIdx = i

	return minVal, minIdx


def extractInstanceIdxsOfMotifMDL(instance1, instance2, Xnorm, m,
	fromSeq=None, mdlBits=8, idx1=-1, idx2=-1, bitSaveContainer=None,
	mdlAbandonCriterion=None, searchEachTime=False, shiftStep=0., **sink):
	"""dedicated func to compute motif instances using Minimum
	Description Length

	Notes: m is only used as a minimum spacing between returned instances
	"""

	# # figure out where these instances came from if this info isn't supplied
	# if idx1 < 0:
	# 	dists1 = distsToRows(Xnorm, instance1)
	# 	idx1 = np.argmin(dists1)
	# if idx2 < 0:
	# 	dists2 = distsToRows(Xnorm, instance2)
	# 	idx2 = np.argmin(dists2)

	# figure out what data corresponds to the best instance indices if the
	# data itself isn't supplied
	if idx1 >= 0 and instance1 is None:
		instance1 = Xnorm[idx1]
	if idx2 >= 0 and instance2 is None:
		instance2 = Xnorm[idx2]

	# sanity check the best instances / idxs
	assert(np.array_equal(instance1, Xnorm[idx1]))
	assert(np.array_equal(instance2, Xnorm[idx2]))

	# if we were told to try the regions around the motif, do so
	if shiftStep > 0.:
		if idx1 < 0 or idx2 < 0:
			raise ValueError("Seeding motif extraction requires specifying"
				"original motif pair indices (by setting idx1 and idx2)!")

		# generate "seed" motifs by taking the indices near the original motif
		numShifts = int(1. / shiftStep)
		stepLen = shiftStep * m # convert from fraction to length
		origPair = np.array([idx1, idx2], dtype=np.int)
		pairs = []
		for shft in range(2 * numShifts + 1):
			stepsForward = shft - numShifts # ranges between +/- numShifts
			offset = int(stepsForward * stepLen)
			pair = origPair + offset
			if pair[0] < 0:
				continue
			if pair[1] >= len(Xnorm):
				continue
			pairs.append(pair)

		# try each region around the original motif and return the best
		bestBitsave = -np.inf
		bestIdxs = None
		for pair in pairs:
			bitSaveContainer2 = [-np.inf]
			idxs = extractInstanceIdxsOfMotifMDL(None, None, Xnorm, m,
				fromSeq=fromSeq, mdlBits=mdlBits, idx1=pair[0], idx2=pair[1],
				bitSaveContainer=bitSaveContainer2)
			thisBitsave = bitSaveContainer2[0]

			if thisBitsave > bestBitsave:
				bestBitsave = thisBitsave
				bestIdxs = idxs[:]

		if bitSaveContainer is not None:
			bitSaveContainer[0] = bestBitsave
		return bestIdxs

	# print "extractInstanceIdxsOfMotifMDL(): idx1, idx2 = ", idx1, idx2
	# print "extractInstanceIdxsOfMotifMDL(): mdlBits = ", mdlBits
	# print "extractInstanceIdxsOfMotifMDL(): m = ", m

	instanceIdxs = [idx1, idx2]

	# store which indices overlap with the original pair
	disallowedIdxs = set()
	# print "extractInstanceIdxsOfMotifMDL(): initial disallowedIdxs = ", disallowedIdxs
	disallowedIdxs |= set(range(idx1-m+1, idx1+m))
	disallowedIdxs |= set(range(idx2-m+1, idx2+m))
	# print "extractInstanceIdxsOfMotifMDL(): pair disallowedIdxs = ", disallowedIdxs

	# compute quantized subsequences
	numLevels = int(2**mdlBits)
	mins = np.min(Xnorm, axis=1).reshape((-1, 1))
	maxs = np.max(Xnorm, axis=1).reshape((-1, 1))
	ranges = (maxs - mins)
	Xquant = (Xnorm - mins) / ranges * (numLevels - 1) # 8 bits -> {0..255}
	Xquant = Xquant.astype(np.int)

	# compute bitsave resulting from just the original pair
	row1 = Xquant[idx1]
	row2 = Xquant[idx2]
	centroidSums = row1 + row2
	centroidQuant = centroidSums // 2
	hypothesisEnt = entropy(centroidQuant)
	origEnt = entropy(row1) + entropy(row2)
	newEnt = entropy(row1 - centroidQuant) + entropy(row2 - centroidQuant)
	codingSave = origEnt - newEnt
	bitsave = codingSave - hypothesisEnt

	if bitsave <= 0.:
		# print("-- length {}, bitsave from best pair = {}. Wat?".format(Xquant.shape[1], bitsave))
		# print("-- origEnt, newEnt, hypoEnt = {}, {}, {}".format(origEnt, newEnt,
			# hypothesisEnt))
		if mdlAbandonCriterion == 'bestPairNegative':
			if bitSaveContainer is not None:
				bitSaveContainer[0] = bitsave * Xnorm.shape[1]
			return instanceIdxs
		# assert(newEnt < origEnt) # TODO remove, since no guarantee this is true
		# TODO early abandon here even though it makes the results far worse,
		# because that's clearly what they do in the paper

	# 	import matplotlib.pyplot as plt

	# 	# hmm. centroids look right...
	# 	plt.figure()
	# 	plt.plot(row1)
	# 	plt.plot(row2)
	# 	plt.plot(centroidQuant)

	# 	# so do differences...
	# 	plt.figure()
	# 	plt.plot(row1 - centroidQuant)
	# 	plt.plot(row2 - centroidQuant)

	# 	plt.show()

	# 	# huh...so this actually is right--it's just that encoding the
	# 	# hypothesis is often more expensive than the reduced encoding cost
	# 	# of the differences, even for a *very* close pair of matches; this
	# 	# sort of makes sense in that small differences do not necessarily
	# 	# mean a lower-entropy collection of particular difference values.

	# 	assert(bitsave > 0.)

	# determine the order in which to consider possible instances;
	# use dists based on znormalized comparisons, since this is what everyone
	# was using if they were using MK (or its derivatives) as the motif
	# finding subroutine. It doesn't matter that these aren't the quantized
	# subsequences, since znormalization will undo the offset and scaling anyway
	centroid = (instance1 + instance2) / 2.
	centroid = zNormalize(centroid)
	dists = distsToRows(Xnorm, centroid)

	# add subseqs similar to the centroid until the bitsave would decrease;
	# no paper has actually specified what they do to pick the best locations,
	# (unless it's just to greedily take the best non-overlapping ones) so
	# I'm going to use the greedy approach

	if searchEachTime: # compute nearest neighbor after each centroid update
		# TODO factor out dup code; this is terrible
		while True:
			disallowedIdxsSorted = np.sort(list(disallowedIdxs))
			_, idx = minValueExcludingIdxs(dists, disallowedIdxsSorted)
			if idx < 0: # no more possible locations in ts
				break
			subseq = Xquant[idx]
			currentNumIdxs = len(instanceIdxs)

			# compute original entropy of this instance along with current ones
			newOrigEnt = origEnt + entropy(subseq)

			# compute centroid when this instance is added
			newCentroidSums = centroidSums + subseq
			newCentroid = (newCentroidSums / (currentNumIdxs + 1)).astype(np.int)
			# newCentroid = np.mean(Xquant[newInstanceIdxs], axis=0).astype(np.int)

			# compute coded entropy when this instance is added
			newInstanceIdxs = instanceIdxs[:]
			newInstanceIdxs.append(idx)
			# diffs = Xquant[instanceIdxs] - newCentroid # works better, but nonsensical
			diffs = Xquant[newInstanceIdxs] - newCentroid
			newCodedEnt = np.sum(entropy(diffs, axis=1))

			# compute total bitsave if this instance is added
			newCodingSave = newOrigEnt - newCodedEnt
			newHypothesisEnt = entropy(newCentroid)
			newBitsave = newCodingSave - newHypothesisEnt

			# add this idx to the set of instances if it yields better bitsave
			if newBitsave > bitsave:
				# print "bitsave {} > {} for idx {}".format(newBitsave, bitsave, idx)
				# print "bitsave {} > {} for idxs {}".format(newBitsave, bitsave, newInstanceIdxs)
				bitsave = newBitsave
				origEnt = newOrigEnt
				centroidSums = newCentroidSums
				instanceIdxs = newInstanceIdxs
				disallowedIdxs |= set(range(idx-m+1, idx+m))

				# update centroid based on examples so far; not much effect
				centroid = centroidSums / float(len(instanceIdxs))
				centroid = zNormalize(centroid)
				dists = distsToRows(Xnorm, centroid)

			else: # early abandon if adding kth neighbor decreases bitsave
				break

	else: # just use nearest neighbors for original centroid
		sortIdxs = np.argsort(dists)
		for idx in sortIdxs:
			if idx in disallowedIdxs: # overlaps with an existing instance
				continue
			subseq = Xquant[idx]
			currentNumIdxs = len(instanceIdxs)

			# compute original entropy of this instance along with current ones
			newOrigEnt = origEnt + entropy(subseq)

			# compute centroid when this instance is added
			newCentroidSums = centroidSums + subseq
			newCentroid = (newCentroidSums / (currentNumIdxs + 1)).astype(np.int)
			# newCentroid = np.mean(Xquant[newInstanceIdxs], axis=0).astype(np.int)

			# compute coded entropy when this instance is added
			newInstanceIdxs = instanceIdxs[:]
			newInstanceIdxs.append(idx)
			# diffs = Xquant[instanceIdxs] - newCentroid # works better, but nonsensical
			diffs = Xquant[newInstanceIdxs] - newCentroid
			newCodedEnt = np.sum(entropy(diffs, axis=1))

			# compute total bitsave if this instance is added
			newCodingSave = newOrigEnt - newCodedEnt
			newHypothesisEnt = entropy(newCentroid)
			newBitsave = newCodingSave - newHypothesisEnt

			# add this idx to the set of instances if it yields better bitsave
			if newBitsave > bitsave:
				# print "bitsave {} > {} for idx {}".format(newBitsave, bitsave, idx)
				# print "bitsave {} > {} for idxs {}".format(newBitsave, bitsave, newInstanceIdxs)
				bitsave = newBitsave
				origEnt = newOrigEnt
				# centroidQuant = newCentroid
				centroidSums = newCentroidSums
				instanceIdxs = newInstanceIdxs
				disallowedIdxs |= set(range(idx-m+1, idx+m))
			else: # early abandon if adding kth neighbor decreases bitsave
				break

	if bitSaveContainer is not None:
		bitSaveContainer[0] = bitsave * Xnorm.shape[1]

	return instanceIdxs


def extractInstanceIdxsOfMotif(instance1, instance2, Xnorm, m, fromSeq=None,
	linkage='max', threshAlgo='minnen', threshOnMinima=False, addData=None,
	addDataFractionOfLength=1.0, mdlBits=8, maxOverlapFraction=0., **kwargs):
	"""
	Given the closest pair, return all instances of the motif
	defined by this pair.

	instance{1,2}: closest pair of (normalized) subsequences
	Xnorm: mat whose rows are appropriately normalized subseqs
	m: length of instances in original space; used to determine overlaps
		in Xnorm
	fromSeq: array such that fromSeq[i] is the number of the original sequence
		from which Xnorm[i] was extracted (used to remove overlaps)

	linkage: {'max', 'min', 'avg'}
		used to determine distances to motif pair
	threshAlgo: {'avg_sep', 'max_sep', 'mdl'}
		-The first is Minnen's algorithm in Minnen et al., "Improving
		Activity Discovery with Automatic Neighborhood Estimation".
		-The second is what unsupervised shapelets use--ie, it just takes the
		threshold to be halfway between the two distances with the largest
		gap between them.
		-The third picks instances to minimize the description
		length of the data, when quantized using mdlBits bits.
	threshOnMinima: bool
		whether threshAlgo should be run on all distances, or only
		the non-overlapping relative minima
	mdlBits: int
		the number of bits to use to quantize the data when selecting
		instances using minimum description length. See, e.g., "Efficient
		Proper Length Time Series Motif Discovery".
	maxOverlapFraction: [0., 1.], default 0.
		allows overlap between instances; 0. is no overlap, 1. is any amount
		of overlap (except the exact same (start, end) positions twice)
	"""

	# print ("reducing m from {} to {}".format(m, int(m*(1. - maxOverlapFraction))))
	m = int(m*(1. - float(maxOverlapFraction))) # allow the specified amount of overlap

	if threshAlgo == 'mdl':
		return extractInstanceIdxsOfMotifMDL(instance1, instance2, Xnorm, m,
			fromSeq=fromSeq, mdlBits=mdlBits, **kwargs)

	dists1 = distsToRows(Xnorm, instance1)
	dists2 = distsToRows(Xnorm, instance2)

	if linkage == 'max':
		dists = np.maximum(dists1, dists2)
	elif linkage == 'min':
		dists = np.minimum(dists1, dists2)
	elif linkage == 'avg' or linkage == 'mean':
		dists = (dists1 + dists2) / 2.
	else:
		raise ValueError("Received invalid linkage %s" % linkage)

	if addData or addDataFractionOfLength:
		if not addData:
			addData = 'gauss'
		shape = list(Xnorm.shape)
		shape[0] = int(shape[0] * addDataFractionOfLength)
		if addData == "gauss":
			data = synth.randconst(shape)
		elif addData == "randwalk":
			data = synth.randwalk(shape)
		elif addData == "freqMag":
			data = synth.randWithFreqMagMatching(Xnorm, shape)
		else:
			raise ValueError("Received invalid addData algorithm %s" % addData)

		data = zNormalizeRows(data)
		noiseDists = distsToRows(data, instance1)

		dists = np.r_[dists, noiseDists]

	# length normalize so we can get cosSim; note that this works whether
	# we norm each dimension or all dimensions (well, aside from 0 sections
	# of signal, which we count as identical here)
	dists /= Xnorm.shape[1]

	similarities = 1. - np.sqrt(dists) / 2. # map znormed ED^2 to cosSim

	# XXX optimal alignment doesn't know about fromSeq and so will be
	# wrong if we received multiple seqs
	# TODO pass in subseqs for each fromSeq value and combine results
	# if we ever want to search through multiple seqs at once
	bestEndIdxs = optimalAlignment(similarities, m, scoresForEndIdxs=False)
	bestDists = dists[bestEndIdxs]

	# compute cutoff distance for being considered a motif instance using
	# either the distances to relative minima or all subsequences
	if threshOnMinima:
		cutoffDist = findThreshold(bestDists, threshAlgo)
	else:
		cutoffDist = findThreshold(dists, threshAlgo)

	if not np.isinf(cutoffDist): # didn't want a cutoff
		cutoffSim = 1. - np.sqrt(cutoffDist) / 2.
		similarities -= cutoffSim

	# if addData:
	# 	print "avg non-noise similarity: ", np.mean(similarities[:len(Xnorm)])
	# 	print "avg noise similarity: ", np.mean(similarities[len(Xnorm):])

	# now that we've computed the cutoff, remove similarities from noise
	# for our final answer
	similarities = similarities[:len(Xnorm)]

	return optimalAlignment(similarities, m, scoresForEndIdxs=False)


def findAllMotifInstances(seqs, lengths, **kwargs):
	if isListOrTuple(seqs):
		raise ValueError("can't extract instaces from multiple sequences (yet)!")

	motif = findMotif([seqs], lengths, **kwargs)
	m = motif.length
	idx1, idx2 = motif.idx1, motif.idx2
	Xnorm, fromSeq = motif.Xnorm, motif.fromSeq
	s1, s2 = Xnorm[idx1], Xnorm[idx2]
	# ^ can't just use instances findMotif returns cuz might not be normalized

	# m = Xnorm.shape[1] / seqs[0].shape[1] # flattened len / nDims
	instanceIdxs = extractInstanceIdxsOfMotif(s1, s2, Xnorm, m, fromSeq,
		idx1=idx1, idx2=idx2, **kwargs)
	instances = Xnorm[instanceIdxs]
	return instanceIdxs, instances, motif


def findMotifPatternInstances(seqs, lengths, **kwargs):
	motif = findMotif(seqs, lengths, **kwargs)
	return patternInstancesFromMotif(motif)


# TODO replace above func with this one; PatternInstances
# make more sense to return
def findAllMotifPatternInstances(seqs, lengths, **kwargs):
	instancesTuple = findAllMotifInstances(seqs, lengths, **kwargs)
	return patternInstancesFromAllMotifInstances(*instancesTuple)


def patternInstancesFromMotif(motif):
	p1 = evaluate.createPatternInstance(fromSeq=motif.fromSeq[motif.idx1], startIdx=motif.idx1,
		endIdx=motif.idx1 + motif.length, data=motif.subseq1)
	p2 = evaluate.createPatternInstance(fromSeq=motif.fromSeq[motif.idx2], startIdx=motif.idx2,
		endIdx=motif.idx2 + motif.length, data=motif.subseq2)
	return (p1, p2)


def patternInstancesFromAllMotifInstances(instanceIdxs, instances, motif):
	# construct PatternInstance objs from the above; we only need to
	# populate the start and end idxs since we're assuming everything is
	# from the same class and sequence for now
	# TODO don't assume everything is from the same sequence

	return map(lambda idx, instance: evaluate.createPatternInstance(
		startIdx=idx, endIdx=idx+motif.length, data=instance),
		instanceIdxs, instances)


def evaluateMotifPair(seqs, lengths, trueSeqs, **kwargs):
	"""returns (precision, recall, f1 score) for closest-pair motif extracted
	from seqs, pretending that there were only two instances to be found"""
	motif = findMotif(seqs, lengths, **kwargs)
	reportedSeqs = patternInstancesFromMotif(motif)

	# tell it there were only two true seqs to be found (since we're only
	# reporting the top pair), even though we're letting it search through
	# all of the true seqs to find matches
	return evaluate.scoreSubseqs(reportedSeqs, trueSeqs, spoofNumTrue=2, **kwargs)


def evaluateMotifInstances(seqs, lengths, trueSeqs, **kwargs):
	reportedSeqs = findAllMotifPatternInstances(seqs, lengths, **kwargs)
	return evaluate.scoreSubseqs(reportedSeqs, trueSeqs, **kwargs)


if __name__ == '__main__':
	import doctest
	doctest.testmod()

	# x = np.cumsum(np.random.randn(400, 2), axis=0)
	# y = np.arange(20)
	# # y = np.c_[y, y[::-1]]
	# y = np.c_[y, 10 + 10*np.sin(y / 3.)]

	# # plant the motif
	# x[10:10+len(y),:] = y
	# x[200:200+len(y),:] = y

	# d1 = subseqDists(x, y)

	# spitting out garbage--fix if we care
	# d2 = subseqDistsFFT(x, y)
	# print d1, d2s
	# assert(np.array_equal(d1, d2))

	# seems to be working for d1
	# import matplotlib.pyplot as plt
	# plt.plot(x)
	# plt.plot(y)
	# plt.plot(d1)
	# # plt.plot(d2)
	# plt.show()

	# import matplotlib.pyplot as plt
	# motif = findMotif([x], len(y))
	# # print motif[0].shape
	# # print motif[1].shape
	# plt.plot(motif[0])
	# plt.plot(motif[1])
	# plt.show()

