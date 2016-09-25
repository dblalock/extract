#!/usr/env/python

import time
import numpy as np
# import matplotlib.pyplot as plt
from scipy import signal as sig

from numba import jit

# from ..datasets import synthetic as synth
# from ..datasets import read_msrc as msrc
from ..utils import arrays as ar
from ..utils import sliding_window as window
from ..utils import subseq as sub
# from ..viz import viz_utils as viz

# import representation as rep
# from motif import distsToRandomVects
import motif

# from ff2 import localMaxFilterSimMat, filterSimMat
from ff3 import maxSubarray
# from ff5 import embedExamples
from ff6 import vectorizeWindowLocs


@jit
def _mkSearch(Xnorm, Qnorm, distsX, distsQ, fromSeqX, fromSeqQ, minGap):

	n_x = Xnorm.shape[0]

	sortIdxs_x = np.argsort(distsX[:, 0])
	unsortIdxs_x = np.arange(n_x)[sortIdxs_x] # unsortIdxs: projected idx -> orig idx
	Xsort = Xnorm[sortIdxs_x, :]
	fromSeq_x_sort = fromSeqX[sortIdxs_x]
	distsX_sort = distsX[sortIdxs_x, :]

	orderlineDists = distsX_sort[:, 0]

	Xsort = np.ascontiguousarray(Xsort)
	distsX_sort = np.ascontiguousarray(distsX_sort)
	orderlineDists = np.ascontiguousarray(orderlineDists)

	dMin = np.inf
	start1, start2 = -1, -1
	for i, q in enumerate(Qnorm):
		dists_q = distsQ[i]
		dist_q_orderline = dists_q[0]
		fromSeq1 = fromSeqQ[i]
		origIdx1 = i # XXX: assumes Qnorm is just a transformation of Xnorm
		orderlineIdx = np.searchsorted(orderlineDists, dist_q_orderline)
		j = orderlineIdx
		while j >= 0:
			# ignore overlapping subseqs from the same seq
			fromSeq2 = fromSeq_x_sort[j]
			if fromSeq1 == fromSeq2:
				origIdx2 = unsortIdxs_x[j]
				if int(np.abs(origIdx1 - origIdx2)) < minGap:
					j -= 1
					continue

			# check lower bounds to try avoiding distance computation; we
			# can abandon early if we reach something in the orderline
			# with too great a distance
			diffs = np.abs(distsX_sort[j] - dists_q)
			if diffs[0] >= dMin: # ordered by this value
				break
			elif np.any(diffs >= dMin):
				j -= 1
				continue

			# compute full distance and check if new motif
			diff = Qnorm[i] - Xsort[j]
			d = np.sqrt(np.dot(diff, diff))
			if d < dMin:
				dMin = d
				start1, start2 = i, j

			j -= 1

		j = orderlineIdx + 1
		while j < n_x: # same as above, but going forwards in the array

			fromSeq2 = fromSeq_x_sort[j]
			if fromSeq1 == fromSeq2:
				origIdx2 = unsortIdxs_x[j]
				if int(np.abs(origIdx1 - origIdx2)) < minGap:
					j += 1
					continue
			diffs = np.abs(distsX_sort[j] - dists_q)
			if diffs[0] >= dMin: # ordered by this value
				break
			elif np.any(diffs >= dMin):
				j += 1
				continue
			diff = Qnorm[i] - Xsort[j]
			d = np.sqrt(np.dot(diff, diff))
			if d < dMin:
				dMin = d
				start1, start2 = i, j
			j += 1

	return start1, unsortIdxs_x[start2], dMin


def closestPairFF(seqs, seqsBlur, windowLen, minSpacing=-1, norm=None,
	verbose=False, **kwargs):

	# SELF: seqs and seqsBlur need to have contiguous rows or this will
	# probably run extremely slowly

	# t0 = time.time()
	if minSpacing < 0 or not minSpacing:
		minSpacing = windowLen // 2

	norm = norm.lower()
	dataNorm, queryNorm = norm, norm
	if norm == 'mips':
		dataNorm, queryNorm = 'mips_data', 'mips_query'

	# all sliding window positions, normalized however specified
	Xnorm, _, _, fromSeq = window.flattened_subseqs_of_length(seqs, windowLen,
		norm=dataNorm, return_from_seq=True)
	Qnorm, _, _ = window.flattened_subseqs_of_length(seqsBlur, windowLen, norm=queryNorm)

	# compute distances to reference vectors
	distsX, refVects = motif.distsToRandomVects(Xnorm, **kwargs)
	distsQ, = motif.referenceDists(Qnorm, refVects)

	# find closest pair
	start1, start2, dMin = _mkSearch(Xnorm, Qnorm, distsX, distsQ,
		fromSeq, fromSeq, minSpacing)

	if start2 < start1: # ensure ascending order
		start1, start2 = start2, start1

	return start1, start2


def learnFF(X, Xblur, Lmin, Lmax, length):
	"""main algorithm"""

	# ------------------------ derived stats
	kMax = int(X.shape[1] / Lmin + .5)
	windowLen = Lmax - length + 1

	p0 = np.mean(X) # fraction of entries that are 1 (roughly)
	# p0 = np.mean(X > 0.) # fraction of entries that are 1 # TODO try this
	# p0 = 2 * np.mean(X) # lambda for l0 reg based on features being bernoulli at 2 locs
	minSim = p0
	expectedOnesPerWindow = p0 * X.shape[0] * windowLen
	noiseSz = p0 * expectedOnesPerWindow # num ones to begin with
	# noiseSz *= -np.log2(p0) # TODO this is right mathematically, but what will it do?
	# noiseSz = p0 * X.shape[0] * windowLen # way too hard to beat

	colSims = np.dot(X.T, Xblur)
	filt = np.zeros((windowLen, windowLen)) + np.diag(np.ones(windowLen)) # zeros except 1s on diag
	windowSims = sig.convolve2d(colSims, filt, mode='valid')

	windowVects = vectorizeWindowLocs(X, windowLen)
	windowVectsBlur = vectorizeWindowLocs(Xblur, windowLen)

	# plt.figure()
	# plt.imshow(windowSims, interpolation='nearest', aspect='auto')

	# ------------------------ find stuff

	#
	# Version where we look for similarities to orig seq and use nearest
	# enemy dist as M0, and use mean values instead of intersection
	#
	bsfScore = 0
	bsfLocs = None
	bsfIntersection = None
	for i, row in enumerate(windowSims):
		if i % 50 == 0:
			print("computing stuff for row {}".format(i))
		# early abandon if this location has so little stuff that no
		# intersection with it can possibly beat the best score
		if windowSims[i,i] * kMax <= bsfScore: # highest score is kMax identical locs
			continue

		# best combination of idxs such that none are within Lmin of each other
		idxs = sub.optimalAlignment(row, Lmin)

		# order idxs by descending order of associated score
		sizes = windowSims[i, idxs]
		sortedSizesOrder = np.argsort(sizes)[::-1]
		sortedIdxs = idxs[sortedSizesOrder]

		# iteratively intersect with another near neighbor, compute the
		# associated score, and check if it's better (or if we can early abandon)
		intersection = windowVects[i]
		numIdxs = len(sortedIdxs)
		nextSz = np.sum(intersection)
		nextFilt = np.array(intersection, dtype=np.float)
		nextFiltSum = np.array(nextFilt, dtype=np.float)
		for j, idx in enumerate(sortedIdxs):
			k = j + 1
			filt = np.copy(nextFilt)
			sz = nextSz
			if k < numIdxs:
				nextIdx = sortedIdxs[k] # since k = j+1
				# so we're zeroing out all the places where the filt is 0, but
				# where it isn't zero, we're not just adding the non-zeroed places
				# to the sum, but instead adding either them or the filter value
				# there, whichever is smaller; this is sort of a weird thing to
				# do. Maybe it gets us submodularity?
				# -actually, yes, this ensures that the weight of a given
				# feature is nonincreasing as locations are added
				# 	-which enables admissible early abandoning
				nextIntersection = np.minimum(filt, windowVectsBlur[nextIdx])
				nextFiltSum += nextIntersection
				nextFilt = nextFiltSum / (k+1)  # avg value of each feature in intersections
				# nextSz = np.sum(nextFilt) # big even if like no intersection...
				# nextSz = np.sum(nextIntersection)
				bigEnoughIntersection = nextIntersection[nextIntersection > minSim]
				nextSz = np.sum(bigEnoughIntersection)
			else:
				nextSz = sz * p0
			enemySz = max(nextSz, noiseSz)

			score = (sz - enemySz) * k
			if k > 1 and score > bsfScore:
				print("window {0}, k={1}, score={2} is the new best!".format(i, k, score))
				# print("sortedIdxs = {}".format(str(sortedIdxs)))
				# print("sortedIdxScores = {}".format(str(windowSims[i, sortedIdxs])))
				print("------------------------")
				bsfScore = score
				bsfLocs = sortedIdxs[:k]
				bsfIntersection = np.copy(filt)
			# early abandon if this can't possibly beat the best score, which
			# is the case exactly when the intersection is so small that perfect
			# matches at all future locations still wouldn't be good enough
			elif sz * numIdxs <= bsfScore:
				break # TODO can we actually early abandon here?
			elif noiseSz > nextSz:
				break

	# ------------------------ recover original ts

	print "bestScore = {}".format(bsfScore)
	print "bestLocations = {}".format(str(bsfLocs))

	bsfIntersection *= bsfIntersection >= minSim
	bsfIntersectionWindow = bsfIntersection.reshape((-1, windowLen))
	sums = np.sum(bsfIntersectionWindow, axis=0)

	kBest = len(bsfLocs)
	expectedOnesFrac = np.power(p0, kBest)
	expectedOnesPerCol = expectedOnesFrac * X.shape[0]
	sums -= expectedOnesPerCol

	# plt.figure()
	# plt.plot(sums)

	start, end, _ = maxSubarray(sums)

	print "learnFF: startIdxs, endIdxs:"
	print np.array(bsfLocs) + start
	print np.array(bsfLocs) + end
	print "learnFF: filtLen, windowLen = {}, {}".format(end - start, windowLen)

	return bsfLocs, bsfIntersectionWindow, start, end
