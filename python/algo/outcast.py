#!/usr/env/python

import time
import numpy as np
from collections import namedtuple

from motif import findMotifOfLengthFast, nonOverlappingMinima
from motif import entropy

from ..utils import evaluate
from ..utils.arrays import isScalar
from ..utils.sequence import isListOrTuple
from ..utils.subseq import distsToRows


# ================================================================
# Data Structures
# ================================================================

_infoFields = [
	'score',
	'idxs',
	'length',
	'fromSeq',
]
OutcastInfo = namedtuple('OutcastInfo', _infoFields)


# ================================================================
# Functions
# ================================================================

# ------------------------------------------------ utility funcs

def computeAllSeedIdxsFromPair(idx1, idx2, stepLen, numShifts=0):
	if stepLen > 0. and numShifts < 1:
		numShifts = int(1. / stepLen)

	seedIdxs = [idx1, idx2]
	for idx in seedIdxs[:]:
		i = idx
		j = idx
		for shft in range(numShifts):
			i -= stepLen
			j += stepLen
			seedIdxs += [i, j]
	return seedIdxs


def startsAndEndsWithinBounds(startIdxs, subseqLen, seqLen):
	startIdxs = np.sort(np.asarray(startIdxs))
	startIdxs = startIdxs[startIdxs >= 0]
	startIdxs = startIdxs[startIdxs < seqLen - subseqLen]
	endIdxs = startIdxs + subseqLen
	return startIdxs, endIdxs


# ------------------------------------------------ search funcs

def findAllOutcastInstances(seqs, lengths):
	outcast = findOutcast(seqs, lengths)

	instances = []
	for idx in outcast.idxs:
		print "creating instance {}-{}".format(idx, idx + outcast.length)
		inst = evaluate.createPatternInstance(idx, idx + outcast.length)
		instances.append(inst)

	return instances


def findOutcast(seqs, lengths):
	tstart = time.clock()

	if isScalar(lengths):
		lengths = [lengths]
	if not isListOrTuple(seqs):
		seqs = [seqs]

	bestScore = -np.inf
	bestOutcast = None

	for m in lengths:
		info = findOutcastOfLength(seqs, m)
		if info and info.score > bestScore:
			bestScore = info.score
			bestOutcast = info

	print("Found best outcast in {}s".format(time.clock() - tstart))

	return bestOutcast


def findOutcastOfLength(seqs, length, shiftStep=.1, norm='each', mdl=False):

	# if numShifts < 0:
	# 	numShifts = 1
	# stepLen = shiftStep * length

	motif = findMotifOfLengthFast(seqs[:], length, norm=norm)
	Xnorm = motif.Xnorm

	seedIdxs = computeAllSeedIdxsFromPair(motif.idx1, motif.idx2, shiftStep)
	# XXX this assumes only one seq
	seedIdxs, _ = startsAndEndsWithinBounds(seedIdxs, length, len(seqs[0]))

	bestScore = -np.inf
	bestOutcast = None
	for idx in seedIdxs:
		seed = Xnorm[idx]
		if mdl:
			info = findOutcastInstancesMDL(Xnorm, seed, length)
		else:
			info = findOutcastInstances(Xnorm, seed, length)
		if info and info.score > bestScore:
			bestScore = info.score
			bestOutcast = info

	print "bestOutcast idxs at length {}: {}".format(length, bestOutcast.idxs,
		bestOutcast.length)

	return bestOutcast


def findOutcastInstances(Xnorm, seed, length, maxOverlapFraction=.1, fromSeq=None):
	minSpacing = max(int((1. - maxOverlapFraction) * length), 1)

	dists = distsToRows(Xnorm, seed)
	minimaIdxs = nonOverlappingMinima(dists, minSpacing, fromSeq=fromSeq)
	minimaDists = dists[minimaIdxs]

	# sort indices of relative minima in increasing order of distance
	sortIdxs = np.argsort(minimaDists)
	idxs = minimaIdxs[sortIdxs]
	dists = minimaDists[sortIdxs]

	centroidSums = seed
	centroid = np.copy(seed)
	distSum_pattern = 0

	vectLen = len(seed)

	bestGap = -np.inf
	bestIdxs = None
	for i, idx in enumerate(idxs[1:]):
		k = i + 2.

		# pattern model
		x = Xnorm[idx]
		diff = centroid - x
		distSum_pattern += np.dot(diff, diff) / vectLen

		centroidSums += x
		centroid = centroidSums / k

		# random walk
		AVG_DIST_TO_RAND_WALK = 1.
		# AVG_DIST_TO_RAND_WALK = .5
		distSum_walk = AVG_DIST_TO_RAND_WALK * k

		# nearest enemy
		distSum_enemy = np.inf
		if k < len(idxs):
			nextIdx = idxs[k]
			nextX = Xnorm[nextIdx]
			diff_enemy = centroid - nextX
			distSum_enemy = np.dot(diff_enemy, diff_enemy) / vectLen * k

		rivalSum = min(distSum_walk, distSum_enemy)
		gap = rivalSum - distSum_pattern
		if gap > bestGap:
			bestGap = gap
			bestIdxs = idxs[:k]

	return OutcastInfo(score=bestGap, idxs=bestIdxs, length=length, fromSeq=fromSeq)


def findOutcastInstancesMDL(Xnorm, seed, length, maxOverlapFraction=.1,
	fromSeq=None, mdlBits=6, useEnemy=True):
	minSpacing = max(int((1. - maxOverlapFraction) * length), 1)

	dists = distsToRows(Xnorm, seed)
	minimaIdxs = nonOverlappingMinima(dists, minSpacing, fromSeq=fromSeq)
	minimaDists = dists[minimaIdxs]

	# sort indices of relative minima in increasing order of distance
	sortIdxs = np.argsort(minimaDists)
	idxs = minimaIdxs[sortIdxs]
	dists = minimaDists[sortIdxs]

	# instanceIdxs = [idx1, idx2]

	# compute quantized subsequences
	numLevels = int(2**mdlBits)
	mins = np.min(Xnorm, axis=1).reshape((-1, 1))
	maxs = np.max(Xnorm, axis=1).reshape((-1, 1))
	ranges = (maxs - mins)
	Xquant = (Xnorm - mins) / ranges * (numLevels - 1) # 8 bits -> {0..255}
	Xquant = Xquant.astype(np.int)

	# initialize MDL stats
	row = Xquant[idxs[0]]
	centroidSums = np.copy(row)
	hypothesisEnt = entropy(row)
	origEnt = hypothesisEnt
	bitsave = -np.inf # ensure 2nd subseq gets added

	instanceIdxs = [idxs[0]]
	for i, idx in enumerate(idxs[1:]):
		k = i + 2.
		subseq = Xquant[idx]

		# compute original entropy of this instance along with current ones
		newOrigEnt = origEnt + entropy(subseq)

		# compute centroid when this instance is added
		newCentroidSums = centroidSums + subseq
		newCentroid = (newCentroidSums / k).astype(np.int)

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

		# divide by 2 as heuristic to reduce entropy, since description length
		# doesn't correspond to any obvious probabilistic model
		# noiseDiffs = Xquant[newInstanceIdxs] // 2
		# noiseCodedEnt = np.sum(entropy(noiseDiffs, axis=1))
		noiseCodedEnt = newCodedEnt / 2

		enemyCodedEnt = -np.inf
		if k < len(idxs):
			nextIdx = idxs[k]
			enemySubseq = Xquant[nextIdx]
			enemyDiffs = Xquant[newInstanceIdxs] - enemySubseq
			enemyCodedEnt = np.sum(entropy(enemyDiffs, axis=1))
		rivalEnt = min(noiseCodedEnt, enemyCodedEnt)
		newBitsave += rivalEnt

		if newBitsave > bitsave:
			bitsave = newBitsave
			origEnt = newOrigEnt
			centroidSums = newCentroidSums
			instanceIdxs = newInstanceIdxs
		# else:
		# 	break

	bestIdxs = sorted(instanceIdxs)
	return OutcastInfo(score=bitsave, idxs=bestIdxs, length=length, fromSeq=fromSeq)


def old_findOutcastInstances(Xnorm, seed, length, maxOverlapFraction=.1, fromSeq=None):
	minSpacing = max(int((1. - maxOverlapFraction) * length), 1)

	dists = distsToRows(Xnorm, seed)
	minimaIdxs = nonOverlappingMinima(dists, minSpacing, fromSeq=fromSeq)
	# invertMinimaIdxs = np.arange(len(dists))[minimaIdxs]

	# print "dists shape: ", dists.shape
	# print "found minimaIdxs: ", minimaIdxs

	minimaDists = dists[minimaIdxs]

	# sort indices of relative minima in increasing order of distance
	# TODO use a min heap, since that's O(n) and this is O(nlgn)
	sortIdxs = np.argsort(minimaDists)
	# unsortIdxs = np.arange(len(minimaDists))[sortIdxs]
	minimaIdxs = minimaIdxs[sortIdxs]
	minimaDists = minimaDists[sortIdxs]

	# initialize with best pair so we don't return anomalies
	idxs = [minimaIdxs[0], minimaIdxs[1]]
	# totalDist = 2 * minimaDists[1] # don't count self distance, since 0
	# maxDist = minimaDists[1]
	dist = minimaDists[1]
	nextIdx, nextDist = minimaIdxs[2], minimaDists[2]

	# bestScore = nextDist * len(idxs) - totalDist
	# bestScore = (nextDist - dist) * len(idxs)
	# bestScore = (nextDist - dist) * np.log(len(idxs))
	bestScore = (nextDist / dist) * np.log(len(idxs))
	bestIdxs = idxs[:]

	np.set_printoptions(precision=0)
	# print "minimaDists:", minimaDists
	print "minima diffs:", np.r_[0, minimaDists[1:] - minimaDists[:-1]]

	for i in range(2, len(minimaIdxs) - 1):
		idx, dist = nextIdx, nextDist
		nextIdx, nextDist = minimaIdxs[i+1], minimaDists[i+1]

		idxs.append(idx)
		# totalDist += dist
		# score = nextDist * len(idxs) - totalDist
		# score = (nextDist - dist) * len(idxs)
		# score = (nextDist - dist) * np.log(len(idxs))
		score = (nextDist / dist) * np.log(len(idxs))

		if score > bestScore:
			# print "new best score {} for idxs {}".format(score, idxs)
			bestScore = score
			bestIdxs = idxs[:]
		# else:
		# 	break

	bestIdxs = sorted(bestIdxs)

	return OutcastInfo(score=bestScore, idxs=bestIdxs, length=length, fromSeq=fromSeq)


# ================================================================ Main

def randWalkSeq(length):
	x = np.cumsum(np.random.randn(length)) # yields a weird plateau thing
	# x = np.random.randn(length) # yields a normal distro
	return (x - np.mean(x)) / np.std(x)
	# return (x - np.mean(x)) / np.std(x, ddof=1) # divide by n-1

def randWalkDists(numPairs=100000, seqLen=32):
	dists = np.empty(numPairs)
	for i in range(numPairs):
		x = randWalkSeq(seqLen)
		y = randWalkSeq(seqLen)
		diff = x - y
		dists[i] = np.dot(diff, diff)

	return dists / seqLen

if __name__ == '__main__':
	dists = randWalkDists()

	import matplotlib.pyplot as plt
	nBins = 50
	# plt.hist(dists, nBins, normed=True)
	# dists = np.sort(dists)
	# hist, edges = np.histogram(dists, nBins, normed=True)
	# cdf = np.cumsum(hist)
	# cdf /= np.max(cdf)
	# plt.plot(cdf) # basically a perfectly straight line


	# plt.yscale('log')
	plt.show()
	# ^ interesting; .5 and 3.5 are -3dB points, and basically flat between
	# those; full range is of course 0 to 4.
	#  -this roughly corresponds to a sigmoidal CDF; basically flat near
	# the edges, and linear regime (constant derivative) in the middle
	#  -but why would this yield a sigmoidal CDF?

	print "mu, sigma = {}, {}".format(np.mean(dists), np.std(dists))
	# prints ~2.007, ~.98, which suggests this is really 2 and 1
	# -and note that we're znorming the rand walks and returning squared
	# L2 dist over length
	# -interesting that it always returns sigma of ~.98, never 1.









