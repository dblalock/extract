#!/usr/bin/env python

import time
import numpy as np
from scipy import signal as sig

import arrays as ar
import feature as feat

# ================================================================
# Public functions
# ================================================================

def extract(seq, Lmin, Lmax, Lfilt=0):
	"""
	Finds the repeating pattern in `seq`.

	Parameters
	----------
	seq : 2D array
		2D array whose rows are time steps and whose columns are dimensions
		of the time series. If there is only one dimension, ensure that it
		is a column vector (so that it is 2D).
	Lmin : int > 0
		The minimum length that an instance of the pattern could be Must be
		> Lmax / 2.
	Lmax : int > 0
		The maximum length that an instance of the pattern could be. Must be
		< 2 * Lmin.
	Lfilt : int, optional
		The width of the hamming filter used to blur the feature matrix. If
		set to < 1, will default to Lmin.

	Returns
	-------
	startIdxs : 1D array
		The times (rows) in seq in which the estimated pattern instances
		begin (inclusive).
	endIdxs : 1D array
		The times (rows) in seq in which the estimated pattern instances
		end (non-inclusive).
	model : 2D array
		A learned model of the pattern. Can be seen as a digital filter
		that, when multiplied with a window of data, yields the log odds of
		that window being an instance of the pattern instead of iid
		Bernoulli random variables (ignoring overlap constraints). In
		practice, this is returned so that it can be plotted to show what
		features are selected.
	X : 2D array
		The feature matrix
	Xblur : 2D array
		The blurred feature matrix
	"""

	Lmin = int(len(seq) * Lmin) if Lmin < 1. else Lmin
	Lmax = int(len(seq) * Lmax) if Lmax < 1. else Lmax
	Lfilt = int(len(seq) * Lfilt) if Lfilt < 1. else Lfilt

	if not Lfilt or Lfilt < 0:
		Lfilt = Lmin

	# extend the first and last values out so that features using
	# longer windows are present for more locations
	padLen = Lmax
	seq = feat.extendSeq(seq, padLen, padLen)

	# build the feature matrix and blurred feature matrix
	X = feat.buildFeatureMat(seq, Lmin, Lmax)
	X, Xblur = feat.preprocessFeatureMat(X, Lfilt)

	# undo padding after constructing feature matrix
	X = X[:, padLen:-padLen]
	Xblur = Xblur[:, padLen:-padLen]
	seq = seq[padLen:-padLen]

	# catch edge case where all nonzeros in a row were in the padding
	keepRowIdxs = ar.nonzeroRows(X, thresh=1.)
	X = X[keepRowIdxs]
	Xblur = Xblur[keepRowIdxs]

	# feature matrices must satisfy these (if you plan on using your own)
	assert(np.min(X) >= 0.)
	assert(np.max(X) <= 1.)
	assert(np.min(Xblur) >= 0.)
	assert(np.max(Xblur) <= 1.)
	assert(np.all(np.sum(X, axis=1) > 0))
	assert(np.all(np.sum(Xblur, axis=1) > 0))

	startIdxs, endIdxs, bsfFilt = _learn(seq, X, Xblur, Lmin, Lmax, Lfilt)
	return startIdxs, endIdxs, bsfFilt, X, Xblur


# ================================================================
# Private functions
# ================================================================

def _dotProdsWithAllWindows(x, X):
	"""Slide x along the columns of X and compute the dot product"""
	return sig.correlate2d(X, x, mode='valid').flatten()


def _findAllInstancesFromSeedLoc(X, Xblur, seedStartIdx, seedEndIdx,
	Lmin, Lmax, Lfilt, p0, p0blur, logs_0, bsfScore=0):

	windowLen = seedEndIdx - seedStartIdx # assume end idx not inclusive

	# ================================ candidate location generation

	x0 = Xblur[:, seedStartIdx:seedEndIdx]
	dotProds = _dotProdsWithAllWindows(x0, X)

	# compute best locations to try and then sort them in decreasing order
	bestIdxs = ar.nonOverlappingMaxima(dotProds, Lmin)
	sortIdxs = np.argsort(dotProds[bestIdxs])[::-1]
	idxs = bestIdxs[sortIdxs]

	# ================================ now figure out which idxs should be instances

	# initialize counts
	idx = idxs[0]
	counts = np.copy(X[:, idx:idx+windowLen])
	countsBlur = np.copy(Xblur[:, idx:idx+windowLen])

	bestOdds = -np.inf
	bestFilt = None
	bestLocs = None
	for i, idx in enumerate(idxs[1:]):
		k = i + 2.

		# update counts
		window = X[:, idx:idx+windowLen]
		windowBlur = Xblur[:, idx:idx+windowLen]
		counts += window
		countsBlur += windowBlur

		# pattern odds
		theta_1 = countsBlur / k
		logs_1 = np.log(theta_1)
		logs_1[np.isneginf(logs_1)] = -999 # any non-inf number--will be masked by counts
		logDiffs = (logs_1 - logs_0)
		gains = counts * logDiffs # *must* use this so -999 is masked
		threshMask = gains > 0
		threshMask *= theta_1 > .5
		filt = logDiffs * threshMask
		logOdds = np.sum(counts * filt)

		# nearest enemy odds
		nextWindowOdds = -np.inf
		if k < len(idxs):
			idx = idxs[k]
			nextWindow = X[:, idx:idx+windowLen]
			nextWindowOdds = np.sum(filt * nextWindow) * k

		# subtract nearest enemy or "expected" enemy from noise
		randomOdds = np.sum(filt) * p0blur * k
		penalty = max(randomOdds, nextWindowOdds)
		logOdds -= penalty

		if logOdds > bestOdds:
			bestOdds = logOdds
			bestFilt = np.copy(filt)
			bestLocs = idxs[:k]

	return bestOdds, bestLocs, bestFilt


def _findInstancesUsingSeedLocs(X, Xblur, seedStartIdxs, seedEndIdxs, Lmin,
	Lmax, Lfilt, windowLen=None):

	# precompute feature mat stats so that evaluations of each seed
	# don't have to duplicate work
	p0, p0blur = np.mean(X), np.mean(Xblur)
	featureMeans = np.mean(Xblur, axis=1, keepdims=True)
	theta_0 = np.ones((Xblur.shape[0], windowLen)) * featureMeans
	logs_0 = np.log(theta_0)

	# main loop; evaluate each seed loc and return locs from the best one
	bsfScore = 0
	bsfFilt, bsfLocs = None, None
	for i in range(len(seedStartIdxs)):
		score, locs, filt = _findAllInstancesFromSeedLoc(X, Xblur,
			seedStartIdxs[i], seedEndIdxs[i], Lmin, Lmax, Lfilt,
			p0=p0, p0blur=p0blur, logs_0=logs_0, bsfScore=bsfScore)

		if score > bsfScore:
			bsfScore = score
			bsfFilt = np.copy(filt)
			bsfLocs = np.copy(locs)

	return bsfScore, bsfLocs, bsfFilt


def _extractTrueLocs(X, Xblur, bsfLocs, bsfFilt, windowLen, Lmin, Lmax):

	if bsfFilt is None:
		print "WARNING: _extractTrueLocs(): received None as filter"
		return np.array([0]), np.array([1])

	# compute the total filter weight in each column, ignoring low values
	bsfFiltWindow = np.copy(bsfFilt)
	sums = np.sum(bsfFiltWindow, axis=0)

	# subtract off the amount of weight that we'd expect in each column by chance
	kBest = len(bsfLocs)
	p0 = np.mean(Xblur)
	expectedOnesFrac = np.power(p0, kBest-1)
	expectedOnesPerCol = expectedOnesFrac * X.shape[0]
	sums -= expectedOnesPerCol

	# pick the optimal set of indices to maximize the sum of sequential column sums
	start, end, _ = ar.maxSubarray(sums)

	# ensure we picked at least Lmin points
	sumsLength = len(sums)
	while end - start < Lmin:
		nextStartVal = sums[start-1] if start > 0 else -np.inf
		nextEndVal = sums[end] if end < sumsLength else -np.inf
		if nextStartVal > nextEndVal:
			start -= 1
		else:
			end += 1
	# ensure we picked at most Lmax points
	while end - start > Lmax:
		if sums[start] > sums[end-1]:
			end -= 1
		else:
			start += 1

	locs = np.sort(np.asarray(bsfLocs))
	return locs + start, locs + end


def _computeAllSeedIdxsFromPair(seedIdxs, numShifts, stepLen, maxIdx):
	for idx in seedIdxs[:]:
		for shft in range(numShifts):
			seedIdxs += [idx + stepLen * shft, idx - stepLen * shft]

	seedIdxs = np.sort(np.unique(seedIdxs))
	return seedIdxs[(seedIdxs >= 0) * (seedIdxs <= maxIdx)]


def _learn(seq, X, Xblur, Lmin, Lmax, Lfilt, generateSeedsStep=.1):

	padLen = (len(seq) - X.shape[1]) // 2
	X = ar.addZeroCols(X, padLen, prepend=True)
	X = ar.addZeroCols(X, padLen, prepend=False)
	Xblur = ar.addZeroCols(Xblur, padLen, prepend=True)
	Xblur = ar.addZeroCols(Xblur, padLen, prepend=False)

	timeStartSeed = time.clock()

	# find seeds; i.e., candidate instance indices from which to generalize
	numShifts = int(1. / generateSeedsStep) + 1
	stepLen = int(Lmax * generateSeedsStep)
	windowLen = Lmax + stepLen

	# score all subseqs based on how much they don't look like random walks
	# when examined using different sliding window lengths
	scores = np.zeros(len(seq))
	for dim in range(seq.shape[1]):
		# compute these just once, not once per length
		dimData = seq[:, dim].flatten()
		std = np.std(dimData[1:] - dimData[:-1])
		for divideBy in [1, 2, 4, 8]:
			partialScores = feat.windowScoresRandWalk(dimData, Lmin // divideBy, std)
			scores[:len(partialScores)] += partialScores

	# figure out optimal pair based on scores of all subseqs
	bestIdx = np.argmax(scores)
	start = max(0, bestIdx - Lmin)
	end = min(len(scores), start + Lmin)
	scores[start:end] = -1 # disqualify idxs within Lmin of bestIdx
	secondBestIdx = np.argmax(scores)

	# compute all seed idxs from this pair
	seedIdxs = [bestIdx, secondBestIdx]
	maxIdx = X.shape[1] - windowLen - 1
	seedStartIdxs = _computeAllSeedIdxsFromPair(seedIdxs, numShifts, stepLen, maxIdx)
	seedEndIdxs = seedStartIdxs + windowLen

	timeEndSeed = time.clock()

	bsfScore, bsfLocs, bsfFilt = _findInstancesUsingSeedLocs(X, Xblur,
		seedStartIdxs, seedEndIdxs, Lmin, Lmax, Lfilt, windowLen=windowLen)

	startIdxs, endIdxs = _extractTrueLocs(X, Xblur, bsfLocs, bsfFilt,
		windowLen, Lmin, Lmax)

	timeEndFF = time.clock()
	print "learn(): seconds to find seeds, regions, total =\n\t{:.3f}\t{:.3f}\t{:.3f}".format(
		timeEndSeed - timeStartSeed, timeEndFF - timeEndSeed, timeEndFF - timeStartSeed)

	return startIdxs, endIdxs, bsfFilt
