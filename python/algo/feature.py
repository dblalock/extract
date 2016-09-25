#!/usr/bin/env python

import numpy as np
from scipy.ndimage import filters

import arrays as ar

# ================================================================
# Public functions
# ================================================================

# ------------------------------------------------
# Structure scores
# ------------------------------------------------

def windowScoresRandWalk(seq, length, std=-1, numRandWalks=100):
	"""
	Computes the Structure Score for each subsequence of length `length`
	within `seq`.

	Parameters
	----------
	seq : 1D array
		1D time series (or a single dimension of an multidimensional time
		series)
	length : int > 0
		The length of subsequences. Each subsequence of this length is
		assigned a score.
	std : float, optional
		The standard deviation to use when creating random walks. Defaults
		to the standard devation of the first discrete derivative of `seq`.
	numRandWalks: int > 0, optional
		The number of random walk sequences to use to compute the score.
		Scores do not appear to become more meaningful for numbers > 100.

	Returns
	-------
	scores : 1D array
		The score for each subsequence of `seq`. This array is scaled by
		its maximimum value, so that the maximum value is always 1.
	"""
	numSubseqs = len(seq) - length + 1

	if length < 4: # length < 4 is meaninglessly short
		# if m <= 0, n - m + 1 is > n, which makes us return too long an array
		numSubseqs = numSubseqs if length > 0 else len(seq)
		return np.zeros(numSubseqs)

	std = std if std > 0 else np.std(seq[1:] - seq[:-1]) # std of discrete deriv

	walks = _createRandWalks(numRandWalks, length, std)

	windowScores = np.zeros(numSubseqs)
	subseqs = ar.sliding_window_1D(seq, length)

	for i, subseq in enumerate(subseqs):
		diffs = walks - (subseq - np.mean(subseq))
		dists = np.sum(diffs * diffs, axis=1) / length
		windowScores[i] = np.min(dists)

	return windowScores / np.max(windowScores) # normalize to max score of 1


# ------------------------------------------------
# Feature mat construction
# ------------------------------------------------

def buildFeatureMat(seq, Lmin, Lmax):
	"""
	Constructs a feature matrix for the time series seq using shape features.

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

	Returns
	-------
	X : 2D array
		The feature matrix. Each column is one time step and each row is one
		feature. Entries are the similarity between the row's feature and
		the data in seq centered at that time step.
	"""
	lengths = _defaultLengths(Lmin, Lmax)
	numNeighbors = int(np.log2(len(seq))) # log(n) neighbors
	X = _neighborSimsMat(seq, lengths, numNeighbors)
	X = X[np.sum(X, axis=1) > 1.] 		# features that happen more than once
	return (X > 0).astype(np.float64) 	# binarize matrix


def preprocessFeatureMat(X, Lfilt):
	"""
	Binarizes and blurs the feature matrix

	Parameters
	----------
	X : 2D array
		The original feature matrix (presumably output by buildFeatureMat())
	Lfilt : int
		The width of the hamming filter used to blur the feature matrix.

	Returns
	-------
	X : 2D array
		The modified feature matrix without blur
	Xblur : 2D array
		The modified feature matrix with blur
	"""
	Xblur = _filterRows(X, Lfilt)

	# ensure that the maximum value in Xblur is 1; we do this by dividing
	# by the largets value within Lfilt / 2, rather than just clamping, so
	# that there's a smooth dropoff as you move away from dense groups of
	# 1s in X; otherwise it basically ends up max-pooled
	maxima = filters.maximum_filter1d(Xblur, Lfilt // 2, axis=1, mode='constant')
	Xblur[maxima > 0] /= maxima[maxima > 0]

	# have columns be adjacent in memory
	return np.asfortranarray(X), np.asfortranarray(Xblur)


def extendSeq(seq, prePadLen, postPadLen):
	"""
	Extends time series `seq` by prepending `prePadLen` copies of the first row
	and appending `postPadLen` copies of the last row. `seq` is not modified.
	"""
	first = np.tile(seq[0], (prePadLen, 1))
	last = np.tile(seq[-1], (postPadLen, 1))
	return np.vstack((first, seq, last))


# ================================================================
# Private functions
# ================================================================

# ------------------------------------------------
# Structure scores
# ------------------------------------------------

def _createRandWalks(num, length, walkStd):
	walks = np.random.randn(num, length) * walkStd
	np.cumsum(walks, axis=1, out=walks)
	return walks - np.mean(walks, axis=1, keepdims=True)


# ------------------------------------------------
# Shape (near neighbor) features
# ------------------------------------------------

def _randChoice(a, size=1, replace=True, p=None):
	"""Wrapper for np.random.choice that deals with zero probabilities in p"""
	if p is None or not len(p):
		return np.random.choice(a, size, replace)

	nonzeroProbIdxs = p > 0
	p, a = p[nonzeroProbIdxs], a[nonzeroProbIdxs]
	size = min(size, len(a))

	return np.random.choice(a, size, replace, p)


def _neighborSims1D(seq, length, numNeighbors=100, maxDist=.25):

	seq = seq.flatten()
	X = ar.sliding_window_1D(seq, length)
	numSubseqs = X.shape[0]

	if numNeighbors < 1 or numNeighbors > numSubseqs:
		numNeighbors = numSubseqs

	probs = windowScoresRandWalk(seq, length)

	# select random subseqs
	probs /= np.sum(probs)
	allIdxs = np.arange(numSubseqs)
	startIdxs = _randChoice(allIdxs, numNeighbors, replace=False, p=probs)
	neighbors = X[startIdxs]

	# mean normalize all subseqs
	X = X - np.mean(X, axis=1, keepdims=True)
	neighbors = neighbors - np.mean(neighbors, axis=1, keepdims=True)

	# compute similarity to each shape
	sims = np.zeros((numNeighbors, numSubseqs)) # extra rows for uniform output
	for i, neighbor in enumerate(neighbors):
		variance = np.var(neighbor)
		if variance < ar.DEFAULT_NONZERO_THRESH: # ignore flat neighbors
			continue

		# compute squared dists; would be within [0, 4] if znormed
		diffs = X - neighbor
		dists = np.sum(diffs * diffs, axis=1) / (length * variance)

		sims[i] = (1. - dists) * (dists <= maxDist) # zero out dists > maxDist

	return sims

# ------------------------------------------------
# Feature mat construction
# ------------------------------------------------

def _neighborSimsMat(seq, lengths, numNeighbors):
	seq = seq if len(seq.shape) > 1 else seq.reshape((-1, 1)) # ensure 2d

	mats = []
	for dim, col in enumerate(seq.T):
		if np.var(col) < ar.DEFAULT_NONZERO_THRESH: # ignore flat dims
			continue
		mat = np.zeros((len(lengths) * numNeighbors, len(seq)))
		for i, m in enumerate(lengths):
			sims = _neighborSims1D(col, m, numNeighbors=numNeighbors)

			# we preallocated a matrix of the appropriate dimensions,
			# so we need to calculate where in that matrix to dump
			# the similarities computed at this length
			rowStart = i * numNeighbors # length inner loop
			rowEnd = rowStart + numNeighbors
			colStart = (mat.shape[1] - sims.shape[1]) // 2
			colEnd = colStart + sims.shape[1]

			mat[rowStart:rowEnd, colStart:colEnd] = sims

			# populate data past end of sims with median of each row;
			# better than 0 padding so that overly frequent stuff remains
			# overly frequent (so we'll remove it below)
			medians = np.median(mat[rowStart:rowEnd], axis=1, keepdims=True)
			mat[rowStart:rowEnd, :colStart] = medians
			mat[rowStart:rowEnd, colEnd:] = medians

		# remove rows where no features happened
		mat = mat[ar.nonzeroRows(mat)]

		# remove rows that are mostly nonzero, since this means the feature
		# is happening more often than not and thus isn't very informative
		minorityOnesRows = np.where(np.mean(mat > 0, axis=1) < .5)[0]
		mats.append(mat[minorityOnesRows])

	return np.vstack(mats)


def _filterRows(X, filtLength):
	filt = np.hamming(filtLength)
	return filters.convolve1d(X, weights=filt, axis=1, mode='constant')


def _logSpacedLengths(Lmin, Lmax, logStep=1.):
	logMaxLength = int(np.floor(np.log2(Lmax)))
	logMinLength = int(np.floor(np.log2(Lmin)))
	lengths = np.arange(logMinLength, logMaxLength + logStep, logStep)
	return (2. ** lengths).astype(np.int)


def _defaultLengths(Lmin, Lmax):
	lengths = _logSpacedLengths(8, Lmax)
	if len(lengths) == 0:
		lengths = [8, 16]
	elif len(lengths) == 1:
		lengths = np.array([lengths[0] // 2, lengths[0]])
	return lengths
