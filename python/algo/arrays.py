#!/usr/bin/env python

import numpy as np
from numpy.lib.stride_tricks import as_strided

DEFAULT_NONZERO_THRESH = .001


def addZeroCols(A, howMany=1, prepend=False):
	"""Returns a copy of the 2D array A with all-zero columns prepended or
	appended.

	Parameters
	----------
	A : 2D array
		The array to which zeros should be added.
	howMany: int > 0
		The number of all-zero columns to add.
	prepend : bool, optional
		Whether to prepend (or append) the columns

	Returns
	-------
	B : 2D array
		A copy of the matrix `A` with `howMany` zero columns added
	"""
	padding = np.zeros((A.shape[0], howMany))
	return np.hstack((padding, A)) if prepend else np.hstack((A, padding))


def nonzeroRows(A, thresh=DEFAULT_NONZERO_THRESH):
	"""Returns a 1D array containing indices of the rows of A whose summed
	absolute value is above the value thresh"""
	return np.where(np.sum(np.abs(A), axis=1) > thresh)[0]


# adapted from http://stackoverflow.com/a/15063394
def maxSubarray(x):
	"""Returns (a, b, c) such that sum(x[a:b]) = c and c is maximized.
	See https://en.wikipedia.org/wiki/Maximum_subarray_problem"""
	cur, best = 0, 0
	curi = starti = besti = 0
	for ind, i in enumerate(x):
		if cur + i > 0:
			cur += i
		else: # reset start position
			cur, curi = 0, ind + 1

		if cur > best:
			starti, besti, best = curi, ind + 1, cur
	return starti, besti, best


def sliding_window_1D(x, windowLen):
	"""
	Constructs a 2D array whose rows are the data in a sliding window that
	advances one time step at a time.

	Parameters
	----------
	x : 1D, array-like
		An ordered collection of objects
	windowLen : int > 0
		The legnth of the sliding window

	Returns
	-------
	X : 2D array
		A matrix such that X[i, :] = x[i:i+windowLen]
	"""
	x = x.flatten()
	numBytes = x.strides[0]
	numSubseqs = len(x) - windowLen + 1
	return as_strided(x, strides=(numBytes, numBytes), shape=(numSubseqs, windowLen))


def nonOverlappingMaxima(x, m):
	"""
	Finds the indices i such that |i - j| < m -> (x[i] > x[j]).

	Parameters
	----------
	x : 1D, array-like
		An ordered collection of objects for which "<" is defined
	m : int > 0
		The minimum spacing between indices returned

	Returns
	-------
	idxs : 1D array
		The indices i such that |i - j| < m -> (x[i] > x[j])
	"""

	idxs = []
	candidateIdx = 0
	candidateVal = x[0]
	for idx in range(1, len(x)):
		val = x[idx]
		# overlaps if within m of each other and from the same original seq
		if np.abs(idx - candidateIdx) < m: # overlaps
			if val > candidateVal:
				# replace current candidate
				candidateIdx = idx
				candidateVal = val
		else:
			# no overlap, so safe to flush candidate
			idxs.append(candidateIdx)

			# set this point as new candidate
			candidateIdx = idx
			candidateVal = val

	# the final candidate idx hasn't been added
	idxs.append(candidateIdx)

	return np.array(idxs)
