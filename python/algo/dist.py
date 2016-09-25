#!/usr/env/python

import numpy as np

from ..utils import arrays as ar
# from ..utils import sliding_window as window


def distsToVectors(X, V):
	"""Returns the distances between each row of X and each row of V. If X is
	n x m and v is p x m, returns an n x p matrix of L2 distances.

	>>> n, m, p = 100, 40, 10
	>>> X = np.random.randn(n, m)
	>>> V = np.random.randn(p, m)
	>>> dists = distsToVectors(X, V)
	>>> for i in range(n):
	... 	for j in range(p):
	... 		diff = X[i] - V[j]
	... 		d = np.sqrt(np.sum(diff * diff))
	...	        assert(np.abs(dists[i, j] - d) < .0001)
	>>>
	"""

	V = V.T # each col is now one of the vectors
	# ||x - v||^2 = ||x||^2 + ||v||^2 - 2 * x.v
	dists = -2. * np.dot(X, V)
	dists += np.sum(X*X, axis=1).reshape((-1, 1)) # add to each col
	dists += np.sum(V*V, axis=0) # add to each row

	# n = len(X)
	# numVects = len(V)

	# dists = np.empty((n, numVects))

	# for i, row in enumerate(X):
	# 	for j, col in enumerate(V):
	# 		diff = row - col
	# 		dists[i, j] = np.dot(diff, diff)
	# dists = np.sqrt(dists) # triangle inequality holds for norm, not norm^2

	return np.sqrt(dists)


def distsToRandomVects(X, numReferenceVects=10, referenceVects=None,
	referenceVectAlgo='gauss', norm='z', **sink):
	"""Creates a set of numReferenceVects Gaussian vectors and returns the
	distances of each row of X to each of these vectors as an
	(N x numReferenceVects) array, where N is the number of rows in X.
	Further, the columns are sorted in descending order of standard deviation.
	"""
	n, m = X.shape
	assert(m > 1) #
	if referenceVects is None:
		if referenceVectAlgo == 'gauss':
			referenceVects = np.random.randn(numReferenceVects, m) # rows are projection vects
		elif referenceVectAlgo == 'randwalk':
			referenceVects = np.random.randn(numReferenceVects, m)
			referenceVects = np.cumsum(referenceVects, axis=1)
		elif referenceVectAlgo == 'sample':
			idxs = np.random.choice(np.arange(len(X)))
			referenceVects = np.copy(X[idxs])

		if norm == 'z':
			referenceVects = ar.zNormalizeRows(referenceVects)
		elif norm == 'mean':
			referenceVects = ar.meanNormalizeRows(referenceVects)

	referenceDists = distsToVectors(X, referenceVects)

	# figure out std deviations of dists to different projections and
	# sort projections by decreasing std
	distStds = np.std(referenceDists, axis=0)
	refSortIdxs = np.argsort(distStds)[::-1]
	referenceDists = referenceDists[:, refSortIdxs]
	referenceVects = referenceVects[refSortIdxs]

	return referenceDists, referenceVects


def buildOrderline(X, numReferenceVects=10, **kwargs):
	# TODO comment this if we use it

	projDists, projVects = distsToRandomVects(X, numReferenceVects, **kwargs)

	sortIdxs = np.argsort(projDists[:, 0])
	unsortIdxs = np.arange(len(sortIdxs))[sortIdxs] # unsortIdxs: projected idx -> orig idx
	Xsort = X[sortIdxs, :]
	projDistsSort = projDists[sortIdxs, :]

	return Xsort, projDistsSort, projVects, unsortIdxs





