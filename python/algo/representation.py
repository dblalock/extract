#!/usr/bin/env/python

import os
import numpy as np

from scipy.stats import norm
from numba import jit

from ..utils import arrays as ar
from ..utils import sliding_window as window

from motif import nonOverlappingMaxima
from misc import windowScoresRandWalk

# ================================================================ Utilities


# ------------------------------------------------ Sparsification

def oneHotCodeFromIndices(indicesVect, cardinality=0, valuesVect=None, noZeroCols=False):
	n = len(indicesVect)
	# print "oneHotCodeFromIndices(): received cardinality {}".format(cardinality)

	if cardinality < 1:
		cardinality = np.max(indicesVect) + 1

	numCols = cardinality
	if noZeroCols:
		# indicesVect now contains col idxs using only nonzero cols
		uniqIndices, indicesVect = np.unique(indicesVect, return_inverse=True)
		numCols = len(uniqIndices)

	out = np.zeros((n, numCols))

	allRowIdxs = np.arange(n, dtype=np.int)
	if valuesVect is None:
		out[allRowIdxs, indicesVect] = 1.
	else:
		out[allRowIdxs, indicesVect] = valuesVect
	return out


# ------------------------------------------------ SAX

def quantiles(cardinality, noEndPoints=True):
	quantiles = np.linspace(0, 1, cardinality + 1)
	if noEndPoints:
		quantiles = quantiles[1:-1] # don't need 0 and 1
	return quantiles


def saxBreakpoints(cardinality):
	return norm.ppf(quantiles(cardinality))


def saxSymbolize(x, breakpoints):
	return np.digitize(x, breakpoints)


def L1DistsBetweenSymbols(breakpoints):
	cardinality = len(breakpoints) + 1
	dists = np.zeros((cardinality, cardinality))
	# dist(i, j):
	#  |i - j| <= 1 -> 0
	#  j - i > 1 -> breakPoints[j-1] - breakPoints[i]
	#  i - j > 1 -> above, but flip i and j
	for i in range(0, cardinality-2):
		for j in range(i+2, cardinality):
			dists[i, j] = breakpoints[j-1] - breakpoints[i]
	dists += dists.T

	return dists


def squaredL2DistsBetweenSymbols(breakpoints):
	diffs = L1DistsBetweenSymbols(breakpoints)
	return diffs * diffs


def paaMat(origLen, wordLen):
	"""returns a matrix A such that dot(x.T, A) = u, where u is the Piecewise
	Aggregate Approximation of x; basically, this lets us do a matrix
	multiply instead of iterating through x and computing means of its
	successive length (origLen // wordLen) subsequences. This is
	especially nice when operating on a whole data matrix X.

	Note that this just ignores the last few points if origLen isn't a
	multiple of wordLen.

	The returned matrix A is origLen x wordLen, so that XA returns the PAA
	of the rows of X
	"""

	symbolLen = origLen // wordLen
	startIdxs = np.arange(0, origLen, symbolLen)
	endIdxs = np.r_[startIdxs[1:], origLen]

	filterMat = np.zeros((wordLen, origLen))
	for i in range(wordLen):
		start, end = startIdxs[i], endIdxs[i]
		filterMat[i, start:end] = 1. / (end - start)

	return filterMat.T


def saxWords(Xnorm, saxMat, breakpoints):
	Xpaa = np.dot(Xnorm, saxMat)
	Xsax = np.empty(Xpaa.shape, dtype=np.int)
	for i in range(Xnorm.shape[0]):
		Xsax[i] = saxSymbolize(Xpaa[i], breakpoints)
	return Xsax


@jit
def sparsifySaxWords(Xsax, cardinality):
	"""one-hot encode sax word as a long sparse vector; eg:
	[0, 2, 1], cardinality 3 -> [1,0,0, 0,0,1, 0,1,0]"""

	numWords, wordLen = Xsax.shape
	sparseWordLen = wordLen * cardinality
	Xsparse = np.zeros((numWords, sparseWordLen), dtype=np.bool)

	baseIndicesInRow = np.arange(wordLen, dtype=np.int) * cardinality
	Xindices = (Xsax + baseIndicesInRow).astype(np.int)
	for i, row in enumerate(Xindices):
		Xsparse[i, row] = 1

	return Xsparse


def hashSaxWords(Xsax, cardinality):
	"""treat the entries of each row of Xsax as digits of a number expressed
	in base cardinality."""
	v = cardinality ** np.arange(Xsax.shape[1])
	return np.dot(Xsax, v)


def sparseSaxHashes(Xsax, cardinality, noZeroCols=True):
	hashes = hashSaxWords(Xsax, cardinality)
	# print "sparseSaxHashes; max hash = ", np.max(hashes)
	# print "Xsax shape: ", Xsax.shape
	hashMax = cardinality ** (Xsax.shape[1]) - 1 # should be encapsulated
	# print "sparseSaxHashes; hashMax = ", hashMax
	return oneHotCodeFromIndices(hashes, cardinality=(hashMax+1), noZeroCols=noZeroCols)
	# return oneHotCodeFromIndices(hashes, noZeroCols=noZeroCols)


# ------------------------------------------------ Sliding Normalization

class SlidingNormalizer(object):
	__slots__ = ["slidingVar"]

	def __init__(self):
		self.slidingVar = SlidingVariance()

	def initialize(self, initialX):
		self.slidingVar.initialize(initialX)

	def update(self, oldX, newX):
		self.slidingVar.update(oldX, newX)

	def normalize(self, x):
		var = self.slidingVar.variance
		# if var < .00001:
		# 	return 0.
		return (x - self.slidingVar.mean) / np.sqrt(var)


# ------------------------------------------------ Sliding Variance

# adapted from http://www.johndcook.com/blog/standard_deviation/
class RunningVariance(object): # variance of all data seen so far
	__slots__ = ("n", "mean", "SSE")

	def __init__(self):
		self.mean = 0
		self.SSE = 0
		self.n = 0

	@jit
	def update(self, x):
		self.n += 1
		if self.n == 1:
			self.mean = x
			return
		delta = (x - self.mean)
		self.mean += delta / self.n
		self.SSE += delta * (x - self.mean)

	def getVariance(self):
		return self.SSE / (self.n + 1)


def _computeVariance(mean2, mean):
	variance = mean2 - mean*mean
	if variance < .0001:
		variance = 0.
	return variance


class SlidingVariance(object):
	__slots__ = ('length', 'mean', 'mean2', 'variance')

	def initialize(self, initialData):
		"""initialData: a vector"""
		self.length = len(initialData)
		self.mean = np.mean(initialData)
		self.mean2 = np.mean(initialData*initialData)
		self.variance = self.mean2 - self.mean*self.mean

	def update(self, oldX, newX):
		self.mean += (newX - oldX) / self.length
		self.mean2 += (newX*newX - oldX*oldX) / self.length
		self.variance = _computeVariance(self.mean2, self.mean)

#
# aha. the key is to compute *running* SSE (using Welford's stable algorithm)
# at beginning and end of window, and just use difference between those to
# robustly get SSE within the window
#  -or perhaps compute the variances and weight them appropriately
#    -although I think this will still result in having big numbers
#
# EDIT: this just doesn't seem to work very well...
#
# class SlidingVariance2(object):
# 	__slots__ = ('length', 'runningVar1', 'runningVar2', 'SSE')

# 	def __init__(self):
# 		self.runningVar1 = RunningVariance()
# 		self.runningVar2 = RunningVariance()

# 	def initialize(self, initialData):
# 		"""initialData: a vector"""
# 		self.length = len(initialData)
# 		for x in initialData:
# 			self.runningVar2.update(x)
# 		self.SSE = self.runningVar2.SSE

# 	def update(self, x):
# 		# self.runningVar1.update(x)
# 		# self.runningVar2.update(x)
# 		# self.SSE = self.runningVar2.SSE - self.runningVar1.SSE

# 		# we split the data into the portion seen by runningVar1 and the
# 		# portion not seen by runningVar1, and derive the variance of the
# 		# latter from the formula for combining means + variances of datasets;
# 		# see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
# 		#
# 		# d = ua - ub
# 		# uc = (na*ua + nb*ub) / (na + nb)
# 		# Sc = Sa + Sb + d^2 * (na*nb / (na + nb))
# 		#
# 		# Therefore:
# 		#
# 		# uc(na + nb) = na*ua + nb*ub
# 		# -> uc(na + nb) - na*ua = nb*ub
# 		# -> ub = (uc(na + nb) - na*ua)/nb
# 		#   	= (uc*na + uc*nb - na*ua) / nb
# 		# 		=  uc*na/nb + uc - ua*na/nb
# 		# d = ua - ub 	# unchanged
# 		# Sb = Sc - Sa - d^2 * (na*nb / (na + nb))

# 		u_a = self.runningVar1.mean
# 		u_c = self.runningVar2.mean
# 		S_a = self.runningVar1.SSE
# 		S_c = self.runningVar2.SSE
# 		n_a = self.runningVar1.n
# 		n_c = self.runningVar2.n
# 		n_b = n_c - n_a
# 		u_b = (u_c*n_c - u_a*n_a) / float(n_b) # total sum - sum before window

# 		d = u_a - u_b
# 		S_b = S_c - S_a - d * d * (n_a * n_b / float(n_c))

# 		self.SSE = S_b

# 	def getVariance(self):
# 		return self.SSE / (self.length + 1)


# ------------------------------------------------ Straight Line Projection

@jit
def updateDotProduct(xy_old, x_sum_old, x_new, length): # this is the clever part
	return xy_old - x_sum_old + x_new * length


@jit
def updateProjection(Exy_old, Ey, Syy, Ex_old, Ex2_old, x_old, x_new, length):
	Exy = updateDotProduct(Exy_old, Ex_old, x_new, length)

	# note that parens in this block matter for numerical reasons
	Ex = Ex_old + (x_new - x_old)
	Ex2 = Ex2_old + (x_new * x_new - x_old * x_old)
	Sxx = Ex2 - Ex * Ex / length
	Sxy = Exy - Ex * Ey / length

	if Sxx < .0001 or np.abs(Sxy) < .001: # flat signal
		corr = 0.
		Sxy = 0.
		# print "Ex, Ex2, Sxx, Syy, Sxy, r:"
		# print Ex, Ex2, Sxx, Syy, Sxy, corr
	else:
		corr = Sxy / np.sqrt(Sxx * Syy)
		if not (-1.001 < corr < 1.001):
			print "Ex, Ex2, Sxx, Syy, Sxy, r:"
			print Ex, Ex2, Sxx, Syy, Sxy, corr
			assert(False)

	return corr, Ex, Ex2, Exy, Sxy


class SlidingStraightLineProjection(object):
	__slots__ = ('length', 'Ey', 'Syy', 'Ex', 'Ex2', 'Exy', 'r', 'slope')

	def initialize(self, initialX):
		"""initialX: a vector"""
		length = len(initialX)
		self.length = length
		line = np.arange(1, length + 1) # 1 thru length, inclusive

		# initial data stats
		Ex = np.sum(initialX)
		Ex2 = np.sum(initialX*initialX)
		self.Ex = Ex
		self.Ex2 = Ex2

		# xy stats
		Exy = np.dot(initialX.flatten(), line)
		self.Exy = Exy

		# line data stats
		# https://proofwiki.org/wiki/Sum_of_Sequence_of_Squares
		Ey = (length * (length + 1)) / 2 # sum of 1..length, inclusive
		Ey2 = (length * (length + 1) * (2*length + 1)) / 6
		Syy = Ey2 - Ey * Ey / length
		self.Ey = Ey
		self.Syy = Syy

		# corr coef
		Sxx = Ex2 - Ex * Ex / length
		Sxy = Exy - Ex * Ey / length
		if Sxx > .001:
			self.r = Sxy / np.sqrt(Syy * Sxx)
			self.slope = Sxy / Syy
		else:
			self.r = 0.
			self.slope = 0.

	def update(self, oldX, newX):
		"""oldX just left the window, and newX just entered it"""
		self.r, self.Ex, self.Ex2, self.Exy, Sxy = updateProjection(self.Exy, self.Ey,
			self.Syy, self.Ex, self.Ex2, oldX, newX, self.length)
		self.slope = Sxy / self.Syy


# ------------------------------------------------ Munging

def applyToEachCol(func, X, numOutputRows, numOutputCols, *args, **kwargs):
	X = np.asarray(X)
	nDims = 1
	# print "applyToEachCol(): outRows, outCols = {}, {}".format(numOutputRows, numOutputCols)
	if len(X.shape) == 2 and X.shape[1] > 1:
		nDims = X.shape[1]
		out = np.empty((nDims, numOutputRows, numOutputCols))
		for dim in range(nDims):
			data = X[:, dim]
			if np.var(data) < .01: # ignore flat dims
				out[dim] = 0
				continue
			output = func(data, *args, **kwargs)
			# print "applyToEachCol, dim {} output shape = {}".format(dim, output.shape)
			out[dim] = output.reshape((numOutputRows, numOutputCols))

		if numOutputCols == 1:
			out = out.reshape(out.shape[:-1]) # remove trailing 3rd cold
		return out

	return func(X, *args, **kwargs)


def hstack3Tensor(X):
	nDims = X.shape[0]
	transposed = np.empty((nDims, X.shape[2], X.shape[1]))
	for dim in range(nDims):
		transposed[dim] = X[dim].T
	return vstack3Tensor(transposed).T


def vstack3Tensor(X):
	# assumes time progresses along columns
	return X.reshape((-1, X.shape[2])) # add rows for each leading dim


# ================================================================ Signal Transforms

# ------------------------------------------------ SAX

# ------------------------ sax words

def saxify1D(seq, length, wordLen, cardinality):
	Xnorm, _, _ = window.flattened_subseqs_of_length([seq], length, norm='each')
	saxMat = paaMat(length, wordLen)
	breakpoints = saxBreakpoints(cardinality)
	return saxWords(Xnorm, saxMat, breakpoints)


def saxify(X, length, wordLen, cardinality):
	numSubseqs = len(X) - length + 1
	numOutputRows = numSubseqs
	numOutputCols = wordLen

	return applyToEachCol(saxify1D, X, numOutputRows, numOutputCols,
		length, wordLen, cardinality)


# ------------------------ sparse sax words

def sparseSaxify1D(seq, length, wordLen, cardinality):
	words = saxify1D(seq, length, wordLen, cardinality)
	return sparsifySaxWords(words, cardinality)


def sparseSaxify(X, length, wordLen, cardinality):
	numSubseqs = len(X) - length + 1
	numOutputRows = numSubseqs
	numOutputCols = wordLen * cardinality

	return applyToEachCol(sparseSaxify1D, X, numOutputRows, numOutputCols,
		length, wordLen, cardinality)


# ------------------------ sax hashes

def sparseSaxHash1D(seq, length, wordLen, cardinality, noZeroCols=True):
	saxWords = saxify1D(seq, length, wordLen, cardinality)
	return sparseSaxHashes(saxWords, cardinality, noZeroCols=noZeroCols)


def sparseSaxHashify(X, length, wordLen, cardinality):
	numSubseqs = len(X) - length + 1
	numOutputRows = numSubseqs
	numOutputCols = cardinality ** wordLen
	# print "sparseSaxHashify: numOutputCols = {}".format(numOutputCols)

	return applyToEachCol(sparseSaxHash1D, X, numOutputRows, numOutputCols,
		length, wordLen, cardinality, noZeroCols=False)


# ------------------------------------------------ Variance

def runningVariance1D(seq):
	n = len(seq)
	out = np.empty(n)

	monitor = RunningVariance()
	for i in range(n):
		monitor.update(seq[i])
		out[i] = monitor.SSE / (i+1)

	return out


def slidingVariance1D(seq, length):
	n = len(seq)
	numSubseqs = n - length + 1
	out = np.empty(numSubseqs)

	monitor = SlidingVariance()
	# monitor = SlidingVariance2()
	monitor.initialize(seq[:length])
	out[0] = monitor.variance
	for i in range(length, n):
		oldX, newX = seq[i - length], seq[i]
		monitor.update(oldX, newX)
		# monitor.update(seq[i])
		# out[i - length + 1] = monitor.SSE
		out[i - length + 1] = monitor.variance

	# return out / length
	return out


def slidingVariance(X, length):
	numSubseqs = len(X) - length + 1
	numOutputRows = numSubseqs
	numOutputCols = 1

	return applyToEachCol(slidingVariance1D, X, numOutputRows, numOutputCols, length)


def sparseVariance(X, length, breakpoints=None):
	if breakpoints is None:
		breakpoints = np.linspace(0., 1., 9)[1:-1] # 7 breakpoints, 8 levels
		# breakpoints = saxBreakpoints(8)

	# compute variance for each sliding window position
	variance = slidingVariance(X, length).T
	variance /= np.max(variance) # set max variance to 1 for quantization
	variance = ar.zNormalizeCols(variance)

	# quantize and one-hot-encode variances
	quantized = quantize(variance, breakpoints)
	cardinality = len(breakpoints) + 1
	numOutputRows = len(quantized)
	numOutputCols = cardinality
	out = applyToEachCol(oneHotCodeFromIndices, quantized, numOutputRows,
		numOutputCols, cardinality=cardinality)

	# zero out places with like no variance
	# out[..., 0] = 0

	return out


# ------------------------------------------------ Normalization

def slidingNormalize1D(seq, length, whichPoint='middle'):
	n = len(seq)
	numSubseqs = n - length + 1
	out = np.empty(numSubseqs)

	monitor = SlidingNormalizer()
	monitor.initialize(seq[:length])

	if whichPoint == 'first': # normalize first point in window
		out[0] = monitor.normalize(seq[0])
		for i in range(length, n):
			oldX, newX = seq[i - length], seq[i]
			monitor.update(oldX, newX)
			out[i - length + 1] = monitor.normalize(seq[i - length + 1])

	elif whichPoint == 'last': # normalize last point in window
		out[0] = monitor.normalize(seq[0])
		for i in range(length, n):
			oldX, newX = seq[i - length], seq[i]
			monitor.update(oldX, newX)
			out[i - length + 1] = monitor.normalize(seq[i])

	else: # normalize middle point in window
		halfLen = int(length/2)
		out[0] = monitor.normalize(seq[length - halfLen])
		for i in range(length, n):
			oldX, newX = seq[i - length], seq[i]
			monitor.update(oldX, newX)
			out[i - length + 1] = monitor.normalize(seq[i - halfLen])

	return out


def slidingNormalize(X, length):
	numSubseqs = len(X) - length + 1
	numOutputRows = numSubseqs
	numOutputCols = 1

	return applyToEachCol(slidingNormalize1D, X, numOutputRows, numOutputCols, length)


# ------------------------------------------------ Quantization

def quantize(X, breakpoints):
	if len(X.shape) == 1:
		return np.digitize(X, breakpoints)

	X = np.array(X, dtype=np.int)
	for i, row in enumerate(X):
		X[i] = np.digitize(row, breakpoints)
	return X
	# return np.digitize(X, breakpoints)
	# if len(X.shape) > 1:
	# 	numOutputRows, numOutputCols = X.shape
	# else:
	# 	numOutputRows, numOutputCols = len(X), 1
	# return applyToEachCol(np.digitize, X, numOutputRows, numOutputCols, breakpoints)


def normalizeAndSparseQuantize1D(X, length, breakpoints=None):
	normed = slidingNormalize1D(X, length)
	if breakpoints is None:
		normed = ar.zeroOneScaleMat(normed)
		breakpoints = np.linspace(0, 1, 9)[1:-1] # 8 levels
		# normed = ar.zNormalize(normed)
		# breakpoints = saxBreakpoints(8)
	quantized = quantize(normed, breakpoints)
	cardinality = len(breakpoints) + 1
	return oneHotCodeFromIndices(quantized, cardinality)


def normalizeAndSparseQuantize(X, length, breakpoints=None):
	numSubseqs = len(X) - length + 1
	numOutputRows = numSubseqs
	if breakpoints is None:
		numOutputCols = 8
	else:
		numOutputCols = len(breakpoints) + 1

	return applyToEachCol(normalizeAndSparseQuantize1D, X,
		numOutputRows, numOutputCols, length, breakpoints)


# ------------------------------------------------ Line

# @jit
def lineProjection1D(seq, length):
	n = len(seq)
	numSubseqs = n - length + 1
	correlations = np.empty(numSubseqs)
	slopes = np.empty(numSubseqs)

	proj = SlidingStraightLineProjection()
	proj.initialize(seq[:length])
	correlations[0] = proj.r
	slopes[0] = proj.slope
	for i in range(length, n):
		oldX, newX = seq[i - length], seq[i]
		proj.update(oldX, newX)
		correlations[i - length + 1] = proj.r
		slopes[i - length + 1] = proj.slope

	return correlations, slopes


# @jit
# def lineProject(X, length):
# 	numSubseqs = len(X) - length + 1
# 	numOutputRows = numSubseqs
# 	numOutputCols = 1

# 	return applyToEachCol(lineProjection1D, X, numOutputRows, numOutputCols, length)


def sparseLineProjection1D(seq, length, breakpoints, maxFilter=False, ignoreFlat=False):

	corrs, slopes = lineProjection1D(seq, length)
	increases = slopes * length

	numSubseqs = len(corrs)
	numBreakpoints = len(breakpoints)
	cardinality = 2 * numBreakpoints + 1

	signs = 2 * (corrs >= 0) - 1
	indices = np.digitize(increases * signs, breakpoints) * signs + numBreakpoints
	out = np.zeros((numSubseqs, cardinality))

	# if maxFilter is true, we use the indices associated with whichever point
	# best explains this time step at this length, as measured by |correlation|
	if maxFilter:
		# import matplotlib.pyplot as plt
		# plt.figure()
		# plt.plot(np.abs(corrs))
		# plt.plot(indices)

		maxIdxs = ar.slidingMaximaIdxs(np.abs(corrs), length // 2, pastEnd=True)
		indices = indices[maxIdxs]

		# plt.plot(indices)
		# plt.show()

	allRowsIdxs = np.arange(numSubseqs, dtype=np.int)

	# if False:
	if ignoreFlat:
		zeroIdx = numBreakpoints
		# whereFlat = np.where(indices == zeroIdx)[0]
		# corrs[whereFlat] = 0.
		whereNotFlat = np.where(indices != zeroIdx)[0]
		allRowsIdxs = allRowsIdxs[whereNotFlat]
		indices = indices[whereNotFlat]
		corrs = corrs[whereNotFlat]

	out[allRowsIdxs, indices] = np.abs(corrs)

	return out


def sparseLineProject(X, length, breakpoints, **kwargs):
	# TODO allow different breakpoints across dims

	numSubseqs = len(X) - length + 1
	numOutputRows = numSubseqs
	numOutputCols = 2 * len(breakpoints) + 1

	return applyToEachCol(sparseLineProjection1D, X, numOutputRows,
		numOutputCols, length, breakpoints, **kwargs)


def defaultSparseLineBreakpoints(X, scaleHowMany=1, cardinality=8, logScale=False):
	minVals, maxVals = np.min(X, axis=0), np.max(X, axis=0)
	valRange = np.max(maxVals - minVals) # biggest range of any dimension
	# minVariance = valRange / 100.
	if logScale:
		minVariance = valRange / 100.
		numBreakpoints = int(np.ceil(np.log2(valRange) - np.log2(minVariance)))
		numBreakpoints *= scaleHowMany
		return np.logspace(np.log2(minVariance), np.log2(valRange), num=numBreakpoints, base=2.)
	else:
		return np.linspace(0, valRange, cardinality+1)[1:-1] # 7 breakpoints -> 8 levels


# ------------------------------------------------ Neighbors

def randChoice(a, size=1, replace=True, p=None):
	"""
	Wrapper for numpy random.choice that deals with zero probabilities (in p)
	"""
	if p is None or not len(p):
		return np.random.choice(a, size, replace)

	nonzeroProbIdxs = p > 0
	numNonzeroProbs = np.sum(nonzeroProbIdxs)
	if numNonzeroProbs == 0 or (numNonzeroProbs < size and not replace):
		raise ValueError("randChoice(): fewer nonzero probabilities"
		"({}) than requested size ({})".format(numNonzeroProbs, size))
	p = p[nonzeroProbIdxs]
	a = a[nonzeroProbIdxs]
	size = min(size, len(a))

	return np.random.choice(a, size, replace, p)


def randIdxs(length, num, minSpacing=1, probabilities=None,
	reduceSpacingIfNeeded=False, reduceNumIfNeeded=False):
	"""
	Returns `num` unique indices within ``[0, length)``; probabilities of selecting
	each index can be specified with `probabilities`, and a desired minimum
	spacing between indices (not guaranteed) can be specified with `minSpacing`.

	If `reduceSpacingIfNeeded` is True, minSpacing will be set to a smaller
	value when it is not possible to return `num` samples `minSpacing` indices
	apart.
	"""
	hasProbs = probabilities is not None and len(probabilities)

	if hasProbs and len(probabilities) != length:
		raise ValueError("Probabilities have length"
			"{} != requested length {}".format(len(probabilities), length))

	# convert fractional minSpacing and ensure that it's an int >= 1
	if 0. <= minSpacing < 1.:
		minSpacing *= length
	minSpacing = int(max(1, minSpacing))

	neededLength = (num - 1) * minSpacing + 1
	if neededLength > length:
		if reduceSpacingIfNeeded:
			minSpacing = int((length - 1) / (num - 1))
		elif reduceNumIfNeeded:
			num = neededLength // minSpacing + 1
		else:
			raise ValueError("cannot return {} indices out of {} with minSpacing {}".format(
				num, length, minSpacing))
	if hasProbs and len(probabilities) != length:
		raise ValueError("Probabilities vector had length {}, not {}".format(
			len(probabilities), length))

	# if too many samples requested to bother spacing them, just sample
	# directly from the set of possible idxs
	# if minSpacing == 1:
	# 	idxs = np.random.choice(np.arange(length), num,
	# 		replace=False, p=probabilities)
	# 	return np.sort(idxs.astype(np.intp)) # TODO don't sort for speed

	# sum together the probabilities of every minSpacing adjacent points
	probs = probabilities if hasProbs else None
	if minSpacing > 1:
		remainder = length % minSpacing
		padLen = (minSpacing - remainder) % minSpacing # mod in case remainder=0
		probsReshape = np.append(probs, np.zeros(padLen))

		# print "length, remainder, minSpacing, padLen = ", length, remainder, minSpacing, padLen

		probsReshape = probsReshape.reshape((-1, minSpacing))
		probs = np.sum(probsReshape, axis=1)

	# compute the (rough) idxs; we divide by minSpacing before deciding
	# and then multiply by minSpacing so that the points are at least
	# minSpacing apart
	allIdxs = np.arange(np.ceil(length / float(minSpacing)), dtype=np.intp)
	idxs = randChoice(allIdxs, num, replace=False, p=probs)
	# print "len(allIdxs), len(probs)", len(allIdxs), len(probs)
	# print "length, minSpacing, ceil(len/minSpacing)", length, minSpacing, np.ceil(length / minSpacing)
	# print "num nonzero probs:", np.sum(probs > 0)

	# random.choice can't deal with probabilities of zero
	# nonzeroProbIdxs = probs > 0
	# probs = probs[nonzeroProbIdxs]
	# allIdxs = allIdxs[nonzeroProbIdxs]
	# num = min(num, len(allIdxs))
	# # print "num, len(allIdxs), len(probs)", num, len(allIdxs), len(probs)
	# # print "num nonzero probs:", np.sum(probs > 0)

	# idxs = np.random.choice(allIdxs, num, replace=False, p=probs)

	# add offset to each idx so that we don't always return multiples
	# of minSpacing
	probs = probabilities if hasProbs else np.ones(length) / length # uniform
	if minSpacing > 1:
		idxs *= minSpacing
		allOffsetIdxs = np.arange(minSpacing)
		for i, idx in enumerate(idxs):
			offsetProbs = probs[idx:idx+minSpacing]
			offsetProbs /= np.sum(offsetProbs)
			possibleIdxs = allOffsetIdxs[:len(offsetProbs)] # edge case if near end
			offset = randChoice(possibleIdxs, replace=True, p=offsetProbs)
			idxs[i] += int(offset)

	# return idxs.astype(np.intp)
	return np.sort(idxs.astype(np.intp)) # TODO don't sort for speed

@jit
def pairwiseDists(X):
	numSubseqs = len(X)
	allVariances = np.var(X, axis=1)
	allDists = np.zeros((numSubseqs, numSubseqs))
	for i in range(numSubseqs):
		variance = allVariances[i]
		for j in range(i+1, numSubseqs):
			diff = X[i] - X[j]
			# absDiff = np.abs(diff)
			# allDists[i, j] = np.sum(absDiff * absDiff)
			# allDists[i, j] = np.sum(np.abs(diff))
			allDists[i, j] = np.sum(diff * diff) / variance
			# allDists[i, j] = np.sum(diff * diff)
			# allDists[i, j] = i + j

	allDists += allDists.T

	# ignore self-similarity
	diagIdxs = np.arange(numSubseqs)
	allDists[[diagIdxs, diagIdxs]] = X.shape[1] * np.max(allVariances)

	return allDists


def neighborSims1D(seq, length, numNeighbors=100, samplingAlgo='walk',
	similarityAlgo='meanOnly', maxDist=.25, localMaxFilter=False,
	spacedMaxFilter=False, tryNumNeighbors=-1, **sink):
	# spacedMaxFilter=True, tryNumNeighbors=-1, **sink):

	# print "neighborSims1D(); seq shape, requested len, requested count"
	# print seq.shape, length, numNeighbors

	seq = seq.flatten()
	X = window.sliding_window_1D(seq, length)
	numSubseqs = X.shape[0]

	if numNeighbors < 1 or numNeighbors > numSubseqs:
		numNeighbors = numSubseqs
		# origNumNeighbors = numNeighbors
	# elif baseLength:
	# 	origNumNeighbors = numNeighbors
	# 	numNeighbors = int(numNeighbors * float(length) / baseLength)

	if samplingAlgo == 'std':
		probs = np.std(X, axis=1)
	elif samplingAlgo == 'var':
		probs = np.var(X, axis=1)
	elif samplingAlgo == 'unif':
		probs = np.ones(numSubseqs)
	elif samplingAlgo == 'walk':
		probs = windowScoresRandWalk(seq, length)
	else:
		raise ValueError("Unrecognized sampling algorithm {}".format(samplingAlgo))

	# must assess at least as many subseqs as we want to return, and no more
	# than the largest number possible
	tryNumNeighbors = max(tryNumNeighbors, numNeighbors)
	tryNumNeighbors = min(tryNumNeighbors, numSubseqs)

	# print "neighborSims1D(); X shape ", X.shape

	# print np.var(X, axis=1)

	# allDists = pairwiseDists(X)
	# # allDists = pairwiseDists(X) / length
	# # import matplotlib.pyplot as plt
	# # from ..viz import viz_utils as viz
	# # plt.figure()
	# # viz.imshowBetter(allDists)
	# # plt.show()
	# # import sys
	# # sys.exit()

	# # closeEnough = (allDists < maxDist).astype(np.int)
	# # closeEnough = allDists < maxDist
	# closeEnough = allDists < (maxDist * length)
	# neighborCounts = np.sum(closeEnough, axis=1)
	# print neighborCounts
	# eligibleIdxs = np.where(neighborCounts > 2)[0] # self isn't a neighbor
	# # print eligibleIdxs
	# numEligibleIdxs = len(eligibleIdxs)

	# print "numSubseqs, numEligibleIdxs ", numSubseqs, numEligibleIdxs

	# select random subseqs
	probs /= np.sum(probs)
	allIdxs = np.arange(numSubseqs)
	startIdxs = randChoice(allIdxs, tryNumNeighbors, replace=False, p=probs)
	# minSpacing = length // 2
	# startIdxs = randIdxs(numSubseqs, numNeighbors, minSpacing=minSpacing,
	# 	probabilities=probs, reduceSpacingIfNeeded=True)
	# 	probabilities=probs, reduceNumIfNeeded=True)
	neighbors = X[startIdxs]

	# mean normalize all subseqs
	X = X - np.mean(X, axis=1, keepdims=True)
	neighbors = neighbors - np.mean(neighbors, axis=1, keepdims=True)

	# zNorm = True # TODO remove
	# if zNorm:
	# 	X = ar.zNormalizeRows(X)
	# 	neighbors = ar.zNormalizeRows(neighbors)

	# SELF: pick up here by ensuring sufficient features
	# import dist
	# Xsort, projDistsSort, projVects, unsortIdxs = dist.buildOrderline(X,
	# 	referenceVectAlgo='sample', norm=None)

	# allVariances = np.var(X, axis=1)
	# sortIdxs = np.argsort(allVariances)
	# allVariances = allVariances[sortIdxs]

	# sims = np.zeros((origNumNeighbors, numSubseqs)) # extra rows for uniform output
	sims = np.zeros((tryNumNeighbors, numSubseqs)) # extra rows for uniform output

	if similarityAlgo == 'meanOnly':
		for i, neighbor in enumerate(neighbors):
			variance = np.var(neighbor)
			if variance < .0001:
				continue

			diffs = X - neighbor
			dists = np.sum(diffs * diffs, axis=1) / length
			dists /= variance # would be within [0, 2] if znormed

			dists[dists > maxDist] = np.inf
			neighborSims = np.maximum(0, 1. - dists)

			# print "i, sims shape", i, neighborSims.shape

			if localMaxFilter:
				idxs = ar.idxsOfRelativeExtrema(neighborSims.ravel(), maxima=True)
				sims[i, idxs] = neighborSims[idxs]
			elif spacedMaxFilter:
				idxs = nonOverlappingMaxima(neighborSims, length // 2)
				# idxs = nonOverlappingMaxima(neighborSims, 2) # spacing of 2
				sims[i, idxs] = neighborSims[idxs]
			else:
				sims[i] = neighborSims

	else:
		raise ValueError("Unrecognized similarity algorithm {}".format(
			similarityAlgo))

	if tryNumNeighbors > numNeighbors: # need to remove some neighbors
		# greedily take rows with most total similarity, but only counting
		# trivial matches once
		scores = np.zeros(len(sims))
		for i, row in enumerate(sims):
			maximaIdxs = nonOverlappingMaxima(row, length // 2)
			scores[i] = np.sum(row[maximaIdxs])
		sortIdxs = np.argsort(scores)[::-1]
		sims = sims[sortIdxs[:numNeighbors]]

	return sims.T


def neighborSims(X, length, numNeighbors, **kwargs):
	numSubseqs = len(X) - length + 1
	numOutputRows = numSubseqs
	numOutputCols = numNeighbors

	return applyToEachCol(neighborSims1D, X, numOutputRows,
		numOutputCols, length, numNeighbors, **kwargs)

# ================================================================ Quasi-ensembles

# 	fill='zero', *args, **kwargs):
def applyAtMultipleLengths(func, X, lengths, numFeatures, removeZeroRows=True,
	removeMostlyOnesRows=True, fill='median', *args, **kwargs):
	X = np.asarray(X)

	numLengths = len(lengths)
	maxLength = np.max(lengths)
	minLength = np.min(lengths)
	n = len(X)
	n_out = n - minLength + 1 # samples in output array

	if len(X.shape) > 1:
		numFeatures *= X.shape[1]

	allFeatures = np.zeros((numLengths, numFeatures, n_out))
	if fill == 'zero':
		allFeatures[:, :, :maxLength] = 0
		allFeatures[:, :, -maxLength:] = 0

	for i, length in enumerate(lengths):
		features = func(X, length, *args, **kwargs)
		if len(features.shape) > 2:
			features = hstack3Tensor(features)

		features = features.T
		featuresLen = features.shape[1]

		padLen = n_out - featuresLen
		prePadLen = int((padLen + 1) // 2)

		outRowsStart = prePadLen
		outRowsEnd = outRowsStart + featuresLen
		numFeatures = features.shape[0]
		allFeatures[i, :numFeatures, outRowsStart:outRowsEnd] = features

		if fill == 'median':
			# if almost everything was one value, fill with this value
			# instead of with 0; mostly useful so that features that are
			# always one don't artificially appear to be 0 at the boundaries
			medians = np.median(features, axis=1, keepdims=True)
			allFeatures[i, :numFeatures, :outRowsStart] = medians
			allFeatures[i, :numFeatures, outRowsEnd:] = medians

	out = vstack3Tensor(allFeatures)
	if removeZeroRows:
		out = ar.removeZeroRows(out)
	if removeMostlyOnesRows:
		means = np.mean(out > 0, axis=1)
		minorityOnesRows = np.where(means < .5)[0]
		out = out[minorityOnesRows]

	return out


def multiVariance(X, lengths, **kwargs):
	# numFeatures = len(breakpoints) + 1
	numFeatures = 8 # TODO allow setting breakpoints

	return applyAtMultipleLengths(sparseVariance, X, lengths, numFeatures, **kwargs)


def multiSparseLineProject(X, lengths, breakpoints, **kwargs):
	numFeatures = 2 * len(breakpoints) + 1

	return applyAtMultipleLengths(sparseLineProject, X, lengths, numFeatures,
		breakpoints=breakpoints, **kwargs)


def multiNormalize(X, lengths, **kwargs):
	numFeatures = 1
	return applyAtMultipleLengths(slidingNormalize, X, lengths, numFeatures,
		**kwargs)


def multiNormalizeAndSparseQuantize(X, lengths, breakpoints=None, **kwargs):
	if breakpoints is not None:
		numFeatures = len(breakpoints) + 1
	else:
		numFeatures = 8
	return applyAtMultipleLengths(normalizeAndSparseQuantize, X, lengths,
		numFeatures, breakpoints=breakpoints, **kwargs)


def multiSparseSaxHash(X, lengths, wordLen, cardinality, **kwargs):
	# TODO ideally, allow different wordLens and cardinalities across lengths
	numFeatures = cardinality ** wordLen
	return applyAtMultipleLengths(sparseSaxHashify, X, lengths, numFeatures,
		wordLen=wordLen, cardinality=cardinality, **kwargs)


def multiNeighborSims(X, lengths, numNeighbors, **kwargs):
	numFeatures = numNeighbors
	return applyAtMultipleLengths(neighborSims, X, lengths, numFeatures,
		# numNeighbors=numNeighbors, **kwargs)
		numNeighbors=numNeighbors, baseLength=np.max(lengths), **kwargs)


# ================================================================ Main

def neighborSimsMain():
	import matplotlib.pyplot as plt
	from ..datasets import datasets
	from ..viz import viz_utils as viz
	from ..utils import files
	from ..utils import arrays as ar
	from scipy import signal
	from scipy.ndimage import filters
	from ff2 import localMaxFilterSimMat

	saveDir = None
	# saveDir = 'figs/repr'
	if saveDir:
		files.ensureDirExists(saveDir)

	howMany = 4
	instPerTs = 2
	# np.random.seed(12345)
	np.random.seed(123)
	syntheticDatasets = ['triangles','shapes','rects','sines']
	# saveDir = 'figs/highlight/tidigits/'
	# tsList = datasets.loadDataset('triangles', whichExamples=range(howMany))
	# tsList = datasets.loadDatasets(syntheticDatasets, whichExamples=range(howMany))
	# tsList = datasets.loadDataset('tidigits_grouped_mfcc', whichExamples=range(howMany))
	tsList = datasets.loadDataset('ucr_short', whichExamples=range(1), instancesPerTs=instPerTs)
	# tsList = datasets.loadDataset('ucr_CBF', whichExamples=range(5), instancesPerTs=instPerTs)
	# tsList = datasets.loadDataset('ucr_CBF', whichExamples=range(2), instancesPerTs=instPerTs)
	# tsList = datasets.loadDataset('ucr_Gun_Point', whichExamples=range(2), instancesPerTs=instPerTs)
	# tsList = datasets.loadDataset('ucr_pairs', whichExamples=range(1))
	# tsList = datasets.loadDataset('ucr_adiac', whichExamples=range(1), instancesPerTs=instPerTs)
	# tsList = datasets.loadDataset('ucr_wafer', whichExamples=range(1), instancesPerTs=instPerTs)
	# tsList = datasets.loadDataset('ucr_MedicalImages', whichExamples=range(1), instancesPerTs=instPerTs)
	# tsList = datasets.loadDataset('ucr_ECG200', whichExamples=range(1), instancesPerTs=instPerTs)
	tsList = datasets.loadDataset('dishwasher_groups', whichExamples=range(howMany), instancesPerTs=5)
	# tsList = datasets.loadDataset('msrc', whichExamples=range(howMany))
	print '------------------------'
	# lengths = [16]
	# lengths = np.array([1./8])
	# lengths = np.array([1./16])
	# lengths = np.array([25])
	# lengths = np.array([1./32])
	# lengths = np.array([1./32, 1./16])
	# lengths = np.array([1./64, 1./32])
	# lengths = np.array([1./64, 1./32, 1./16])
	# lengths = np.array([1./128, 1./64, 1./32, 1./16])
	# lengths = np.array([50, 100])
	# lengths = np.array([50])
	# lengths = np.array([16])
	# lengths = np.array([8, 16, 32])
	# lengths = np.array([16, 32, 64])
	# lengths = np.array([8, 1./20, 1./10])
	# from ff8 import logSpacedLengths
	# lengths = 2 ** np.arange(4, 8) # 16,...,128
	lengths = 2 ** np.arange(3, 8) # 8,...,128
	# lengths = 2 ** np.arange(3, 8, .5) # 8,...,256, step by sqrt(2)

	for ts in tsList:
		print "running on ts ", ts.name

		# print "dim stds:", np.std(ts.data, axis=0)

		# ts.data = filters.convolve1d(ts.data, weights=np.hamming(8), axis=0, mode='constant')
		# ts.data = ts.data[:, 0]
		# ts.data = ts.data[:, [6,10]] # super large values in dishwasher
		# ts.data = ts.data[:, [5,9]] # dishwasher pattern dims
		# ts.data = ts.data[:, 5]
		# ts.data = ts.data[:, 9]
		if ts.data.shape[1] > 30: # if msrc, basically
			ts.data = ts.data[:, 23:38]
		ts.data = ar.ensure2D(ts.data)

		lens = np.copy(lengths)
		fractionIdxs = lengths < 1.
		lens[fractionIdxs] = lengths[fractionIdxs] * len(ts.data)
		lens = lens.astype(np.int)
		lens = np.unique(lens)
		lens = lens[lens >= 8]
		lens = lens[lens < len(ts.data) // 4]
		print "main(): using lengths", lens, lens.shape, lens.dtype
		# continue
		# lengths = np.array([25])

		plt.figure(figsize=(8, 10))
		gridShape = (9, 2)
		ax1 = plt.subplot2grid(gridShape, (0,0), rowspan=2, colspan=2)
		ax2 = plt.subplot2grid(gridShape, (2,0), rowspan=6, colspan=2)
		ax3 = plt.subplot2grid(gridShape, (8,0), rowspan=1, colspan=2)

		ts.plot(ax=ax1)
		# numNeighbors = np.max(lens)
		numNeighbors = np.log2(np.max(lens))
		# numNeighbors = int(np.log2(len(ts.data)))
		# numNeighbors = 10 * np.max(lens)
		# numNeighbors = len(ts.data) - np.min(lens) # "sample" everything
		# sims = multiNeighborSims(ts.data, lens, numNeighbors)
		sims = multiNeighborSims(ts.data, lens, numNeighbors, maxDist=.25)
		# print "after sims", ts.data.shape
		# sims = multiNeighborSims(ts.data, lens, numNeighbors, maxDist=.5)
		sims2 = multiVariance(ts.data, lens)
		# print "after variance", ts.data.shape
		# sims3 = multiNormalizeAndSparseQuantize(ts.data, lens)
		# diffs = ts.data[1:] - ts.data[:-1]
		# diffs = ar.addZeroRows(diffs, howMany=1, prepend=True) # same length as data
		# sims4 = multiVariance(diffs, lens)

		sims = np.vstack((sims, sims2))
		# sims = np.vstack((sims, sims2, sims3))
		# sims = np.vstack((sims, sims2, sims3, sims4))

		# from ff8 import buildFeatureMat
		# # sims = buildFeatureMat(ts.data, 1./16, 1./8, includeNeighbors=True,
		# sims = buildFeatureMat(ts.data, 1./16, 1./8, includeNeighbors=True,
		# 	detrend=False, maxDist=.25)

		# sims = multiNeighborSims(ts.data, lengths, numNeighbors, maxDist=.5)
		# print "nonzeros before localMaxFilter", np.count_nonzero(sims)
		# sims = localMaxFilterSimMat(sims) # makes it look much worse
		# print "nonzeros after localMaxFilter", np.count_nonzero(sims)
		maxFilterSims = localMaxFilterSimMat(sims)
		sums = np.sum(maxFilterSims, axis=1)
		# sums = np.sum(sims, axis=1)
		# print "keeping {} / {} rows".format(np.sum(sums > 1), sims.shape[0])
		# numFeaturesPossible = len(lens) * numNeighbors * ts.data.shape[1]
		numFeaturesPossible = np.sum(sums > 0) # pure zero reows only if empty
		print "keeping {} / {} rows".format(np.sum(sums > 1), numFeaturesPossible)
		sims = sims[sums > 1.]
		# sims = sims[np.mean(sims, axis=1) > np.max(lengths)]
		print "sims shape, stats", sims.shape, np.mean(sims), np.count_nonzero(sims)
		sims = ar.centerInMatOfSize(sims, -1, len(ts.data))
		viz.imshowBetter(sims, ax=ax2)

		colSums = np.sum(sims, axis=0)
		ax3.plot(colSums)
		ax3.set_xlim((0, len(colSums)))

		plt.tight_layout()
		if saveDir:
			savePath = os.path.join(saveDir, ts.name + '.pdf')
			plt.savefig(savePath)

	plt.show()


def main():
	neighborSimsMain()
	return

	import matplotlib.pyplot as plt
	from ..datasets import synthetic as synth

	n = 500
	m = 20
	# m = 12
	noiseStd = .5
	numSubseqs = n - m + 1
	# seq, _ = synth.sinesMotif()
	seq, _ = synth.multiShapesMotif()
	# seq = synth.notSoRandomWalk(n, std=noiseStd, trendFilterLength=80, lpfLength=16)

	plt.plot(seq)

	# ------------------------ running + sliding variance

	# variances = slidingVariance1D(seq, m)
	# trueVariances = np.empty(numSubseqs)
	# for i in range(numSubseqs):
	# 	trueVariances[i] = np.var(seq[i:i+m])

	# runningVariances = runningVariance1D(seq)
	# trueRunningVariances = np.zeros(n)
	# for i in range(2, n):
	# 	trueRunningVariances[i] = np.var(seq[:i])

	# # runningSSE = runningVariances * np.arange(1, n+1)
	# # trueRunningSSE = trueRunningVariances * np.arange(1, n+1)

	# # okay, so running variances are identical (offset of 1 to see both on plot)
	# # plt.plot(runningVariances+1, lw=2)
	# # plt.plot(trueRunningVariances, lw=4)

	# # yep, running SSE is, unsurprisingly, also identical
	# # plt.plot(runningSSE+300, lw=1)
	# # plt.plot(trueRunningSSE, lw=2)

	# # oops; this just straight up isn't valid math
	# # runningDiffs = runningSSE[m:] - runningSSE[:-m]
	# # plt.plot(runningDiffs / m)

	# # okay, this is working; hopefully tracking means in a numerically
	# # stable way is sufficient
	# plt.plot(variances + 2)
	# plt.plot(trueVariances, lw=2)

	# plt.show()

	# ------------------------ SAX
	# yep, everything looks good

	wordLen = 4
	# wordLen = 10
	cardinality = 4
	# cardinality = 8

	# saxWords = saxify(seq, m, wordLen, cardinality)
	# if len(saxWords.shape) > 2:
	# 	saxWords = hstack3Tensor(saxWords)

	# # plt.figure()
	# # plt.imshow(saxWords.T, interpolation='nearest', aspect='auto')
	# # plt.colorbar()

	# # sparseSax = sparsifySaxWords(saxWords, cardinality)
	# sparseSax = sparseSaxHashes(saxWords, cardinality)
	# sparseSax = ar.removeZeroCols(sparseSax)
	# # sparseSax = sparseSaxify(seq, m, wordLen, cardinality)

	# if len(sparseSax.shape) > 2:
	# 	sparseSax = hstack3Tensor(sparseSax)

	# plt.figure()
	# plt.imshow(sparseSax.T, interpolation='nearest', aspect='auto')
	# plt.show()


	# lengths = [m]
	lengths = [8, 16, 32, 64]
	multiSaxHash = multiSparseSaxHash(seq, lengths, wordLen, cardinality)

	plt.figure()
	plt.imshow(multiSaxHash, interpolation='nearest', aspect='auto')
	plt.show()

	# ------------------------ Line projection

	# # corrs = lineProject(seq, m)
	# # # plt.plot(corrs * 10) # scale to make it more visible
	# # plt.plot(corrs)

	# # variances = slidingVariance1D(seq, m)
	# # plt.plot(corrs * variances)

	# minVal, maxVal = np.min(seq), np.max(seq)
	# # valRange = maxVal - minVal
	# valRange = (maxVal - minVal)
	# # minVariance = .5 / np.sqrt(m)
	# minVariance = 1. / m
	# numBreakpoints = int(np.ceil(np.log2(valRange) - np.log2(minVariance)))
	# # numBreakpoints *= 2
	# breakpoints = np.logspace(np.log2(minVariance), np.log2(valRange), num=numBreakpoints, base=2.)

	# lengths = [4, 8, 16, 32]
	# features = multiSparseLineProject(seq, lengths, breakpoints)

	# plt.figure()
	# plt.imshow(features, interpolation='nearest', aspect='auto')

	# plt.show()

if __name__ == '__main__':
	main()





