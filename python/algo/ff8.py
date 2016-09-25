#!/usr/env/python

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy import stats
from scipy.ndimage import filters

from ..datasets import synthetic as synth
from ..datasets import read_msrc as msrc
from ..utils import arrays as ar
from ..utils import subseq as sub
from ..viz import viz_utils as viz

import representation as rep

from ff2 import localMaxFilterSimMat, filterSimMat
from ff3 import maxSubarray
from ff5 import embedExamples
from ff6 import vectorizeWindowLocs


# ================================================================ Data Loading

def randWalkSeq(n=500, exampleLengths=[55, 60, 65], noiseStd=.5):
	seq = synth.randwalk(n, std=noiseStd)
	return embedExamples(seq, exampleLengths)


def notSoRandWalkSeq(n=500, exampleLengths=[55, 60, 65], noiseStd=.5):
	seq = synth.notSoRandomWalk(n, std=noiseStd, trendFilterLength=80, lpfLength=16)
	return embedExamples(seq, exampleLengths)


def extendSeq(seq, prePadLen, postPadLen):
	first = np.tile(seq[0], (prePadLen, 1))
	last = np.tile(seq[-1], (postPadLen, 1))
	return np.vstack((first, seq, last))


def msrcSeq(whichIdx=0, downsampleBy=2, whichDims=[24,25,26], lMinFrac=.05, lMaxFrac=.1):
	recordings = msrc.getRecordings(idxs=[whichIdx])
	r = list(recordings)[0]
	seq = r.data[:, whichDims]
	seq = ar.downsampleMat(seq, rowsBy=downsampleBy)
	n = len(seq)
	Lmin = int(n * lMinFrac)
	Lmax = int(n * lMaxFrac)
	# length = Lmin // 2
	length = Lmin
	answerIdxs = r.gestureIdxs / downsampleBy

	# add padding
	prePadLen = Lmax - length
	postPadLen = Lmax - length
	seq = extendSeq(seq, prePadLen, postPadLen)
	# first = np.tile(seq[0], (prePadLen, 1))
	# last = np.tile(seq[-1], (postPadLen, 1))
	# seq = np.vstack((first, seq, last))
	answerIdxs += prePadLen

	return seq, answerIdxs, Lmin, Lmax


# ================================================================ Feature Mat

def logSpacedLengths(Lmin, Lmax, logStep=1., round=True, ints=True):
	logMaxLength = np.log2(Lmax)
	logMinLength = np.log2(Lmin)
	if round:
		logMaxLength = int(np.floor(logMaxLength))
		logMinLength = int(np.floor(logMinLength))
	lengths = np.arange(logMinLength, logMaxLength + logStep, logStep)
	lengths = (2. ** lengths)
	if ints:
		lengths = lengths.astype(np.int)
	return lengths


def localZnormMat(seq, lengths, cardinality=8):
	# lengths = logSpacedLengths(Lmin, Lmax)
	breakpoints = rep.saxBreakpoints(cardinality)
	# X = rep.multiSparseLineProject(seq, lengths, breakpoints, removeZeroRows=False)
	X = rep.multiNormalizeAndSparseQuantize(seq, lengths, breakpoints)
	return X


def localSlopeMat(seq, lengths, cardinality=4, ignoreFlat=True, **kwargs):
	if ignoreFlat:
		cardinality += 1 # lowest level lost if ignoring 0
	breakpoints = rep.defaultSparseLineBreakpoints(seq, cardinality=cardinality)
	X2 = rep.multiSparseLineProject(seq, lengths, breakpoints,
		ignoreFlat=ignoreFlat, **kwargs)
	X2 = X2 > 0. # ignore correlations
	return X2.astype(np.float64)


def saxHashMat(seq, lengths, wordLen, cardinality):
	return rep.multiSparseSaxHash(seq, lengths, wordLen, cardinality)


def defaultLengths(Lmin, Lmax):
	lengths = logSpacedLengths(8, Lmax, round=True)
	if len(lengths) == 0:
		lengths = [8, 16]
	elif len(lengths) == 1:
		lengths = np.array([lengths[0] // 2, lengths[0]])
	return lengths


def neighborsMat(seq, Lmin, Lmax, **kwargs):
	# lengths = logSpacedLengths(16, Lmax, round=True)

	lengths = defaultLengths(Lmin, Lmax)
	numNeighbors = int(np.log2(len(seq)))
	# tryNumNeighbors = numNeighbors * numNeighbors # try log(n)^2 neighbors
	tryNumNeighbors = numNeighbors
	X = rep.multiNeighborSims(seq, lengths, numNeighbors,
		tryNumNeighbors=tryNumNeighbors, **kwargs)
	sums = np.sum(X, axis=1)
	# print "neighborsMat: keeping {} / {} rows".format(
	# 	np.sum(sums > 1), X.shape[0])
	# return X[sums > 1.] # features that happen more than once
	X = X[sums > 1.] # features that happen more than once, binarized
	return (X > 0).astype(np.float64)


def varianceMat(seq, Lmin, Lmax):
	lengths = defaultLengths(Lmin, Lmax)
	return rep.multiVariance(seq, lengths)


def preprocessFeatureMat(X, Lfilt, logX=False, logXblur=False, capXblur=True, **sink):
	# if not capXblur:
	# 	X = localMaxFilterSimMat(X)
	# X = localMaxFilterSimMat(X)

	# Lfilt *= 2 # TODO remove after test
	featureMeans = np.mean(X, axis=1).reshape((-1, 1))
	if logX and logXblur:
		X *= -np.log2(featureMeans) # variable encoding costs for rows
		Xblur = filterSimMat(X, Lfilt, 'hamming', scaleFilterMethod='max1')
	if logX and not logXblur:
		Xblur = filterSimMat(X, Lfilt, 'hamming', scaleFilterMethod='max1')
		X *= -np.log2(featureMeans)
	if not logX and logXblur:
		Xblur = filterSimMat(X, Lfilt, 'hamming', scaleFilterMethod='max1')
		Xblur *= np.log2(featureMeans)
	if not logX and not logXblur:
		Xblur = filterSimMat(X, Lfilt, 'hamming', scaleFilterMethod='max1')
		# Xblur = filterSimMat(X, Lfilt, 'flat', scaleFilterMethod='max1')

	if capXblur: # don't make long stretches also have large values
		maxima = filters.maximum_filter1d(Xblur, Lfilt // 2, axis=1, mode='constant')
		# maxima = filters.maximum_filter1d(Xblur, Lfilt, axis=1, mode='constant')
		Xblur[maxima > 0] /= maxima[maxima > 0]
		# print "preprocessFeatureMat(): max value in Xblur: ", np.max(Xblur) # 1.0
		# import sys
		# sys.exit()
		# Xblur = np.minimum(Xblur, 1.)

	# plt.figure()
	# viz.imshowBetter(X)
	# plt.figure()
	# viz.imshowBetter(Xblur)

	# have columns be adjacent in memory
	X = np.asfortranarray(X)
	Xblur = np.asfortranarray(Xblur)

	# assert(np.all((X > 0.) <= (Xblur > 0.)))

	assert(np.all(np.sum(X, axis=1) > 0))
	assert(np.all(np.sum(Xblur, axis=1) > 0))

	# if np.max(X) > 1.:
	# 	print "X min"

	print "preprocessFeatureMat(): X shape, logX, logXblur", X.shape, logX, logXblur
	print "preprocessFeatureMat(): Lfilt", Lfilt
	print "preprocessFeatureMat(): X min, max", X.min(), X.max()
	print "preprocessFeatureMat(): Xblur min, max", Xblur.min(), Xblur.max()

	# import sys
	# sys.exit()

	return X, Xblur


def buildFeatureMat(seq, Lmin, Lmax, cardinality=4,
	saxWordLen=4, saxCardinality=4,
	includeLocalZnorm=False, includeLocalSlope=False, includeSaxHashes=False,
	includeMaxFilteredSlope=False, includeNeighbors=False, includeVariance=False,
	detrend=False, ignoreFlat=True, **neighborsKwargs):

	# TODO this definitely needs to be zero-padded at the ends so that
	# we can find stuff at the ends

	if not any((includeLocalZnorm, includeLocalSlope, includeSaxHashes,
		includeMaxFilteredSlope, includeNeighbors)):
		raise ValueError("No features requested!")

	if Lmin < 1. or Lmax < 1.:
		Lmin = int(len(seq) * Lmin)
		Lmax = int(len(seq) * Lmax)

	if detrend: # remove overall trend in ts so slope amplitudes don't get hosed
		for i in range(seq.shape[1]):
			col = np.copy(seq[:, i]).flatten()
			x = np.arange(len(col))
			col -= np.mean(col)
			slope, intercept, _, _, _ = stats.linregress(x, col)
			col -= x * slope
			seq[:, i] = col

	# minLen = min(Lmin * 2, Lmax)
	# minLen = min(Lmin, Lmax)
	minLen = min(Lmin // 2, Lmax)
	minLen = max(minLen, 8) # don't go below 8, though
	lengths = logSpacedLengths(minLen, Lmax) # lengths within (Lmin, Lmax]
	if len(lengths) == 1: # ensure we use at least 2 lengths
		lengths = np.array([lengths[0] // 2, lengths[0]])

	print "buildFeatureMat(): seqLen {}; using lengths: {}".format(len(seq), lengths)

	Xs = []
	if includeLocalZnorm:
		Xs.append(localZnormMat(seq, lengths, cardinality))
	if includeLocalSlope:
		Xs.append(localSlopeMat(seq, lengths,
			cardinality=cardinality, ignoreFlat=ignoreFlat))
	if includeMaxFilteredSlope:
		Xs.append(localSlopeMat(seq, lengths,
			cardinality=cardinality, maxFilter=True, ignoreFlat=ignoreFlat))
	if includeSaxHashes:
		Xs.append(saxHashMat(seq, lengths, saxWordLen, saxCardinality))
	if includeNeighbors:
		Xs.append(neighborsMat(seq, Lmin, Lmax, **neighborsKwargs))
	if includeVariance:
		Xs.append(varianceMat(seq, Lmin, Lmax))
	X = np.vstack(Xs)

	return X


def plotSeqAndFeatures(seq, X, createFiltAx=False, padBothSides=False, capYLim=1000):
	"""plots the time series above the associated feature matrix"""
	plt.figure(figsize=(10, 8))
	if createFiltAx:
		nRows = 4
		nCols = 7
		axSeq = plt.subplot2grid((nRows,nCols), (0,0), colspan=(nCols-1))
		axSim = plt.subplot2grid((nRows,nCols), (1,0), colspan=(nCols-1), rowspan=(nRows-1))
		axFilt = plt.subplot2grid((nRows,nCols), (1,nCols-1), rowspan=(nRows-1))
		axes = (axSeq, axSim, axFilt)
	else:
		nRows = 4
		nCols = 1
		axSeq = plt.subplot2grid((nRows,nCols), (0,0))
		axSim = plt.subplot2grid((nRows,nCols), (1,0), rowspan=(nRows-1))
		axes = (axSeq, axSim)

	for ax in axes:
		ax.autoscale(tight=True)

	axSeq.plot(seq)
	axSeq.set_ylim([seq.min(), min(capYLim, seq.max())])

	if padBothSides:
		padLen = (len(seq) - X.shape[1]) // 2
		Xpad = ar.addZeroCols(X, padLen, prepend=True)
		Xpad = ar.addZeroCols(Xpad, padLen, prepend=False)
	else:
		padLen = len(seq) - X.shape[1]
		Xpad = ar.addZeroCols(Xpad, padLen, prepend=False)
	axSim.imshow(Xpad, interpolation='nearest', aspect='auto')

	axSeq.set_title("Time Series")
	axSim.set_title("Feature Matrix")

	if createFiltAx:
		axFilt.set_title("Learned Filter")
		return axSeq, axSim, axFilt
	return axSeq, axSim


def learnFF(X, Xblur, Lmin, Lmax, length):
	"""main algorithm"""

	# ------------------------ derived stats
	kMax = int(X.shape[1] / Lmin + .5)
	# windowLen = Lmax - length + 1
	windowLen = Lmax # try matching ff10

	print "using window len ", windowLen

	p0 = np.mean(X) # fraction of entries that are 1 (roughly)
	# p0 = np.mean(X > 0.) # fraction of entries that are 1 # TODO try this
	# p0 = 2 * np.mean(X > 0.) # lambda for l0 reg based on features being bernoulli at 2 locs
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

	print "p0, noiseSz = ", p0, noiseSz

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
		if i % 10 == 0:
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


def plotExtractedStuff(axSeq, axSim, answerIdxs, bsfLocs, bsfIntersectionWindow,
	start, end, axFilt=None, filt=None):
	# patStart, patEnd = start, end + 1 + length
	patStart, patEnd = start, end + 1
	windowLen = bsfIntersectionWindow.shape[1]

	for idx in bsfLocs:
		viz.plotRect(axSim, idx, idx+windowLen)

	for idx in bsfLocs:
		viz.plotRect(axSeq, idx + patStart, idx + patEnd)
		print "inst: ", idx + patStart, idx + patEnd

	if answerIdxs is not None:
		for idx in answerIdxs:
			viz.plotVertLine(idx, ax=axSeq)

	plt.figure()
	plt.imshow(bsfIntersectionWindow, interpolation='nearest', aspect='auto')

	plt.tight_layout()
	plt.show()


def learnFFfromSeq(seq, Lmin, Lmax, Lfilt=0, extendEnds=True, **kwargs):
	# if Lmin < 1. or Lmax < 1. or Lfilt < 1.:
	# 	Lmin = int(len(seq) * Lmin)
	# 	Lmax = int(len(seq) * Lmax)
	# 	Lfilt = int(len(seq) * Lfilt)
	Lmin = int(len(seq) * Lmin) if Lmin < 1. else Lmin
	Lmax = int(len(seq) * Lmax) if Lmax < 1. else Lmax
	Lfilt = int(len(seq) * Lfilt) if Lfilt < 1. else Lfilt

	if not Lfilt:
		# Lfilt = Lmin // 2 - 1
		Lfilt = Lmin

	print "using Lmin, Lmax, Lfilt= {}, {}, {}".format(Lmin, Lmax, Lfilt)

	# extend the first and last values out so that features using
	# longer windows are present for more locations (if requested)
	padLen = 0
	origSeqLen = len(seq)
	if extendEnds:
		# padLen = Lmax - Lfilt # = windowLen
		padLen = Lmax # match ff10 (ignoring step); wider window
		seq = extendSeq(seq, padLen, padLen)

	X = buildFeatureMat(seq, Lmin, Lmax, **kwargs)
	X, Xblur = preprocessFeatureMat(X, Lfilt)
	if extendEnds: # undo padding after constructing feature matrix
		X = X[:, padLen:-padLen]
		Xblur = Xblur[:, padLen:-padLen]
		seq = seq[padLen:-padLen]

	print "sums:", np.sum(seq), np.sum(X), np.sum(Xblur)

	bsfLocs, bsfIntersectionWindow, start, end = learnFF(X, Xblur, Lmin, Lmax, Lfilt+1)

	bsfLocs = np.asarray(bsfLocs)

	startIdxs = bsfLocs + start
	# endIdxs = bsfLocs + end + Lfilt
	endIdxs = bsfLocs + end + 1

	# account for fact that X is computed based on middle of data
	# TODO does this actually improve accuracy? EDIT: no, this is a huge shift
	print "seq len, X len, Lmin", origSeqLen, X.shape[1], Lmin
	offset = (origSeqLen - X.shape[1]) // 2
	startIdxs += offset
	endIdxs += offset

	# if it includes the padding, disallow it
	# keepTheseStarts = np.where(startIdxs >= 0)[0]
	# keepTheseEnds = np.where(endIdxs <= len(seq))[0]
	# keepIdxs = keepTheseStarts * keepTheseEnds
	# startIdxs = startIdxs[keepIdxs]
	# endIdxs = endIdxs[keepIdxs]

	print("used padLen = {}".format(padLen))

	return startIdxs, endIdxs, bsfIntersectionWindow, X, Xblur


def main():
	# Lmin, Lmax = 20, 100
	# seq = randWalkSeq()
	# seq = notSoRandWalkSeq()

	idx = 10
	seq, answerIdxs, Lmin, Lmax = msrcSeq(idx)

	Lfilt = Lmin // 2 - 1
	print "main(): using Lmin, Lmax, Lfilt = {}, {}, {}".format(Lmin, Lmax, Lfilt)
	X = buildFeatureMat(seq, Lmin, Lmax)
	X, Xblur = preprocessFeatureMat(X, Lfilt)
	axSeq, axSim = plotSeqAndFeatures(seq, X)

	print "sums:", np.sum(seq), np.sum(X), np.sum(Xblur)
	# return # hmm...getting different features at the moment...

	bsfLocs, bsfIntersectionWindow, start, end = learnFF(X, Xblur, Lmin, Lmax, Lfilt+1)
	plotExtractedStuff(axSeq, axSim, answerIdxs, bsfLocs, bsfIntersectionWindow, start, end)


if __name__ == '__main__':
	main()


















