#!/usr/env/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy import signal as sig

from numba import jit

from ..datasets import synthetic as synth
from ..utils import arrays as ar
from ..utils.subseq import optimalAlignK
from ..utils import subseq as sub
from ..utils import sliding_window as window
from ..viz import viz_utils as viz

from ff4 import computeSimMat
from ff3 import maxSubarray
import ff2

def embedExamples(seq, exampleLengths):

	n = len(seq)
	nInstances = len(exampleLengths)

	startIdxs = np.random.choice(np.arange(n / nInstances - max(exampleLengths)))
	startIdxs += np.arange(nInstances) * n // nInstances
	startIdxs = startIdxs.astype(np.int)

	for i in range(nInstances):
		if len(seq.shape) > 1:
			nDims = seq.shape[1]
			for dim in range(nDims):
				# inst = synth.sines(exampleLengths[i])
				# inst = synth.cylinder(exampleLengths[i])
				inst = synth.bell(exampleLengths[i])
				# inst = synth.funnel(exampleLengths[i])
				# inst = synth.warpedSeq(inst) * 5
				synth.embedSubseq(seq[:,dim], inst, startIdxs[i])
				# synth.embedSubseq(seq[:,dim], inst * 5, startIdxs[i])
		else:
			inst = synth.sines(exampleLengths[i])
			# inst = synth.cylinder(exampleLengths[i])
			# inst = synth.bell(exampleLengths[i])
			# inst = synth.funnel(exampleLengths[i])
			inst = synth.warpedSeq(inst) * 5
			# inst *= 5
			synth.embedSubseq(seq, inst, startIdxs[i])

	return seq

def vectorizeWindowLocs(X, windowLen):
	"""
	>>> A = np.array([[1,2,3], [4,5,6]])
	>>> vectorizeWindowLocs(A, 2) # doctest: +NORMALIZE_WHITESPACE
	array([[1, 2, 4, 5],
			[2, 3, 5, 6]])
	"""
	windowShape = (X.shape[0], windowLen)
	stride = np.array([0, 1]) # slide only along cols
	windows = window.sliding_window(X, windowShape, stride, flatten=True)
	flatWindows = windows.reshape((-1, np.product(windowShape)))
	return np.ascontiguousarray(flatWindows)

@jit
def computeIntersections(X, windowVects, windowLen):
	nLocs = X.shape[1] - windowLen + 1
	windowSz = windowVects.shape[1]
	nRows, nCols = X.shape

	# X = np.asfortranarray(X) # cols together in memory

	# colIntersections = np.zeros((nCols, nCols, nRows))
	# for i in range(nCols):
	# 	colIntersections[i, i] = X[:, i]
	# 	for j in range(i+1, nCols):
	# 		colIntersections[i, j] = np.logical_and(X[:, i], X[:, j])
	# 		colIntersections[j, i] = colIntersections[i, j]

	# intersections = np.zeros((nLocs, nLocs, windowSz))
	# for i in range(nLocs):
	# 	intersections[i, i] = windowVects[i]
	# 	for j in range(i+1, nLocs):
	# 		for k in range(windowLen):
	# 			startIdx = k*nRows
	# 			endIdx = (k+1) * nRows
	# 			firstColIdx = i + k
	# 			secondColIdx = j + k
	# 			intersections[i, j, startIdx:endIdx] = colIntersections[firstColIdx, secondColIdx]
	# 			intersections[j, i, startIdx:endIdx] = intersections[i, j, startIdx:endIdx]

	intersections = np.zeros((nLocs, nLocs, windowSz))
	for i in range(nLocs):
		intersections[i, i] = windowVects[i]
		for j in range(i+1, nLocs):
			intersection = np.logical_and(windowVects[i], windowVects[j])
			intersections[i, j] = intersection
			intersections[j, i] = intersection

	return intersections

def main():
	# np.random.seed(123)

	# ================================ consts for everything
	# consts for generating data
	# n = 1000
	n = 500
	# n = 300
	# length = 8
	# length = 16
	length = 32
	# length = 50
	# nInstances = 3
	exampleLengths = [55, 60, 65]
	# exampleLengths = [60, 60, 60]
	noiseStd = .5

	# consts for algorithm
	Lmin = max(20, length)	# only needed for optimalAlignK() spacing
	Lmax = 100				# loose upper bound on pattern length
	minSim = .8				# loose cutoff for what counts as similar

	k0 = len(exampleLengths) # for version where we tell it k

	# ------------------------ synthetic data

	# seq = synth.randconst(n, std=noiseStd)
	# seq = synth.notSoRandomWalk(n, std=noiseStd, trendFilterLength=80, lpfLength=4)
	seq = synth.notSoRandomWalk(n, std=noiseStd, trendFilterLength=80, lpfLength=16)
	seq = embedExamples(seq, exampleLengths)

	# ------------------------ msrc

	# from ..datasets import read_msrc as msrc
	# idxs = [2]
	# recordings = msrc.getRecordings(idxs=idxs)
	# r = list(recordings)[0]
	# seq = r.data[:, 20:23]
	# print "orig seq shape", seq.shape
	# seq = ar.downsampleMat(seq, rowsBy=10)
	# print "downsampled seq shape", seq.shape
	# length = 8
	# Lmin = len(seq) / 20
	# Lmax = len(seq) / 10
	# # Lmax = len(seq) / 20
	# k0 = 10
	# minSim = .5

	# noise = synth.randconst(seq.shape) # add noise for debugging
	# seq = np.r_[noise, seq, noise]

	# ================================ simMat

	X = computeSimMat(seq, length)
	# X[X < minSim] = 0.

	# X = ff2.localMaxFilterSimMat(X)
	# maxPoolWidth = min(length-1, Lmin-1)
	# maxPoolWidth /= 2
	# X = filters.maximum_filter1d(X, maxPoolWidth, axis=1)
	# X = filters.maximum_filter1d(X, length-1, axis=1)
	# X = filters.maximum_filter1d(X, length/2, axis=1)
	# X = filters.maximum_filter1d(X, 3, axis=1)
	# X = np.array(X > minSim, dtype=np.float)
	# X = X > minSim
	X[X < minSim] = 0.
	# X = ff2.filterSimMat(X, length-1, 'hamming', scaleFilterMethod='max1')

	# X = sub.removeCorrelatedRows(X, .9, accumulate=True) # correlation > .9 -> kill it
	# X = sub.removeCorrelatedRows(X, .9, accumulate=False) # correlation > .9 -> kill it

	print "simMat dims:", X.shape
	print "simMat nonzeros, total, frac = ", np.count_nonzero(X), X.size, np.count_nonzero(X) / float(X.size)

	# ================================ plotting crap

	plt.figure()
	axSeq = plt.subplot2grid((4,1), (0,0))
	axSim = plt.subplot2grid((4,1), (1,0), rowspan=3)
	for ax in (axSeq, axSim):
		ax.autoscale(tight=True)
	axSeq.plot(seq)
	axSim.imshow(synth.appendZeros(X, length-1), interpolation='nearest', aspect='auto')
	# im = axSim.imshow(synth.appendZeros(X, length-1), interpolation='nearest', aspect='auto')
	# plt.colorbar(im, cax=axSim)

	axSeq.set_title("Time Series")
	axSim.set_title("Similarities Matrix")

	# ================================ science

	# ------------------------ derived stats
	kMax = int(X.shape[1] / Lmin + .5)
	windowWidth = Lmax - length + 1
	# windowShape = (X.shape[0], Lmax)
	# windowSize = np.prod(windowShape)
	nLocs = X.shape[1] - windowWidth + 1

	# ------------------------ pairwise sims
	# colSims = np.dot(X.T, X)
	# filt = np.zeros((Lmax, Lmax)) + np.diag(np.ones(Lmax)) # zeros except 1s on diag
	# windowSims = sig.convolve2d(colSims, filt, mode='valid')

	print "computing intersections..."

	windowVects = vectorizeWindowLocs(X, windowWidth)
	windowSz = windowVects.shape[1]

	intersections = computeIntersections(X, windowVects, windowWidth)
	windowSims = np.sum(intersections, axis=2)
	# windowSims /= windowSz

	# assert(np.array_equal(windowSims, windowSims2)) # works

	# plt.figure()
	# plt.imshow(windowSims2)
	plt.figure()
	plt.imshow(windowSims / windowSz)
	plt.colorbar()

	# plt.show()
	# return

	print "computing similarity lower bound..."

	# TODO maybe try introducing beta prior to weight different values of k
	# 	-or, alternatively, see what happens if we tell it the right k

	#
	# Version where we look for stuff matching each intersection
	#

	# initialize with closest pair at least Lmin apart
	nonTrivialWindowSims = np.triu(windowSims) # zero lower half
	for i in range(nLocs):
		nonTrivialWindowSims[i, i:min(nLocs, i+Lmin)] = 0 # zero Lmin past diag
	highestSimIdx = np.argmax(nonTrivialWindowSims)

	bsfLocs = sorted([highestSimIdx // nLocs, highestSimIdx % nLocs])
	bsfScore = windowSims[tuple(bsfLocs)] * 2 # list will yield a list
	bsfIntersection = intersections[bsfLocs]

	print "finding best locations..."

	# rowIntersectionSims = np.zeros(nLocs)
	# for i in range(nLocs):
	# 	if i % 20 == 0:
	# 		print("computing stuff for row {}".format(i))
	# 	bestPossibleScores = windowSims[i,min(nLocs,i+Lmin):] * kMax / 2.
	# 	candidateIdxs = np.where(bestPossibleScores > bsfScore)[0]
	# 	# print "candidateIdxs shape", candidateIdxs.shape
	# 	for j in candidateIdxs:
	# 		intersection = intersections[i, j]
	# 		rowIntersectionSims *= 0
	# 		rowIntersectionSims[candidateIdxs] = np.dot(intersections[i, candidateIdxs], intersection)
	# 		idxs = sub.optimalAlignment(rowIntersectionSims, Lmin)

	# 		# order idxs by descending order of associated score
	# 		sizes = rowIntersectionSims[idxs]
	# 		sortedSizesOrder = np.argsort(sizes)[::-1]
	# 		sortedIdxs = idxs[sortedSizesOrder]

	# 		# iteratively intersect with another near neighbor, compute the
	# 		# associated score, and check if it's better (or if we can early abandon)
	# 		numIdxs = len(sortedIdxs)
	# 		k = 2
	# 		for idx in sortedIdxs[2:]: # first 2 are no better than orig intersection
	# 			k += 1
	# 			intersection = np.logical_and(intersection, intersections[i, idx])
	# 			sz = np.sum(intersection)
	# 			score = sz * k
	# 			if score > bsfScore:
	# 				print("window {0}, k={1}, score={2} is the new best!".format(i, k, score))
	# 				bsfScore = score
	# 				bsfLocs = sortedIdxs[:k]
	# 				bsfIntersection = np.copy(intersection)
	# 			elif sz * numIdxs <= bsfScore:
	# 				# print("early abandoning window {} at k={}".format(i, k))
	# 				break

	#
	# Version where we look for similarities to orig seq
	#
	bsfScore = 0
	bsfLocs = None
	bsfIntersection = None
	for i, row in enumerate(windowSims):
		if i % 20 == 0:
			print("computing stuff for row {}".format(i))
		# early abandon if this location has so little stuff that no
		# intersection with it can possibly beat the best score
		if windowSims[i,i] * kMax <= bsfScore: # highest score is kMax identical locs
			# print("immediately abandoning window {}!".format(i))
			continue
		# print("not abandoning window {}!".format(i))
		# best combination of idxs such that none are within Lmin of each other
		idxs = sub.optimalAlignment(row, Lmin)
		# order idxs by descending order of associated score
		sizes = windowSims[i, idxs]
		sortedSizesOrder = np.argsort(sizes)[::-1]
		sortedIdxs = idxs[sortedSizesOrder]
		# retrieve intersection and compute score for best 2 locs
		k = 2
		intersection = intersections[sortedIdxs[0], sortedIdxs[1]]
		score = windowSims[sortedIdxs[0], sortedIdxs[1]] * k
		# possibly update best-so-far score and window locations
		if score > bsfScore:
			bsfScore = score
			bsfLocs = sortedIdxs[:k]
			bsfIntersection = np.copy(intersection)
		# iteratively intersect with another near neighbor, compute the
		# associated score, and check if it's better (or if we can early abandon)
		numIdxs = len(sortedIdxs)
		for idx in sortedIdxs[2:]:
			k += 1
			intersection = np.logical_and(intersection, windowVects[idx])
			sz = np.count_nonzero(intersection)
			score = sz * k
			if score > bsfScore:
				print("window {0}, k={1}, score={2} is the new best!".format(i, k, score))
				bsfScore = score
				bsfLocs = sortedIdxs[:k]
				bsfIntersection = np.copy(intersection)
			# early abandon if this can't possibly beat the best score, which
			# is the case exactly when the intersection is so small that perfect
			# matches at all future locations still wouldn't be good enough
			elif sz * numIdxs <= bsfScore:
				# print("early abandoning window {} at k={}".format(i, k))
				break

	#
	# Version where we we tell it k
	#
	# bsfScore = 0
	# bsfLocs = None
	# bsfIntersection = None
	# # selfSims = np.diagonal(windowSims)
	# # candidateRowIdxs = np.where(selfSims * k0 <= bsfScore)[0]
	# # for i in candidateRowIdxs:
	# # 	row = windowSims[i]
	# for i, row in enumerate(windowSims):
	# 	if i % 20 == 0:
	# 		print("computing stuff for row {}".format(i))
	# 	if windowSims[i,i] * k0 <= bsfScore:
	# 		continue
	# 	idxs = sub.optimalAlignK(row, Lmin, k0)
	# 	intersection = intersections[idxs[0], idxs[1]]
	# 	sz = 0
	# 	for idx in idxs[2:]:
	# 		intersection = np.logical_and(intersection, windowVects[idx])
	# 		sz = np.count_nonzero(intersection)
	# 		if sz * k0 <= bsfScore:
	# 			break
	# 	score = sz * k0
	# 	if score > bsfScore:
	# 		print("window {0}, k={1}, score={2} is the new best!".format(i, k0, score))
	# 		bsfScore = score
	# 		bsfLocs = idxs
	# 		bsfIntersection = np.copy(intersection)

	# ================================ show output

	print "bestScore = {}".format(bsfScore)
	print "bestLocations = {}".format(str(bsfLocs))

	for idx in bsfLocs:
		viz.plotRect(axSim, idx, idx+windowWidth)

	bsfIntersectionWindow = bsfIntersection.reshape((-1, windowWidth))
	sums = np.sum(bsfIntersectionWindow, axis=0)
	print bsfIntersectionWindow.shape
	print sums.shape

	plt.figure()
	plt.imshow(bsfIntersectionWindow, interpolation='nearest', aspect='auto')
	plt.colorbar()

	plt.figure()
	plt.plot(sums)

	p0 = np.mean(X)
	kBest = len(bsfLocs)
	p0 = np.power(p0, kBest)
	# expectedOnesPerCol = p0 * X.shape[1]
	expectedOnesPerCol = p0 * X.shape[1] * 2
	sums -= expectedOnesPerCol

	plt.plot(sums)

	start, end, _ = maxSubarray(sums)
	patStart, patEnd = start, end + 1 + length
	viz.plotRect(plt.gca(), start, end + 1)
	for idx in bsfLocs:
		viz.plotRect(axSeq, idx + patStart, idx + patEnd)


	# plt.figure()
	# # windowSims[bestRowIdx, bestColIdxs] *= 10 # color these differently
	# # plt.imshow(windowSims, interpolation='none')
	# plt.imshow(windowSims)
	# plt.colorbar()

	# # for col in range(colSims.shape[1]):
	# bestRowIdx = -1
	# bestColIdxs = []
	# bestSum = -1
	# kVals = np.arange(2,kMax)
	# for i, row in enumerate(windowSims):
	# 	optimalIdxs = sub.optimalAlignK(row, Lmin, kVals)
	# 	if not len(optimalIdxs):
	# 		continue
	# 	# print "optimalIdxs", optimalIdxs
	# 	sums = map(lambda idxs: np.sum(row[idxs]), optimalIdxs)
	# 	# print "sums", sums
	# 	sums = np.asarray(sums)
	# 	bestSumIdx = np.argmax(sums)
	# 	if sums[bestSumIdx] > bestSum:
	# 		bestRowIdx = i
	# 		bestColIdxs = optimalIdxs[bestSumIdx]
	# 		bestSum = sums[bestSumIdx]

	# print "bestRow = ", bestRowIdx
	# print "best end locs = ", bestColIdxs

	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()

	# from doctest import testmod
	# testmod()
