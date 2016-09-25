#!/usr/env/python

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.ndimage import filters
from scipy import signal as sig

from numba import jit

from ..datasets import synthetic as synth
from ..utils import arrays as ar
from ..utils.subseq import optimalAlignK
from ..utils import subseq as sub
from ..utils import sliding_window as window
from ..viz import viz_utils as viz

from ff5 import embedExamples
from ff3 import maxSubarray
import ff2


def computeSimMat(seq, length=8, full=False):
	print("computing simMat on seq of shape {}".format(str(seq.shape)))

	# TODO have it iteratively add dict seqs, rather than using all pairs

	nDims = 1
	nSubseqs = len(seq) - length + 1
	# ND seq; call for each dim separately, then combine resulting simMats
	if len(seq.shape) > 1 and seq.shape[1] > 1:
		# print "it's an nd seq!, nDims = ", seq.shape[1]
		nDims = seq.shape[1]
		mats = np.zeros((nDims, nSubseqs, nSubseqs))
		for dim in range(nDims):
			data = seq[:, dim]
			if np.var(data) < .01: # ignore flat dims
				continue
			# compute sims for each dimension
			mats[dim] = computeSimMat(data, length)

			# plt.figure()
			# mats[dim][mats[dim] < .75] = 0.
			# plt.imshow(mats[dim] > .75)
			# plt.imshow(mats[dim])

		# stack sim mats for each dim on top of one another (ie, append rows)
		sims = mats.reshape((-1, mats.shape[2]))
		print "number of nonzero rows:", len(ar.nonzeroRows(sims))
		sims = ar.removeZeroRows(sims)
		print "number of nonzero rows:", len(ar.nonzeroRows(sims))
		return sims

	# seqRange = np.max(seq) - np.min(seq)
	# seqRange = 1.

	# 1D seq; actually compute similarities
	sims = np.zeros((nSubseqs, nSubseqs))
	# sqrtLength = 2 * np.sqrt(length)
	for i in range(nSubseqs):
		x = np.copy(seq[i:(i+length)])
		# x = (x - np.mean(x)) / np.std(x)
		x -= np.mean(x)
		varX = np.var(x)
		# x /= seqRange
		# x /= np.linalg.norm(x)
		# x /= np.sqrt(np.sum(x*x))
		# print x.shape, x.mean(), x.std() * np.sqrt(len(x))
		for j in range(i+1,nSubseqs):
			y = np.copy(seq[j:(j+length)])
			# y = (y - np.mean(y)) / np.std(y)
			y -= np.mean(y)
			varY = np.var(y)
			# y /= seqRange
			# y /= np.linalg.norm(y)
			# y /= np.sqrt(np.sum(y*y))
			# sims[i,j] = np.dot(x, y)
			diff = x - y
			# dist = np.dot(diff, diff) / varX
			dist = np.dot(diff, diff) / min(varX, varY)
			sims[i, j] = 1. - np.sqrt(dist / length)

	# fill in lower triangle
	sims += sims.T

	# populate diagonal with 2nd-best match
	for i in range(nSubseqs):
		sims[i,i] = np.max(sims[i]) - .0001 # offset to not mess up rel maxima

	# assert(np.min(sims) >= -1.00001)
	maxSim = np.max(sims)
	if maxSim >= 1.00001:
		print "max is too high: ", maxSim
		print "happens at: ", zip(np.where(sims >= maxSim))
		assert(maxSim <= 1.00001)

	return sims


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


def computeIntersections(X, windowLen):

	nLocs = X.shape[1] - windowLen + 1
	nRows, nCols = X.shape
	X = np.asfortranarray(X) # cols together in memory

	print "computing col sims"

	# Xs = sparse.csc_matrix(X)
	# colIntersections = np.zeros((nCols, nCols, nRows))
	colIntersections = np.zeros((nCols, nCols), dtype=np.object)
	for i in range(nCols):
		# colIntersections[i, i] = sparse.csr_matrix(X[:, i], dtype=np.bool)
		for j in range(i, nCols):
			intersection = np.logical_and(X[:, i], X[:, j])
			# xi = Xs[:, i]
			# xj = Xs[:, j]
			# intersection = (xi > 0)
			# intersection = (intersection == xj)
			# intersection = sparse.csr_matrix(intersection)
			colIntersections[i, j] = intersection
			# colIntersections[j, i] = intersection

	# print colIntersections[3, 7] # what does this look like?
	# print colIntersections[3, 7].shape # what does this look like?
	# print colIntersections[3, 7].__class__ # what does this look like?

	print "computing window sims"

	windowSz = X.shape[0] * windowLen

	# intersections = np.zeros((nLocs, nLocs, windowSz))
	intersections = np.zeros((nLocs, nLocs), dtype=np.object)
	firstIdxs = np.arange(windowLen, dtype=np.int) - 1
	secondIdxs = np.arange(windowLen, dtype=np.int)
	for i in range(nLocs):
		firstIdxs += 1
		for j in range(i, nLocs):
			secondIdxs[:] = firstIdxs + j - i
			combinedIdxs = (firstIdxs, secondIdxs)
			# intersection = sparse.hstack(colIntersections[combinedIdxs])
			intersection = np.hstack(colIntersections[combinedIdxs])
			intersection = sparse.csr_matrix(intersection)
			intersections[i, j] = intersection
			intersections[j, i] = intersection

			# intersectionsList = []
			# for k in range(windowLen):
			# 	firstColIdx = i + k
			# 	secondColIdx = j + k
			# 	intersectionsList.append(colIntersections[firstColIdx, secondColIdx])
			# # print intersectionsList
			# # print map(lambda x: x.shape, intersectionsList)
			# # intersectionsList = map(lambda x: x.reshape((1, -1)), intersectionsList)
			# # print map(lambda x: x.shape, intersectionsList)
			# # print map(lambda x: x.__class__, intersectionsList)
			# intersection = sparse.hstack(intersectionsList)
			# intersections[i, j] = intersection
			# intersections[j, i] = intersection

	print "computing brute force version"

	windowVects = vectorizeWindowLocs(X, windowLen)

	intersections2 = np.zeros((nLocs, nLocs, windowSz))
	for i in range(nLocs):
		# intersections2[i, i] = windowVects[i].astype(np.bool)
		for j in range(i, nLocs):
			intersection = np.logical_and(windowVects[i], windowVects[j])
			intersections2[i, j] = intersection
			# intersections2[j, i] = intersection

	print "computing convolution version"

	# colSims = np.dot(X.T, X)
	# filt = np.zeros((windowLen, windowLen)) + np.diag(np.ones(windowLen)) # zeros except 1s on diag
	# windowSims = sig.convolve2d(colSims, filt, mode='valid')
	# windowSims2 = np.sum(intersections2, axis=2)
	# assert(np.allclose(windowSims, windowSims2))

	# plt.figure()
	# plt.imshow(windowSims)
	# plt.figure()
	# plt.imshow(windowSims2)
	# sums = np.empty((nLocs, nLocs))
	# # print sums.shape
	# for i in range(nLocs):
	# 	for j in range(i, nLocs):
	# 		intersection = intersections[i, j]
	# 		# intersection = colIntersections[i, j]
	# 		sums[i, j] = intersection.nnz
	# 		# sum = intersection.nnz
	# 		# print i, j, sum
	# 		# sums[i, j] = 7
	# plt.figure()
	# plt.imshow(sums)

	# plt.show()
	# return

	# print "checking if crap is equal"

	# for i in range(nLocs):
	# 	for j in range(i, nLocs):
	# 		int1 = intersections[i, j]
	# 		# int1 = intersections[i, j].toarray().flatten()
	# 		int2 = intersections2[i, j]
	# 		# int2 = sparse.csr_matrix(int2, dtype=np.bool)
	# 		# eq = (int1 != int2).nnz == 0
	# 		# eq = np.array_equal(int1, int2)
	# 		# eq = (int1.nnz == np.count_nonzero(int2))
	# 		nz2 = windowSims2[i, j]
	# 		nz = windowSims[i, j] # wrong or something
	# 		eq = (int1.nnz == nz2)
	# 		# print eq, nz
	# 		# eq = eq and (int1.nnz == nz)
	# 		if not eq:
	# 			print "i, j = ", i, j
	# 			print "sum1, sum2, trueSum = ", \
	# 				int1.nnz, np.count_nonzero(int2), nz
	# 			print "shape1, shape2 = ", int1.shape, int2.shape
	# 			# diffs = int1 != int2
	# 			# print "differences at ", np.where(diffs)[0]
	# 			# print "1 nonzero idxs", np.where(int1)[0]
	# 			# print "2 nonzeros idxs", np.where(int2)[0]
	# 			# print "vect1 ="
	# 			# print int1
	# 			# print "vect2 ="
	# 			# print int2
	# 			assert(False)

	# moral of the story--use .nnz, rather than np.count_nonzero(), or it
	# will freak the **** out

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
	minSim = .5				# loose cutoff for what counts as similar

	# k0 = len(exampleLengths) # for version where we tell it k
	answerIdxs = None

	# ------------------------ synthetic data

	# seq = synth.randconst(n, std=noiseStd)
	# seq = synth.notSoRandomWalk(n, std=noiseStd, trendFilterLength=80, lpfLength=4)
	seq = synth.notSoRandomWalk(n, std=noiseStd, trendFilterLength=80, lpfLength=16)
	seq = embedExamples(seq, exampleLengths)

	# ------------------------ msrc

	from ..datasets import read_msrc as msrc
	idxs = [2]
	# idxs = [0]
	downsampleBy = 2
	recordings = msrc.getRecordings(idxs=idxs)
	r = list(recordings)[0]
	# seq = r.data
	# seq = r.data[:, :40]
	# seq = r.data[:, 20:23]
	seq = r.data[:, 24:27]
	# seq = r.data[:, 20:27]
	print "orig seq shape", seq.shape
	seq = ar.downsampleMat(seq, rowsBy=downsampleBy)
	print "downsampled seq shape", seq.shape
	length = max(8, Lmin / 2)
	Lmin = len(seq) / 20
	Lmax = len(seq) / 10
	# Lmax = len(seq) / 20
	# k0 = 10
	minSim = .5
	answerIdxs = r.gestureIdxs / downsampleBy
	# print "seq shape", seq.shape
	prePadLen = Lmax - length
	postPadLen = length - 1
	first = np.tile(seq[0], (prePadLen, 1))
	last = np.tile(seq[-1], (postPadLen, 1))
	seq = np.vstack((first, seq, last)) # pad with fixed val to allow all window positions
	# ^ TODO pad simMat with zeros instead--this introduces fake subseqs
	answerIdxs += prePadLen
	# seq = np.vstack((seq, np.tile(flat, (length-1, 1)))) # lets it get the last rep
	# print "seq shape", seq.shape


	# r.plot()

	# plt.figure()
	# plt.plot(r.sampleTimes)

	# answerIdxs = r.gestureIdxs / downsampleBy
	# print r.gestureIdxs
	# print answerIdxs

	# plt.figure()
	# plt.plot(seq)
	# for idx in answerIdxs:
	# 	ax = plt.gca()
	# 	viz.plotVertLine(idx, ax=ax)
	# plt.show()

	# return

	# noise = synth.randconst(seq.shape) # add noise for debugging
	# seq = np.r_[noise, seq, noise]

	# ================================ simMat

	X = computeSimMat(seq, length)
	X[X < minSim] = 0.
	# Xorig = np.copy(X)
	X = ff2.localMaxFilterSimMat(X)
	Xblur = ff2.filterSimMat(X, length-1, 'hamming', scaleFilterMethod='max1')
	# Xblur = ff2.filterSimMat(X, Lmin-1, 'hamming', scaleFilterMethod='max1')
	Xblur = np.minimum(Xblur, 1.)

	print "simMat dims:", X.shape
	Xnonzeros = np.count_nonzero(X)
	print "simMat nonzeros, total, frac = ", Xnonzeros, X.size, Xnonzeros / float(X.size)

	# ================================ plotting crap

	plt.figure()
	axSeq = plt.subplot2grid((4,1), (0,0))
	axSim = plt.subplot2grid((4,1), (1,0), rowspan=3)
	for ax in (axSeq, axSim):
		ax.autoscale(tight=True)
	axSeq.plot(seq)
	if answerIdxs is not None:
		for idx in answerIdxs:
			viz.plotVertLine(idx, ax=axSeq)
	Xpad = synth.appendZeros(X, length-1)
	axSim.imshow(Xpad, interpolation='nearest', aspect='auto')
	# im = axSim.imshow(Xpad, interpolation='nearest', aspect='auto')
	# plt.colorbar(im, cax=axSim)

	axSeq.set_title("Time Series")
	axSim.set_title("Similarities Matrix")

	# plt.figure()
	# plt.imshow(Xorig, interpolation='nearest', aspect='auto')
	# plt.colorbar()

	# plt.figure()
	# plt.imshow(X, interpolation='nearest', aspect='auto')
	# plt.colorbar()

	# # plt.figure()
	# # Xfilt = ff2.localMaxFilterSimMat(X, allowEq=True)
	# # plt.imshow(Xfilt, interpolation='nearest', aspect='auto')
	# # plt.colorbar()

	# # plt.figure()
	# # Xfilt = ff2.localMaxFilterSimMat(X, allowEq=False)
	# # plt.imshow(Xfilt, interpolation='nearest', aspect='auto')
	# # plt.colorbar()

	# plt.figure()
	# plt.imshow(Xblur, interpolation='nearest', aspect='auto')
	# plt.colorbar()

	# plt.show()
	# return

	# ================================ science

	# ------------------------ derived stats
	kMax = int(X.shape[1] / Lmin + .5)
	windowLen = Lmax - length + 1
	# windowShape = (X.shape[0], Lmax)
	# windowSize = np.prod(windowShape)
	nLocs = X.shape[1] - windowLen + 1

	p0 = np.mean(X) # fraction of entries that are 1 (roughly)

	# intersections = computeIntersections(X, windowLen)
	# windowSims = np.sum(intersections, axis=2)
	# colSims = np.dot(X.T, X)
	colSims = np.dot(X.T, Xblur)
	filt = np.zeros((windowLen, windowLen)) + np.diag(np.ones(windowLen)) # zeros except 1s on diag
	windowSims = sig.convolve2d(colSims, filt, mode='valid')

	windowVects = vectorizeWindowLocs(X, windowLen)
	windowVectsBlur = vectorizeWindowLocs(Xblur, windowLen)

	plt.figure()
	plt.imshow(windowSims, interpolation='nearest', aspect='auto')

	# plt.show()
	# return

	# ------------------------ find stuff

	# #
	# # Version where we we tell it k
	# #
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
	# 	if windowSims[i, i] * k0 <= bsfScore:
	# 		continue
	# 	idxs = sub.optimalAlignK(row, Lmin, k0)
	# 	intersection = windowVects[i]
	# 	sz = 0
	# 	for idx in idxs:
	# 		intersection = np.minimum(intersection, windowVectsBlur[idx])
	# 		sz = np.sum(intersection)
	# 		if sz * k0 <= bsfScore:
	# 			break
	# 	score = sz * k0
	# 	if score > bsfScore:
	# 		print("window {0}, k={1}, score={2} is the new best!".format(i, k0, score))
	# 		bsfScore = score
	# 		bsfLocs = idxs
	# 		bsfIntersection = np.copy(intersection)

	# #
	# # Version where we look for similarities to orig seq
	# #
	# bsfScore = 0
	# bsfLocs = None
	# bsfIntersection = None
	# for i, row in enumerate(windowSims):
	# 	if i % 20 == 0:
	# 		print("computing stuff for row {}".format(i))
	# 	# early abandon if this location has so little stuff that no
	# 	# intersection with it can possibly beat the best score
	# 	if windowSims[i,i] * kMax <= bsfScore: # highest score is kMax identical locs
	# 		# print("immediately abandoning window {}!".format(i))
	# 		continue
	# 	# print("not abandoning window {}!".format(i))
	# 	# best combination of idxs such that none are within Lmin of each other
	# 	idxs = sub.optimalAlignment(row, Lmin)
	# 	# print i, ": ", idxs
	# 	# order idxs by descending order of associated score
	# 	sizes = windowSims[i, idxs]
	# 	sortedSizesOrder = np.argsort(sizes)[::-1]
	# 	sortedIdxs = idxs[sortedSizesOrder]
	# 	# iteratively intersect with another near neighbor, compute the
	# 	# associated score, and check if it's better (or if we can early abandon)
	# 	intersection = windowVects[i]
	# 	numIdxs = len(sortedIdxs)
	# 	for j, idx in enumerate(sortedIdxs):
	# 		k = j + 1
	# 		intersection = np.minimum(intersection, windowVectsBlur[idx])
	# 		sz = np.sum(intersection) # use apodization window
	# 		# sz = np.count_nonzero(intersection) # just max-pool
	# 		score = sz * k
	# 		if k > 1 and score > bsfScore:
	# 			print("window {0}, k={1}, score={2} is the new best!".format(i, k, score))
	# 			print("sortedIdxs = {}".format(str(sortedIdxs)))
	# 			print("sortedIdxScores = {}".format(str(windowSims[i, sortedIdxs])))
	# 			print("------------------------")
	# 			bsfScore = score
	# 			bsfLocs = sortedIdxs[:k]
	# 			bsfIntersection = np.copy(intersection)
	# 		# early abandon if this can't possibly beat the best score, which
	# 		# is the case exactly when the intersection is so small that perfect
	# 		# matches at all future locations still wouldn't be good enough
	# 		elif sz * numIdxs <= bsfScore:
	# 			# print("early abandoning window {} at k={}".format(i, k))
	# 			break

	# #
	# # Version where we look for similarities to orig seq and use nearest
	# # enemy dist as M0
	# #
	# bsfScore = 0
	# bsfLocs = None
	# bsfIntersection = None
	# for i, row in enumerate(windowSims):
	# 	if i % 20 == 0:
	# 		print("computing stuff for row {}".format(i))
	# 	# early abandon if this location has so little stuff that no
	# 	# intersection with it can possibly beat the best score
	# 	if windowSims[i,i] * kMax <= bsfScore: # highest score is kMax identical locs
	# 		# print("immediately abandoning window {}!".format(i))
	# 		continue

	# 	# best combination of idxs such that none are within Lmin of each other
	# 	# validRow = row[:(-length + 1)] # can't go past end of ts
	# 	# idxs = sub.optimalAlignment(validRow, Lmin)
	# 	idxs = sub.optimalAlignment(row, Lmin) # goes past end of ts, but better

	# 	# order idxs by descending order of associated score
	# 	sizes = windowSims[i, idxs]
	# 	sortedSizesOrder = np.argsort(sizes)[::-1]
	# 	sortedIdxs = idxs[sortedSizesOrder]

	# 	# iteratively intersect with another near neighbor, compute the
	# 	# associated score, and check if it's better (or if we can early abandon)
	# 	intersection = windowVects[i]
	# 	numIdxs = len(sortedIdxs)
	# 	# allZeros = np.zeros(intersection.shape)
	# 	nextIntersection = np.minimum(intersection, windowVectsBlur[sortedIdxs[0]])
	# 	nextSz = np.sum(nextIntersection)
	# 	for j, idx in enumerate(sortedIdxs):
	# 		k = j + 1
	# 		intersection = np.copy(nextIntersection)
	# 		sz = nextSz
	# 		if k < numIdxs:
	# 			nextIdx = sortedIdxs[k] # since k = j+1
	# 			nextIntersection = np.minimum(intersection, windowVectsBlur[nextIdx])
	# 			nextSz = np.sum(nextIntersection) # sum -> use apodization window
	# 		else:
	# 			nextSz = sz * p0

	# 		score = (sz - nextSz) * k
	# 		if k > 1 and score > bsfScore:
	# 			print("window {0}, k={1}, score={2} is the new best!".format(i, k, score))
	# 			print("sortedIdxs = {}".format(str(sortedIdxs)))
	# 			print("sortedIdxScores = {}".format(str(windowSims[i, sortedIdxs])))
	# 			print("------------------------")
	# 			bsfScore = score
	# 			bsfLocs = sortedIdxs[:k]
	# 			bsfIntersection = np.copy(intersection)
	# 		# early abandon if this can't possibly beat the best score, which
	# 		# is the case exactly when the intersection is so small that perfect
	# 		# matches at all future locations still wouldn't be good enough
	# 		elif sz * numIdxs <= bsfScore:
	# 			# print("early abandoning window {} at k={}".format(i, k))
	# 			break

	#
	# Version where we look for similarities to orig seq and use nearest
	# enemy dist as M0, and use mean values instead of intersection
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
			continue

		# best combination of idxs such that none are within Lmin of each other
		# validRow = row[:(-length + 1)] # can't go past end of ts
		# idxs = sub.optimalAlignment(validRow, Lmin)
		idxs = sub.optimalAlignment(row, Lmin) # goes past end of ts, but better

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
				nextIntersection = np.minimum(filt, windowVectsBlur[nextIdx])
				nextFiltSum += nextIntersection
				nextFilt = nextFiltSum / (k+1)  # avg value of each feature in intersections
				# nextSz = np.sum(nextFilt) # big even if like no intersection...
				nextSz = np.sum(nextIntersection)
				bigEnoughIntersection = nextIntersection[nextIntersection > minSim]
				nextSz = np.sum(bigEnoughIntersection)
			else:
				nextSz = sz * p0

			score = (sz - nextSz) * k
			if k > 1 and score > bsfScore:
				print("window {0}, k={1}, score={2} is the new best!".format(i, k, score))
				print("sortedIdxs = {}".format(str(sortedIdxs)))
				print("sortedIdxScores = {}".format(str(windowSims[i, sortedIdxs])))
				print("------------------------")
				bsfScore = score
				bsfLocs = sortedIdxs[:k]
				bsfIntersection = np.copy(filt)
			# early abandon if this can't possibly beat the best score, which
			# is the case exactly when the intersection is so small that perfect
			# matches at all future locations still wouldn't be good enough
			elif sz * numIdxs <= bsfScore:
				# TODO can we actually early abandon here? next window loc
				# could increase filt, and thus score for a given loc isn't
				# necessarily non-increasing...
				# 	-can't abandon using this test, but pretty sure there's
				# 	a lower bound to be had here somewhere
				# print("early abandoning window {} at k={}".format(i, k))
				break

	#
	# Version where we look for similarities to orig seq and use nearest
	# enemy dist as M0, and use mean values instead of intersection,
	# and don't sort the indices, but instead care about overlap
	#
	# bsfScore = 0
	# bsfLocs = None
	# bsfIntersection = None
	# for i, row in enumerate(windowSims):
	# 	if i % 20 == 0:
	# 		print("computing stuff for row {}".format(i))
	# 	# early abandon if this location has so little stuff that no
	# 	# intersection with it can possibly beat the best score
	# 	if windowSims[i,i] * kMax <= bsfScore: # highest score is kMax identical locs
	# 		continue

	# 	# best combination of idxs such that none are within Lmin of each other
	# 	# validRow = row[:(-length + 1)] # can't go past end of ts
	# 	# idxs = sub.optimalAlignment(validRow, Lmin)
	# 	idxs = sub.optimalAlignment(row, Lmin) # goes past end of ts, but better

	# 	# order idxs by descending order of associated score
	# 	sizes = windowSims[i, idxs]
	# 	sortedSizesOrder = np.argsort(sizes)[::-1]
	# 	sortedIdxs = idxs[sortedSizesOrder]

	# 	# iteratively intersect with another near neighbor, compute the
	# 	# associated score, and check if it's better (or if we can early abandon)
	# 	intersection = windowVects[i]
	# 	numIdxs = len(sortedIdxs)
	# 	nextSz = np.sum(intersection)
	# 	nextFilt = np.array(intersection, dtype=np.float)
	# 	nextFiltSum = np.array(nextFilt, dtype=np.float)
	# 	for j, idx in enumerate(sortedIdxs):
	# 		k = j + 1
	# 		filt = np.copy(nextFilt)
	# 		sz = nextSz
	# 		if k < numIdxs:
	# 			nextIdx = sortedIdxs[k] # since k = j+1
	# 			nextIntersection = np.minimum(filt, windowVectsBlur[nextIdx])
	# 			nextFiltSum += nextIntersection
	# 			nextFilt = nextFiltSum / (k+1)  # avg value of each feature in intersections
	# 			# nextSz = np.sum(nextFilt) # big even if like no intersection...
	# 			nextSz = np.sum(nextIntersection)
	# 			bigEnoughIntersection = nextIntersection[nextIntersection > minSim]
	# 			nextSz = np.sum(bigEnoughIntersection)
	# 		else:
	# 			nextSz = sz * p0

	# 		score = (sz - nextSz) * k
	# 		if k > 1 and score > bsfScore:
	# 			print("window {0}, k={1}, score={2} is the new best!".format(i, k, score))
	# 			print("sortedIdxs = {}".format(str(sortedIdxs)))
	# 			print("sortedIdxScores = {}".format(str(windowSims[i, sortedIdxs])))
	# 			print("------------------------")
	# 			bsfScore = score
	# 			bsfLocs = sortedIdxs[:k]
	# 			bsfIntersection = np.copy(filt)
	# 		# early abandon if this can't possibly beat the best score, which
	# 		# is the case exactly when the intersection is so small that perfect
	# 		# matches at all future locations still wouldn't be good enough
	# 		elif sz * numIdxs <= bsfScore:
	# 			# TODO can we actually early abandon here? next window loc
	# 			# could increase filt, and thus score for a given loc isn't
	# 			# necessarily non-increasing...
	# 			# 	-can't abandon using this test, but pretty sure there's
	# 			# 	a lower bound to be had here somewhere
	# 			# print("early abandoning window {} at k={}".format(i, k))
	# 			break

	# ------------------------ recover original ts

	bsfIntersection *= bsfIntersection >= minSim
	bsfIntersectionWindow = bsfIntersection.reshape((-1, windowLen))
	sums = np.sum(bsfIntersectionWindow, axis=0)

	kBest = len(bsfLocs)
	p0 = np.power(p0, kBest)
	# expectedOnesPerCol = p0 * X.shape[1]
	expectedOnesPerCol = p0 * X.shape[1] * 2
	sums -= expectedOnesPerCol

	plt.plot(sums)

	start, end, _ = maxSubarray(sums)
	patStart, patEnd = start, end + 1 + length

	# ================================ show output

	print "bestScore = {}".format(bsfScore)
	print "bestLocations = {}".format(str(bsfLocs))

	for idx in bsfLocs:
		viz.plotRect(axSim, idx, idx+windowLen)

	# print bsfIntersectionWindow.shape
	# print sums.shape

	# plt.plot(sums)
	# viz.plotRect(plt.gca(), start, end + 1)
	for idx in bsfLocs:
		viz.plotRect(axSeq, idx + patStart, idx + patEnd)

	plt.figure()
	plt.imshow(bsfIntersectionWindow, interpolation='nearest', aspect='auto')

	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()

	# from doctest import testmod
	# testmod()

	# A = np.array([[1,2,3], [4,5,6]])
	# computeIntersections(A, 2)
