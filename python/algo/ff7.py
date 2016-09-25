#!/usr/env/python

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

from ..datasets import synthetic as synth
from ..utils import arrays as ar
from ..utils import subseq as sub
from ..viz import viz_utils as viz

import representation as rep

from ff2 import localMaxFilterSimMat, filterSimMat
from ff3 import maxSubarray
from ff5 import embedExamples
from ff6 import vectorizeWindowLocs


# ================================================================ Main

def main():
	# np.random.seed(123)

	# ================================ consts for everything
	# consts for generating data
	# n = 1000
	n = 500
	# n = 300
	# length = 8
	# length = 16
	# length = 32
	# length = 50
	# nInstances = 3
	exampleLengths = [55, 60, 65]
	# exampleLengths = [60, 60, 60]
	noiseStd = .5

	# consts for algorithm
	# Lmin = max(20, length)	# only needed for optimalAlignK() spacing
	Lmin = 20				# only needed for optimalAlignK() spacing
	Lmax = 100				# loose upper bound on pattern length
	# minSim = .5
	minSim = 0.
	length = Lmin // 2
	# length = Lmin // 4
	# length = 3

	answerIdxs = None

	USE_MSRC = True
	# USE_MSRC = False

	# ================================ data

	# ------------------------ synthetic data

	# seq = synth.randconst(n, std=noiseStd)
	seq = synth.randwalk(n, std=noiseStd)
	# seq = synth.notSoRandomWalk(n, std=noiseStd, trendFilterLength=80, lpfLength=4)
	# seq = synth.notSoRandomWalk(n, std=noiseStd, trendFilterLength=80, lpfLength=16)
	seq = embedExamples(seq, exampleLengths)
	# seq = synth.appendZeros(seq, Lmax)

	# ------------------------ msrc

	if USE_MSRC:
		from ..datasets import read_msrc as msrc
		# idxs = [0]
		# idxs = [1]
		# idxs = [2]
		# idxs = [7] # length 1500, but instances of length like 20
		# idxs = [8] # gets owned on this one cuz patterns of length like 100
		# idxs = [9] # missing an annotation, it appears
		idxs = [10] # something crazy about feature rep here # TODO fix
		# idxs = [11] # crap cuz bad, low-variance signals
		# idxs = [12] # has garbagey sections like [10]
		# idxs = [13] # empty feature mat # TODO
		# idxs = [14]
		downsampleBy = 2
		# downsampleBy = 1
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
		# length = max(8, Lmin / 2)
		Lmin = len(seq) // 20
		# Lmax = len(seq) // 8
		Lmax = len(seq) // 10
		length = Lmin // 2
		# Lmax = len(seq) / 20
		# k0 = 10
		# minSim = .5
		answerIdxs = r.gestureIdxs / downsampleBy
		# print "seq shape", seq.shape
		prePadLen = Lmax - length
		# postPadLen = length - 1
		postPadLen = Lmax - length
		first = np.tile(seq[0], (prePadLen, 1))
		last = np.tile(seq[-1], (postPadLen, 1))
		seq = np.vstack((first, seq, last)) # pad with fixed val to allow all window positions
		# ^ TODO pad simMat with zeros instead--this introduces fake subseqs
		answerIdxs += prePadLen
		# seq = np.vstack((seq, np.tile(flat, (length-1, 1)))) # lets it get the last rep
		# print "seq shape", seq.shape

	# ================================ feature construction

	logMaxLength = int(np.floor(np.log2(Lmax)))
	# logMaxLength = int(np.ceil(np.log2(Lmax)))
	# logMinLength = 3 # -> length 8
	# logMinLength = 4 # -> length 16
	logMinLength = int(np.floor(np.log2(Lmin)))
	lengths = np.arange(logMinLength, logMaxLength + 1)
	lengths = 2 ** lengths
	# lengths = [16]

	cardinality = 8
	breakpoints = rep.saxBreakpoints(cardinality)

	X = rep.multiNormalizeAndSparseQuantize(seq, lengths, breakpoints)
	# X = rep.multiSparseLineProject(seq, lengths, breakpoints, removeZeroRows=False)

	# lengths2 = np.arange(3, logMaxLength + 1)
	# lengths2 = 2 ** lengths2
	lengths2 = lengths # TODO uncomment after debug
	# lengths2 = [8, 32]

	# breakpoints2 = rep.defaultSparseLineBreakpoints(seq, scaleHowMany=2)
	breakpoints2 = rep.defaultSparseLineBreakpoints(seq)
	X2 = rep.multiSparseLineProject(seq, lengths2, breakpoints2)
	# X2 = X2 > minSim
	X2 = X2 > 0. # ignore correlations

	# print "shapes:"
	# print X.shape
	# print X2.shape

	X = np.vstack((X, X2))

	# plt.figure()
	# # viz.imshowBetter(X)
	# viz.imshowBetter(X2)
	# plt.figure()
	# viz.imshowBetter(X2 > 0.)
	# plt.show()

	# print seq.shape
	# plt.figure()

	# plt.plot(seq[:,0]) # bit of pattern, but only varies between -.4 and .2

	# okay, so 1st dim is all zeros
	# variances = rep.slidingVariance(seq, 8)
	# for dim in range(len(variances)):
	# 	plt.figure()
	# 	plt.plot(variances[dim].flatten())

	# print variances.shape
	# variances = rep.vstack3Tensor(variances.T)
	# print variances.shape
	# plt.plot(variances)

	# plt.show()
	# return

	X = localMaxFilterSimMat(X)
	# Xbool = np.copy(X)
	featureMeans = np.mean(X, axis=1).reshape((-1, 1))
	# print featureMeans
	X *= -np.log2(featureMeans) # variable encoding costs for rows
	# X /= -np.log(featureMeans)
	# Xblur = localMaxFilterSimMat(X) # try only maxFiltering Xblur
	Xblur = filterSimMat(X, length-1, 'hamming', scaleFilterMethod='max1')

	# plt.figure()
	# viz.imshowBetter(X)
	# plt.figure()
	# viz.imshowBetter(Xblur)

	print "featureMat dims:", X.shape
	Xnonzeros = np.count_nonzero(X)
	print "featureMat nonzeros, total, frac = ", Xnonzeros, X.size, Xnonzeros / float(X.size)

	# plt.show()
	# return

	# ================================ plotting crap

	plt.figure()
	axSeq = plt.subplot2grid((4,1), (0,0))
	axSim = plt.subplot2grid((4,1), (1,0), rowspan=3)
	for ax in (axSeq, axSim):
		ax.autoscale(tight=True)
	axSeq.plot(seq)
	# if answerIdxs is not None:
	# 	for idx in answerIdxs:
	# 		viz.plotVertLine(idx, ax=axSeq)
	padLen = len(seq) - X.shape[1]
	Xpad = synth.appendZeros(X, padLen)
	axSim.imshow(Xpad, interpolation='nearest', aspect='auto')
	# im = axSim.imshow(Xpad, interpolation='nearest', aspect='auto')
	# plt.colorbar(im, cax=axSim)

	axSeq.set_title("Time Series")
	axSim.set_title("Feature Matrix")

	# plt.show()
	# return

	# ================================ science

	# ------------------------ derived stats
	kMax = int(X.shape[1] / Lmin + .5)
	windowLen = Lmax - length + 1

	p0 = np.mean(X) # fraction of entries that are 1 (roughly)
	# p0 = 2 * np.mean(X) # lambda for l0 reg based on features being bernoulli at 2 locs
	minSim = p0
	# p0 = -np.log(np.mean(Xbool)) # fraction of entries that are 1 (roughly)
	# noiseSz = p0 * X.shape[0] * windowLen # way too hard to beat
	expectedOnesPerWindow = p0 * X.shape[0] * windowLen
	noiseSz = p0 * expectedOnesPerWindow # num ones to begin with

	# intersections = computeIntersections(X, windowLen)
	# windowSims = np.sum(intersections, axis=2)
	# colSims = np.dot(X.T, X)
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
				# nextSz = -1
			enemySz = max(nextSz, noiseSz)

			score = (sz - enemySz) * k
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
			elif noiseSz > nextSz:
				break

	# #
	# # Version where we look for similarities to orig seq and use nearest
	# # enemy dist as M0, and use mean values instead of intersection,
	# # and don't sort the indices, but instead care about overlap
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
	# expectedOnesPerCol = p0 * X.shape[1] * 2
	# expectedOnesPerCol = p0 * X.shape[1]
	expectedOnesPerCol = p0 * X.shape[0]
	sums -= expectedOnesPerCol

	# plt.figure()
	# plt.plot(sums)

	start, end, _ = maxSubarray(sums)
	# patStart, patEnd = start, end + 1 + length
	patStart, patEnd = start, end + 1
	# patStart, patEnd = start + length // 2, end + 1 + length

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

	if answerIdxs is not None:
		for idx in answerIdxs:
			viz.plotVertLine(idx, ax=axSeq)

	plt.figure()
	plt.imshow(bsfIntersectionWindow, interpolation='nearest', aspect='auto')

	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()
