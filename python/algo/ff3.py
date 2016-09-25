#!/usr/env/python

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import beta
from scipy.optimize import curve_fit

from ..datasets import synthetic as synth
from ..utils import arrays as ar
from ..utils import subseq as sub

from ..viz import viz_utils as viz

import ff2

# def betaBinom(a, b, n, k):
# 	nChooseK = np.misc.comb(n, k)

# def binomCdf(n, k, p):
# 	return binom.cdf(k, n, p)

def sigmoid(x, x0, w):
	return 1. / (1 + np.exp(-w*(x-x0)))


def fitLogistic(y, x=None, guess=None):
	# assumes y is sorted in descending order
	if x is None:
		x = np.arange(len(y))
	if guess is None:
		guess = (len(y)/2, -1.)  # middle of data, moderate w
	# map y to [0, 1]
	y = y - np.min(y)
	y /= np.max(y)
	# print "fitLogistic(); len(y) = {}".format(len(y))
	# print "fitLogistic(); y = ", y
	popt, _ = curve_fit(sigmoid, x, y, p0=guess)
	return sigmoid(x, *popt)


def tryFindRelevantSubseq():
	# synth.seedRng(123)
	NLL = np.inf

	nExamples = 3
	nNoise = 10
	downsampleBy = 5
	# seqs = synth.makeSinesDataset(numSines=nExamples, numNoise=nNoise)
	seqs = synth.makeSinesDataset(numSines=nExamples, numNoise=nNoise, warped=True)
	# seqs = synth.makeSinesDataset(numSines=0, numNoise=nNoise+nExamples) # all noise
	# seqs = seqs[:nExamples] # only positive examples, no noise for now
	# seqs = map(lambda s: s + synth.randconst(s.shape, std=.25), seqs)
	seqs = map(lambda s: ar.downsampleMat(s, rowsBy=downsampleBy), seqs)

	# length = 40 / downsampleBy
	# pruneCorr = -1
	# mat = sub.similarityMat(seqs, length, pruneCorrAbove=pruneCorr)
	# plt.figure()
	# plt.imshow(mat)
	# plt.show()
	# return

	# for s in seqs:
	# 	plt.figure()
	# 	plt.plot(s)
	# plt.show()
	# return

	# plt.plot(np.vstack(seqs).T) # yep, looks right
	# plt.show()

	# simMat = ff2.computeSimMat(seqs, 8, .2, k=10, matForEachSeq=False)
	# plt.imshow(simMat)
	# plt.show()

	# length = 40 / downsampleBy
	# length = 60 / downsampleBy
	length = 8
	# dMax = length * (1 - minSim) # NOT actually enforcing same threshold
	dMax = length * .1
	# minSim = .5 # not equivalent to above dMax
	minSim = 0. # not equivalent to above dMax
	# dMax = length * 0
	# Xs = ff2.computeSimMat(seqs, length, dMax, k=(2*length), matForEachSeq=True)
	Xs = ff2.computeSimMat(seqs, length, dMax, matForEachSeq=True, removeSelfMatch=True)
		# normFeatures='mean') # works better than znorming
		# normFeatures='z')
	# Xnorms = Xs

	# plt.figure()
	# plt.imshow(Xs)
	# plt.show()
	# return

	# ------------------------ kill small values
	Xs = map(lambda X: X * (X > minSim), Xs)

	# ------------------------ extract relative maxima
	# Xs = map(lambda X: ff2.localMaxFilterSimMat(X), Xs)

	# ------------------------ temporal pooling
	# Xs = map(lambda X: ff2.filterSimMat(X, length-1, 'hamming'), Xs)
	# Xs = map(lambda X: ff2.filterSimMat(X, length-1, 'flat'), Xs)

	# ------------------------ normalize mean
	# use mean for each feature (row)
	X_combined = np.hstack(Xs)
	featureMeans = np.mean(X_combined, axis=1).reshape((-1, 1))
	Xnorms = map(lambda X: X - featureMeans, Xs)

	# use grand mean
	X_combined = np.hstack(Xs)
	# grandMean = np.mean(X_combined)
	# Xnorms = map(lambda X: X - grandMean, Xs)

	# no mean subtraction!
	# Xnorms = map(lambda X: X, Xs)

	# use mean for each element (row, col position)
	# X_mean = np.copy(Xs[0])
	# for X in Xs[1:]:
	# 	X_mean += X
	# X_mean /= nExamples
	# Xnorms = map(lambda X: X - X_mean, Xs)

	# for X in Xnorms:
	# 	print "min", np.min(X)
	# 	print "max", np.max(X)
	# return

	# ------------------------ normalize variance
	# Xnorms = map(lambda x: ar.stdNormalizeRows(x), Xnorms)

	# print X_mean

	# print np.where(np.isnan(X_mean))[0]
	# return

	# for s, m in zip(seqs, Xs):
	# # for s, m in zip(seqs, Xnorms):
	# 	print m.shape
	# 	plt.figure()
	# 	ax1 = plt.subplot2grid((2,1), (0,0))
	# 	ax2 = plt.subplot2grid((2,1), (1,0))
	# 	ax2.autoscale(tight=True)
	# 	ax1.plot(s)
	# 	m2 = np.hstack((m, np.zeros((m.shape[0], length-1))))
	# 	ax2.imshow(m2, interpolation='nearest', aspect='auto')
	# plt.show()

	# return

	# ax3.autoscale(tight=True)

	# ax2.imshow(synth.appendZeros(Xs[0], length-1), interpolation='nearest', aspect='auto')
	# ax1.set_title("Sequence containing sine wave")
	# ax2.set_title('Feature Representation of Sequence')
	# ax3.set_title('Learned Weights')

	# ax1 = plt.subplot2grid((1,3), (0,0))
	# ax2 = plt.subplot2grid((1,3), (0,1))
	# ax3 = plt.subplot2grid((1,3), (0,2))
	# ax2.autoscale(tight=True)
	# ax3.autoscale(tight=True)
	# ax1.plot(seqs[0])
	# ax2.imshow(synth.appendZeros(Xs[0], length-1), interpolation='nearest', aspect='auto')
	# ax1.set_title("Sequence containing sine wave")
	# ax2.set_title('Feature Representation of Sequence')
	# ax3.set_title('Learned Weights')

	# Y = np.empty(len(seqs))
	W = np.ones(Xs[0].shape, dtype=np.float64)
	W /= np.linalg.norm(W)
	Cov0 = np.zeros(W.shape) + np.mean(np.var(X_combined, axis=1))  # variance of everything ever
	Cov = np.copy(Cov0)
	# W += (Xs[0] + Xs[1]) / 2.

	# lamda = 20.
	# lamda_scaleBy = 1.
	lamda_scaleBy = 0.
	penalty = np.copy(W) * lamda_scaleBy
	lamda = penalty[0][0]
	# penalty = np.zeros(Xs[0].shape) + 1

	# plt.figure()
	# plt.imshow(W)

	nSeqs = len(seqs)
	ys = np.empty(nSeqs)
	dWs = np.empty((nSeqs, W.shape[0], W.shape[1]))
	dCovs = np.empty((nSeqs, W.shape[0], W.shape[1]))
	y0s = np.empty(nSeqs)

	# plt.figure()
	for ep in range(10):
		# print "w nan at", np.where(np.isnan(W))[0]

		for i in range(nSeqs):
			# X = Xs[i]
			Xnorm = Xnorms[i] # X - E[X]
			# ys[i] = np.sum(W * X)
			# print "y", y
			# dWs[i] = (Xnorm - W) * ys[i]
			# dW = (Xnorm - W) * np.exp(y)
			# print "max(X)", np.max(X)
			# print "max(X-E[X])", np.max(Xnorm)
			# print "max(dW)", np.max(dW)

			# ys[i] = np.sum(W * X)
			# dWs[i] = (X - W) # just vanilla avg
			# ys[i] = np.sum(W * Xnorm)
			# dWs[i] = (Xnorm - W) # just vanilla avg

			diff = Xnorm - W
			diff_sq = diff * diff
			ys[i] = np.sum(diff_sq / Cov)
			dWs[i] = diff
			dCovs[i] = diff_sq
			y0s[i] = np.sum(Xnorm * Xnorm / Cov0)

		# alpha = 1.
		# probs = np.exp(ys * alpha)
		# probs /= np.sum(probs)
		# probs = np.arange(nSeqs) < nExamples
		sortIdxs = np.argsort(ys)[::-1] # descending order

		p = float(nExamples) / nSeqs
		scaleBy = 10
		positions = np.linspace(0., 1., nSeqs)
		betaProbs = beta.sf(positions, p*scaleBy, (1-p)*scaleBy)

		ySort = ys[sortIdxs]
		sigmoid = fitLogistic(ySort)

		# plt.plot(ySort / ySort[0], 'o')

		# probs = betaProbs
		# probs = sigmoid
		# print ys, y0s
		gaussDims = np.prod(Xnorms[0].shape)
		# divideDistsBy = np.sqrt(gaussDims)
		divideDistsBy = gaussDims
		probsPat = np.exp(-ys/divideDistsBy)
		probs0 = np.exp(-y0s/divideDistsBy)
		probs = probsPat / (probsPat + probs0)

		probs /= np.sum(probs) # set sum of update weights = 1
		for i, p in enumerate(probs):
			idx = sortIdxs[i]
			dWs[i] *= probs[idx]
			dCovs[i] *= probs[idx]

		dW = np.sum(dWs, axis=0)
		dCov = np.sum(dCovs, axis=0)
		# print dW.shape

		lurn = 1. / (np.sqrt(ep+1))
		# print "lurn", lurn
		W += lurn * dW
		covLambda = .95
		Cov = covLambda * dCov + (1 - covLambda) * Cov0
		# Cov += lurn * dCov

		# W /= np.linalg.norm(W) # make it zero not quite as much stuff
		# W /= np.size(W)
		# W = ff2.l1Project(W)	# zeros almost everything ever
		# print np.sum(np.abs(W))

		# W[W < .001 / W.size] = 0
		# W -= np.maximum(np.abs(W), penalty) * np.sign(W)
		# W -= penalty * np.sign(W)
		# W[np.abs(W) < lamda] = 0.

		# W = np.maximum(0., W)

		# TODO proper projection onto L1 ball
		# W /= np.linalg.norm(W) 	# L2 constraint
		# W /= np.sum(W) 				# L1 constraint

		print ys
		# print probs
		print np.sum(ys)
		print np.dot(ys[sortIdxs], probs)
		# oldNLL = NLL
		NLL = -np.sum(np.log(probs))
		# if oldNLL < NLL: # this is far from nondecreasing ATM
		# 	print "================================"
		# 	print "oldNLL %g < new NLL %g" % (oldNLL, NLL)
		# 	print "================================"
		print "NLL: ", NLL
		print "------------------------ /iter%d" % (ep + 1)

		# # logistic function seems to nail the split even better, although
		# # hard to know what would happen if data weren't so contrived
		# plt.figure()
		# # # ySort = ys[sortIdxs] / np.max(ys)
		# # ySort = ys[sortIdxs]
		# plt.plot(ySort / ySort[0], 'o')
		# # sigmoid = fitLogistic(ySort)
		# plt.plot(sigmoid, label='sigmoid')
		# plt.plot(betaProbs / np.max(betaProbs), label='beta')
		# prod = sigmoid * betaProbs
		# plt.plot(prod / np.max(prod), label='product')
		# plt.legend(loc='best')

		# plt.figure()
		# plt.imshow(W)

	# ------------------------ reconstruct stuff from time domain

	# Wscores = W*W / Cov
	# patScores = np.exp(-Cov)
	Wsq = W*W
	# print "Cov0 = ", Cov0[0,0]
	Wsq[Wsq < Cov0] = 0. # XXX remove hack to kill low weights
	# Wsq -= Cov0
	zeroScores = Wsq / Cov0
	# print np.mean(patScores)
	# print np.mean(zeroScores)
	# zeroScores = np.exp(-zeroScores)
	# scoresMat = 1. - patScores / (patScores + zeroScores)
	scoresMat = zeroScores

	# these are like identical, suggesting cov is basically proportional
	# to mean in most cases; apparently just picking big means is probably
	# better than picking big means with small covs
	# plt.figure()
	# plt.imshow(patScores)
	# plt.figure()
	# plt.imshow(zeroScores)
	# plt.colorbar()

	# print np.min(scoresMat, axis=0)
	# print np.max(scoresMat, axis=0)

	# scoresMat[scoresMat < np.max(scoresMat)/2] = 0.
	# Wscores = np.mean(scoresMat, axis=0)
	Wscores = np.mean(scoresMat, axis=0)
	while True: # repeatedly subtract .05 until just one section above 0
		idxsAbove0 = np.where(Wscores > 0)[0]
		changes = idxsAbove0[1:] - idxsAbove0[:-1]
		if np.all(changes) <= 1 and np.min(Wscores) < 0:
			break
		Wscores -= .02 # Wow, this is terrible; really need a way to set this...

	# ^ perhaps figure out value we'd need to get just one contiguous positive
	# section somewhere

	start, end, _ = maxSubarray(Wscores)
	# patStart, patEnd = start - length/2, end + length/2
	# patStart, patEnd = start + length/2, end + length/2
	patStart, patEnd = start, end + length

	# ------------------------ show distro of W
	# plt.figure()
	# plt.plot(np.sort(W[W > 0.].flatten()), 'x')

	# ------------------------ viz learned weights and target seqs
	mainPlot = 1
	if mainPlot:
		plt.figure(figsize=(10,7))

		# plot sequences (and sum of weights at the top)
		axSeq1 = plt.subplot2grid((4,5), (0,0))
		axSeq2 = plt.subplot2grid((4,5), (0,1))
		axSeq3 = plt.subplot2grid((4,5), (0,2))
		axWeightSums = plt.subplot2grid((4,5), (0,3))

		for ax in (axSeq1, axSeq2, axSeq3, axWeightSums):
			ax.autoscale(tight=True)

		axSeq1.set_title("Instance #1")
		axSeq2.set_title("Instance #2")
		axSeq3.set_title("Instance #3")
		axWeightSums.set_title("Sum of Weights")

		axSeq1.plot(seqs[0])
		axSeq2.plot(seqs[1])
		axSeq3.plot(seqs[2])

		# W = ff2.localMaxFilterSimMat(W)
		# W[W < .01] = 0.
		W[W < .05] = 0.
		Wpad = synth.appendZeros(W, length-1, axis=1)
		Wsums = np.sum(Wpad, axis=0)
		axWeightSums.plot(Wsums / np.max(Wsums))
		# numNonzerosInCols = np.sum(Wpad > 0., axis=0) + 1.
		# print numNonzerosInCols
		# Wmeans = Wsums / numNonzerosInCols
		# axWeightSums.plot(Wmeans / np.max(Wmeans))
		viz.plotRect(axWeightSums, 60 / downsampleBy, 140 / downsampleBy)

		# plot simMats for sequences
		axMat1 = plt.subplot2grid((4,5), (1,0), rowspan=3)
		axMat2 = plt.subplot2grid((4,5), (1,1), rowspan=3)
		axMat3 = plt.subplot2grid((4,5), (1,2), rowspan=3)
		axMat4 = plt.subplot2grid((4,5), (1,3), rowspan=3)

		for ax in (axMat1, axMat2, axMat3, axMat4):
			ax.autoscale(tight=True)

		for i, ax in enumerate((axMat1, axMat2, axMat3)):
			ax.set_title("Features {}".format(i))
			# ax.plot(seqs[i])
			# ax.imshow(W)
			ax.imshow(synth.appendZeros(Xs[i], length-1),
				interpolation='nearest', aspect='auto')
		axMat4.set_title("Means")
		axMat4.imshow(synth.appendZeros(W, length-1),
				interpolation='nearest', aspect='auto')
		viz.plotRect(axMat4, 60 / downsampleBy, 140 / downsampleBy)

		# plot weights of stuff for extraction
		axScores = plt.subplot2grid((4,5), (1,4), rowspan=3)
		axScores.autoscale(tight=True)
		axScores.set_title("Scores")
		axScores.imshow(synth.appendZeros(scoresMat, length-1),
			interpolation='nearest', aspect='auto')

		# plot extracted ts
		axExtract = plt.subplot2grid((4,5), (0,4))
		axExtract.autoscale(tight=True)
		axExtract.set_title("Extracted Subsequences")
		for s in seqs[:nExamples]:
			axExtract.plot(s)
		viz.plotRect(axExtract, patStart, patEnd-1, color='g')

		plt.tight_layout(pad=.01)

	# Wmeans = np.mean(np.abs(W), axis=0)
	# Wmeans = np.mean(W*W, axis=0)
	# means = map(lambda X: np.mean(X*X), Xnorm)
	# mean = reduce(lambda x1, x2: (x1 + x2), means)
	# mean /= len(Xnorm)
	# # penalty = np.zeros(len(Wmeans)) + mean
	# # cumPenalty = np.cumsum(penalty)

	# Wscores = Wmeans - mean
	# Wscores -= np.log(.7) - np.log(.3) # difference in log probs of mean vs pattern gauss
	# Wscores = np.maximum(0, Wscores)


	# print np.min(Wscores)
	# print np.max(Wscores)

	# Wscores[Wscores < np.max(Wscores)/10] = 0.

	# plt.figure()
	# plt.imshow(scoresMat)

	# plt.figure()
	# # plt.plot(Wmeans)
	# # plt.gca().ticklabel_format(axis='y', style='plain') # stop being sci notation!
	# plt.plot(Wscores)
	# # plt.ylim((np.min(Wscores), 1.))
	# start, end, _ = maxSubarray(Wscores)
	# # end -= 1 # returned end idx isn't inclusive
	# print "start, end", start, end
	# viz.plotRect(plt.gca(), start, end-1)
	# # patStart, patEnd = start - length/2, end + length/2
	# # patStart, patEnd = start + length/2, end + length/2
	# patStart, patEnd = start, end + length
	# viz.plotRect(plt.gca(), patStart, patEnd-1, color='g')
	# # plt.plot(np.cumsum(Wmeans) - cumPenalty)
	# # plt.plot(np.cumsum(Wmeans[::-1])[::-1] - cumPenalty[::-1])

	# plt.figure()
	# for s in seqs[:nExamples]:
	# 	plt.plot(s)
	# viz.plotRect(plt.gca(), patStart, patEnd-1, color='g')
	# 	# plt.plot(ar.meanNormalizeCols(s[patStart:patEnd]))

	plt.show()

# adapted from http://stackoverflow.com/a/15063394
def maxSubarray(l):
	"""returns (a, b, c) such that sum(l[a:b]) = c and c is maximized"""
	best = 0
	cur = 0
	curi = starti = besti = 0
	for ind, i in enumerate(l):
		if cur+i > 0:
			cur += i
		else: # reset start position
			cur, curi = 0, ind+1

		if cur > best:
			starti, besti, best = curi, ind+1, cur
	return starti, besti, best

if __name__ == '__main__':
	tryFindRelevantSubseq()
	# seqs = synth.makeSinesDataset()
	# seqs = map(lambda s: ar.downsampleMat(s, colsBy=4), seqs)
	# print len(seqs)
	# print [s.shape for s in seqs]
	# print seqs
	# stacked = np.c_[seqs]
	# stacked = ar.zNormalizeRows(stacked)
	# stacked = ar.downsampleMat(stacked, colsBy=4)

	# # get unique subseqs as startIndices
	# occurIdxsForDims, XnormForDims = sub.uniqueSubseqs(seqs, 8, .2)
	# # for each dim, startIdx -> (featureIdx, dist)
	# neighborsForDims = ff2.neighborsFromOccurrencesAndSubseqs(occurIdxsForDims, XnormForDims, k=10)

	# # occurIdxs = occurIdxsForDims[0]
	# # Xnorm = XnormForDims[0]

	# # mat = computeSimMat(seqs, 12, .2, k=10)

	# plt.plot(stacked.T)
	# plt.show()

	# seqs = synth.makeSinesDataset()
	# # print [s.shape for s in seqs]
	# mats = ff2.computeSimMat(seqs, length=12, maxDist=.2, k=10, matForEachSeq=True)

	# for i, m in enumerate(mats):
	# 	plt.figure()
	# 	plt.imshow(m)
	# 	if i > 0:
	# 		break

	# plt.show()

	# ------------------------ try projecting onto random basis seqs
	# this works terribly, even by inspection

	# seqs = synth.makeSinesDataset()
	# seqs = map(lambda s: ar.downsampleMat(s, colsBy=4), seqs)

	# nProjections = 60
	# subseqLen = 40
	# # scaleDistBy = 1. / np.sqrt(subseqLen)
	# scaleDistBy = 1. / subseqLen
	# projMat = synth.randwalk((nProjections, subseqLen)).T # each col is a proj vect
	# projMat = ar.zNormalizeCols(projMat)

	# # plt.figure()
	# # plt.plot(projMat)

	# # projMat /= np.sqrt(subseqLen)
	# from ..utils import sliding_window as window
	# for s in seqs[:3]:
	# 	Xnorm, _, _ = window.flattened_subseqs_of_length([s], subseqLen, norm='each')
	# 	# Xnorm /= np.sqrt(subseqLen)
	# 	print Xnorm.shape
	# 	# print np.std(Xnorm, axis=1)
	# 	# print np.std(projMat, axis=0)
	# 	prods = np.dot(Xnorm, projMat) * scaleDistBy
	# 	print prods.shape
	# 	prods = np.exp(5 * prods) # force some semblance of sparsity
	# 	prods = ar.normalizeRows(prods)
	# 	# prods = np.dot(Xnorm, projMat)
	# 	# print prods
	# 	assert(np.max(prods) <= 1.)
	# 	assert(np.min(prods) >= -1.)

	# 	plt.figure()
	# 	ax1 = plt.subplot2grid((2,1), (0,0))
	# 	ax2 = plt.subplot2grid((2,1), (1,0))
	# 	ax1.plot(s)
	# 	ax2.imshow(prods.T)

	# plt.show()

