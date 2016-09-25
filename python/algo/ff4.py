#!/usr/env/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

from numba import jit

from joblib import Memory
memory = Memory('./output', verbose=0)

from ..datasets import synthetic as synth
from ..utils import arrays as ar
from ..utils.subseq import optimalAlignK
from ..viz.viz_utils import plotRect

import ff2
# from ff3 import maxSubarray
from ..utils import subseq as sub

# @memory.cache
@jit
def computeSimMat(seq, length=8):
	print("computing simMat on seq of shape {}".format(str(seq.shape)))

	nDims = 1
	nSubseqs = len(seq) - length + 1
	# ND seq; call for each dim separately, then combine resulting simMats
	if len(seq.shape) > 1 and seq.shape[1] > 1:
		# print "it's an nd seq!, nDims = ", seq.shape[1]
		nDims = seq.shape[1]
		mats = np.empty((nDims, nSubseqs, nSubseqs))
		for dim in range(nDims):
			# compute sims for each dimension
			mats[dim] = computeSimMat(seq[:,dim], length)

			# plt.figure()
			# mats[dim][mats[dim] < .75] = 0.
			# plt.imshow(mats[dim] > .75)
			# plt.imshow(mats[dim])

		# stack sim mats for each dim on top of one another (ie, append rows)
		return mats.reshape((-1, mats.shape[2]))

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
		sims[i,i] = np.max(sims[i]) # rest of row

	# assert(np.min(sims) >= -1.00001)
	# assert(np.max(sims) <= 1.00001)

	return sims


def main():
	# ================================ consts for everything
	# consts for generating data
	# n = 1000
	n = 500
	# n = 300
	# length = 8
	length = 16
	# length = 50
	nInstances = 3
	exampleLengths = [55, 60, 65]
	noiseStd = .5

	# consts for algorithm
	Lmin = max(20, length)	# only needed for optimalAlignK() spacing
	Lmax = 100				# loose upper bound on pattern length
	maxEpochs = 10 			# not actually needed
	covLambda = .99			# just needed to keep stuff non-singular
	# minSim = .9				# loose cutoff for what counts as similar
	minSim = 0.				# loose cutoff for what counts as similar

	# ================================ prep stuff

	# ------------------------ create data

	# seq = synth.randconst(n, std=noiseStd)
	seq1 = synth.randconst(n, std=noiseStd)
	seq2 = synth.randconst(n, std=noiseStd)
	seq3 = synth.randconst(n, std=noiseStd)
	# seq = synth.notSoRandomWalk(n, std=noiseStd, trendFilterLength=80, lpfLength=8)
	# seq = synth.notSoRandomWalk(n, std=noiseStd, trendFilterLength=80, lpfLength=16)
	# seq1 = synth.notSoRandomWalk(n, std=noiseStd, trendFilterLength=80, lpfLength=16)
	# seq2 = synth.notSoRandomWalk(n, std=noiseStd, trendFilterLength=80, lpfLength=16)
	# seq = np.vstack((seq1, seq2)).T
	seq = np.vstack((seq1, seq2, seq3)).T

	# print seq.shape
	# return

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
			# inst = synth.warpedSeq(inst) * 5
			synth.embedSubseq(seq, inst, startIdxs[i])
			# synth.embedSubseq(seq, inst * 5, startIdxs[i])

	# plt.figure()
	# plt.plot(seq)
	# plt.show()
	# return

	# plt.figure(figsize=(8, 2))
	# plt.plot(seq)
	# plt.title("Time Series")
	# plt.tight_layout()
	# plt.savefig("communicate/figs/sinesTs.pdf")
	# plt.show()
	# return

	from ..datasets import read_msrc as msrc
	idxs = [2]
	recordings = msrc.getRecordings(idxs=idxs)
	r = list(recordings)[0]
	seq = r.data[:, 20:23]
	print "orig seq shape", seq.shape
	seq = ar.downsampleMat(seq, rowsBy=10)
	print "downsampled seq shape", seq.shape
	length = 8
	Lmin = len(seq) / 20
	Lmax = len(seq) / 10
	# return

	# plt.plot(seq)
	# plt.show()
	# return

	# ------------------------ create simMat

	# dists, Xnorm = sub.pairwiseDists(seq, length)
	# dists = dists[0]
	# dists /= length
	# # # dists = np.sqrt(dists)
	# # sims0 = sub.zNormDistsToCosSims(dists, length)
	# # sims0 = np.maximum(0, sims0)
	# plt.figure()
	# plt.imshow(dists)
	# # plt.imshow(sims0)  # identical to sims below
	# plt.colorbar()

	# print "Xnorm stats:", np.mean(Xnorm, axis=1), np.std(Xnorm, axis=1), np.sum(Xnorm*Xnorm, axis=1)
	# print "Xnorm shape:", Xnorm.shape

	# sims = sub.similarityMat(seq, length)
	# plt.figure()
	# plt.imshow(sims)
	# plt.colorbar()

	# X = ff2.computeSimMat([seq], length, -1, removeSelfMatch=True)
	# X = computeSimMat(seq[:,0], length)

	# noise = synth.randconst(seq.shape) # add noise for debugging
	# seq = np.vstack((noise, seq, noise))

	X = computeSimMat(seq, length)
	# X = X*X * np.sign(X)

	# plt.show()
	# return

	# plt.figure()
	# plt.plot(seq)
	# plt.show()
	# # return

	# plt.figure()
	# plt.imshow(X)
	# plt.show()
	# return

	# X = ff2.localMaxFilterSimMat(X)
	# X = ff2.filterSimMat(X, length-1, 'hamming')

	# X[X < minSim] = 0.
	# X *= X # squared cos sims
	X = X > minSim
	# X[X < 0.] = 0.
	# X = ar.removeZeroRows(X)
	# X = filters.maximum_filter1d(X, length-1, axis=1)

	print "simMat dims:", X.shape
	print "simMat nonzeros, total, frac = ", np.count_nonzero(X), X.size, np.count_nonzero(X) / float(X.size)

	# plotting stuff
	plt.figure()
	axSeq = plt.subplot2grid((5,1), (0,0))
	axSim = plt.subplot2grid((5,1), (1,0), rowspan=3)
	axProb = plt.subplot2grid((5,1),(4,0))
	for ax in (axSeq, axSim, axProb):
		ax.autoscale(tight=True)
	axSeq.plot(seq)
	axSim.imshow(synth.appendZeros(X, length-1), interpolation='nearest', aspect='auto')

	axSeq.set_title("Time Series")
	axSim.set_title("Similarities Matrix")
	axProb.set_title("Probability of Pattern Instance Starting at Each Index")

	# plt.show()
	# return

	# ================================ actual algorithm (besides simMat creation)

	# ------------------------ compute basic stats and normalize stuff
	X = ar.meanNormalizeRows(X)
	windowShape = (X.shape[0], Lmax)
	gaussDims = np.prod(windowShape)
	divideDistsBy = gaussDims

	# nLocs = n - Lmax - length + 1
	nLocs = X.shape[1] - Lmax + 1
	print "nLocs", nLocs

	var0 = np.var(X) # global variance
	Cov0 = np.zeros(windowShape) + var0
	Cov = np.copy(Cov0)

	# initialize W to the "biggest" location
	# Mu = np.ones(windowShape)
	# Mu /= np.linalg.norm(Mu)
	bestSum = -np.inf
	bestIdx = -1
	for i in range(nLocs):
		Xi = X[:, i:i+Lmax]
		tot = np.sum(Xi*Xi)
		if tot > bestSum:
			bestSum = tot
			bestIdx = i
	print("initializing Mu based on idx {}".format(bestIdx))
	Mu = X[:, bestIdx:bestIdx+Lmax] - .01 # not an exact match

	# ------------------------ initialize arrays for main loop
	ys = np.empty(nLocs)
	windowUpdatesShape = (nLocs, windowShape[0], windowShape[1])
	dMus = np.empty(windowUpdatesShape)
	dCovs = np.empty(windowUpdatesShape)
	y0s = np.empty(nLocs)
	optimalLocs = None
	oldLocs = None

	# ------------------------ main loop
	for ep in range(maxEpochs):
		print "starting epoch {}".format(ep)
		# compute stats for each location
		for i in range(nLocs):
			Xi = X[:, i:i+Lmax]
			diff = Xi - Mu
			diff_sq = diff * diff
			ys[i] = np.sum(diff_sq / Cov)
			dMus[i] = Xi
			dCovs[i] = diff_sq
			y0s[i] = np.sum(Xi * Xi / Cov0)

		# compute prob of each location being a pattern instance
		probsPat = np.exp(-ys/divideDistsBy)
		probs0 = np.exp(-y0s/divideDistsBy)
		probs = probsPat / (probsPat + probs0)

		# find optimal set of k locations
		optimalLocs = optimalAlignK(probs, Lmin, nInstances)
		print "optimal locs are ", optimalLocs

		# compute MAP estimate of params for pattern gaussian
		# probs /= np.sum(probs) # set sum of update weights = 1
		# for i, p in enumerate(probs):
		# updateWeights = probs[optimalLocs] / np.sum(probs[optimalLocs])
		updateWeights = np.ones(nInstances) / nInstances # ignore probs
		Mu = 0
		Cov = (1-covLambda) * Cov0
		# lurn = 1. / (np.sqrt(ep+1))
		for i, loc in enumerate(optimalLocs):
			# Mu += lurn * dMus[loc] * updateWeights[i]
			Mu += dMus[loc] * updateWeights[i]
			Cov += covLambda * dCovs[loc] * updateWeights[i]
		# Mu += lurn * dMu
		# Cov = covLambda * dCov + (1 - covLambda) * Cov0

		# plt.figure()
		# plt.plot(np.r_[probs, np.zeros(n - len(probs))])
		# plt.imshow(Mu)

		# break if same locations as before
		if np.array_equal(optimalLocs, oldLocs):
			break
		oldLocs = optimalLocs

	# ------------------------ extract instances in time domain

	# remove negative stuff, cuz neg weights just make it care about the whole
	# row where features consistently appear
	# Mu = np.maximum(0, Mu)

	# compute squared dists from 0
	# Mu_sq = Mu*Mu
	# Mu_sq[Mu_sq < 1.*Cov0] = 0. # XXX hack to kill low weights
	# scoresMat = Mu_sq / Cov0
	scoresMat = Mu*Mu / Cov0

	# print np.min(scoresMat) # yep, nonnegative
	# return

	# shift dists down until just one range of scores above 0
	# scoresMatShifted = np.copy(scoresMat)
	# thresh = 1.
	#
	# SELF: adaptive L0/L1 reg isn't enough to play nicely with max subarray
	# cuz these will only get scores for a col to 0, and max subarray only
	# ignores them if negative
	#
	# while True: # repeatedly subtract .05 until just one section above 0
	# 	lamda = thresh / scoresMat.shape[1]
	# 	# scoresMatShifted[scoresMatShifted < lamda] = 0.
	# 	# scoresMatShifted = np.maximum(0, scoresMatShifted - lamda)
	# 	scoresMatShifted = scoresMat - lamda
	# 	Wscores = np.mean(scoresMatShifted, axis=0)
	# 	# Wscores = np.mean(scoresMatShifted, axis=0) - .001
	# 	# print Wscores
	# 	idxsAbove0 = np.where(Wscores > 0)[0]
	# 	if len(idxsAbove0) == len(Wscores): # can't include everything
	# 		continue
	# 	print idxsAbove0
	# 	changes = idxsAbove0[1:] - idxsAbove0[:-1]
	# 	if np.all(changes) <= 1 and np.min(Wscores) < 0:
	# 		break
	# 	thresh += .1
		# scoresMatShifted[scoresMatShifted < thresh]
		# Wscores -= .05 # Wow, this is terrible; really need a way to set this...
		# scoresMatShifted -= .05 / scoresMatShifted.shape[1]

		# plt.figure()
		# plt.imshow(scoresMatShifted)


	# TODO compute window locs by finding starts and ends in X, not
	# in filter; this way we can return stuff of variable lengths


	Wscores = np.mean(scoresMat, axis=0)
	# Wscores = filters.convolve1d(Wscores, np.hamming(4))
	Wscores = filters.convolve1d(Wscores, np.hamming(Lmin))
	derivs = Wscores[1:] - Wscores[:-1]
	start, end = np.argmax(derivs), np.argmin(derivs)
	# start, end, _ = maxSubarray(Wscores)

	patStart, patEnd = start, end + length

	for loc in optimalLocs:
		plotRect(axSeq, loc + patStart, loc + patEnd)

	axProb.plot(np.r_[probs, np.zeros(n - len(probs))])

	# plt.figure()
	# plt.imshow(Mu)

	plt.tight_layout(pad=.05)

	plt.figure()
	plt.plot(Wscores)

	plt.figure()
	plt.imshow(Mu)

	plt.show()

	# plt.figure(figsize=(8, 2))
	# axSeq2 = plt.gca()
	# axSeq2.plot(seq)
	# axSeq2.set_title('Discovered Locations')
	# for loc in optimalLocs:
	# 	plotRect(axSeq2, loc + patStart, loc + patEnd)

	# plt.tight_layout()
	# plt.savefig('communicate/figs/sinesLocs.pdf')
	# plt.show()

if __name__ == '__main__':
	main()

	# seq = np.random.randn(100)
	# X = computeSimMat(seq, 8)

	# plt.imshow(X)
	# plt.show()


