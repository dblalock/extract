#!/usr/env/python

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.ndimage import filters

from ..utils import arrays as ar
from ..utils import subseq as sub
from ..viz import viz_utils as viz

from motif import nonOverlappingMaxima, findMotifOfLengthFast
from misc import windowScoresRandWalk

from ff3 import maxSubarray
from ff8 import buildFeatureMat, extendSeq, preprocessFeatureMat
# from ff5 import embedExamples
# from ff6 import vectorizeWindowLocs

# DEFAULT_SEEDS_ALGO = 'all'
# DEFAULT_SEEDS_ALGO = 'pair'
DEFAULT_SEEDS_ALGO = 'walk'
DEFAULT_GENERALIZE_ALGO = 'map'
# DEFAULT_GENERALIZE_ALGO = 'avg'
# DEFAULT_GENERALIZE_ALGO = 'submodular'
DEFAULT_EXTRACT_LOCS_ALGO = 'map'


def dotProdsWithAllWindows(x, X):
	"""Slide x along the columns of X and compute the dot product

	>>> x = np.array([[1, 1], [2, 2]])
	>>> X = np.arange(12).reshape((2, -1))
	>>> dotProdsWithAllWindows(x, X) # doctest: +NORMALIZE_WHITESPACE
	array([27, 33, 39, 45, 51])
	"""
	return sig.correlate2d(X, x, mode='valid').flatten()


def localMaxFilterRows(X, allowEq=True):
	idxs = ar.idxsOfRelativeExtrema(X, maxima=True, allowEq=allowEq, axis=1)
	maximaOnly = np.zeros(X.shape)
	maximaOnly[idxs] = X[idxs]
	return maximaOnly


def filterRows(X, filtLength, filtType='hamming', scaleFilterMethod='max1'):
	if filtType == 'hamming':
		filt = np.hamming(filtLength)
	elif filtType == 'flat':
		filt = np.ones(filtLength)
	else:
		raise RuntimeError("Unknown/unsupported filter type {}".format(filtType))
	if scaleFilterMethod == 'max1':
		filt /= np.max(filt)
	elif scaleFilterMethod == 'sum1':
		filt /= np.sum(filt)

	return filters.convolve1d(X, weights=filt, axis=1, mode='constant')


def findAllInstancesSubmodular(X, Xblur, seedStartIdx, seedEndIdx, Lmin, Lmax,
	Lfilt, p0=-1, dotProds=None, bsfScore=0, **sink):

	if seedStartIdx < 0:
		print("WARNING: findAllInstancesAvgBlur(); received invalid"
			"start idx {}".format(seedStartIdx))
	if seedEndIdx > Xblur.shape[1]:
		print("WARNING: findAllInstancesAvgBlur(); received invalid"
			"end idx {}".format(seedEndIdx))

	# Version where we look for similarities to orig seq and use nearest
	# enemy dist as M0, and use mean values instead of intersection

	# ------------------------ derived stats
	kMax = int(X.shape[1] / Lmin + .5)
	windowLen = seedEndIdx - seedStartIdx # assume end idx not inclusive

	if p0 < 0.:
		p0 = np.mean(X) # fraction of entries that are 1 (roughly)
		# p0 = np.mean(X > 0.) # fraction of entries that are 1 # TODO try this
		# p0 = 2 * np.mean(X > 0.) # lambda for l0 reg based on features being bernoulli at 2 locs
	minSim = p0
	expectedOnesPerWindow = p0 * X.shape[0] * windowLen
	noiseSz = p0 * expectedOnesPerWindow # num ones to begin with

	# ------------------------ candidate location generation

	# print "X shape, Xblur shape, windowLen", X.shape, Xblur.shape, windowLen

	x0 = X[:, seedStartIdx:seedEndIdx]
	if (dotProds is None) or (not len(dotProds)):
		dotProds = dotProdsWithAllWindows(x0, Xblur)
	# else:
		# print "received dot prods with shape", dotProds.shape
		# trueDotProds = dotProdsWithAllWindows(x0, Xblur)
		# print "true dot prods shape", trueDotProds.shape

	# dotProds[dotProds < noiseSz] = 0. # don't even consider places worse than noise
	# dotProds -= noiseSz

	# compute best locations to try and then sort them in decreasing order
	# bestIdxs = nonOverlappingMaxima(dotProds, Lmin)
	bestIdxs = sub.optimalAlignment(dotProds, Lmin) # wow, this makes a big difference
	bestProds = dotProds[bestIdxs]
	# keepIdxs = np.where(bestProds > noiseSz)[0]
	# bestIdxs = bestIdxs[keepIdxs]
	# bestProds = bestProds[keepIdxs]
	sortIdxs = np.argsort(bestProds)[::-1]
	sortedIdxs = bestIdxs[sortIdxs]

	# ------------------------ lurn stuff

	bsfLocs = None
	bsfFilt = None

	if np.sum(x0 * x0) * kMax <= bsfScore: # highest score is kMax identical locs
		return -1, bsfLocs, bsfFilt

	# best combination of idxs such that none are within Lmin of each other
	# row = windowSims[i]
	# idxs = sub.optimalAlignment(dotProds, Lmin)

	# iteratively intersect with another near neighbor, compute the
	# associated score, and check if it's better (or if we can early abandon)
	intersection = x0
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
			nextWindow = Xblur[:, nextIdx:nextIdx+windowLen]
			nextIntersection = np.minimum(filt, nextWindow)
			nextFiltSum += nextIntersection
			nextFilt = nextFiltSum / (k+1)  # avg value of each feature in intersections
			# bigEnoughIntersection = nextIntersection[nextIntersection > minSim] # TODO was this
			bigEnoughIntersection = nextFilt[nextFilt > minSim]
			nextSz = np.sum(bigEnoughIntersection)
		else:
			nextSz = sz * p0
		enemySz = max(nextSz, noiseSz)

		score = (sz - enemySz) * k
		if k > 1 and score > bsfScore:
			bsfScore = score
			bsfLocs = sortedIdxs[:k]
			bsfFilt = np.copy(filt)

			print("seed {}, k={}, score={}, locs={} is the new best!".format(
				seedStartIdx, k, score, bsfLocs))
			print("------------------------")

		# early abandon if this can't possibly beat the best score, which
		# is the case exactly when the intersection is so small that perfect
		# matches at all future locations still wouldn't be good enough
		elif sz * numIdxs <= bsfScore:
			break
		elif noiseSz > nextSz:
			break

	return bsfScore, bsfLocs, bsfFilt


def findAllInstancesAvgBlur(X, Xblur, seedStartIdx, seedEndIdx, Lmin, Lmax, Lfilt,
	p0=-1, p0blur=-1, bsfScore=0, **sink):

	if seedStartIdx < 0:
		print("WARNING: findAllInstancesAvgBlur(); received invalid "
			"start idx {}".format(seedStartIdx))
	if seedEndIdx > Xblur.shape[1]:
		print("WARNING: findAllInstancesAvgBlur(); received invalid "
			"end idx {}".format(seedEndIdx))

	# ------------------------ stats
	if p0 <= 0.:
		p0 = np.mean(X) # fraction of entries that are 1 (roughly)
	if p0blur <= 0.:
		p0blur = np.mean(Xblur)

	windowLen = seedEndIdx - seedStartIdx # assume end idx not inclusive
	# minSim = p0
	minSim = p0blur
	expectedOnesPerWindow = p0blur * X.shape[0] * windowLen
	noiseSz = p0 * expectedOnesPerWindow # num ones to begin with

	# ------------------------ candidate location generation
	x0 = Xblur[:, seedStartIdx:seedEndIdx]
	if np.sum(x0) <= 0.:
		return -1, None, None
	dotProds = dotProdsWithAllWindows(x0, X)
	assert(np.min(dotProds) >= 0.)
	# dotProds[dotProds < noiseSz] = 0. # don't even consider places worse than noise

	# compute best locations to try and then sort them in decreasing order
	# bestIdxs = nonOverlappingMaxima(dotProds, Lmin)
	bestIdxs = sub.optimalAlignment(dotProds, Lmin)

	if len(bestIdxs) < 2:
		plt.figure()
		plt.plot(dotProds)
		plt.show()
		import sys
		sys.exit(1)

	bestProds = dotProds[bestIdxs]
	# keepIdxs = np.where(bestProds > noiseSz)[0]
	# bestIdxs = bestIdxs[keepIdxs]
	# bestProds = bestProds[keepIdxs]
	sortIdxs = np.argsort(bestProds)[::-1]
	idxs = bestIdxs[sortIdxs]

	# precompute sizes of data at locations so we can early abandon
	# we define remainingSizes[i] = sum(windowSizes[i:])
	# TODO uncomment to see if this bound actually helps anything
	# windows = [Xblur[:, idx:idx+windowLen] for idx in idxs]
	# windowSizes = np.array([np.sum(window) for window in windows])
	# remainingSizes = np.cumsum(windowSizes[::-1])[::-1]
	# remainingSizes = np.r[remainingSizes, 0]

	# ------------------------ now figure out which idxs should be instances
	# avg together the 2 best windows
	idx1, idx2 = idxs[:2]
	window1blur = Xblur[:, idx1:idx1+windowLen]
	window2blur = Xblur[:, idx2:idx2+windowLen]
	filtSums = window1blur + window2blur
	filt = filtSums / 2.
	filtSz = np.sum(filt[filt > minSim])
	# maxima = ar.localMaxFilterRows(filtBlurSums, allowEq=True)

	# plt.figure()
	# viz.imshowBetter(filt)
	# plt.show()

	# TODO remove
	return filtSz, idxs[:2], filt

	if filtSz <= noiseSz: # if best pair worse than noise, just return
		print "best pair at seed {} worse than noise!".format(seedStartIdx)
		return -1, None, None

	# print "seed {}: checking candidate idxs {}".format(seedStartIdx, idxs)

	bsfLocs = [0]
	bsfFilt = filt
	bestScore = -1

	nextFiltSums = filtSums
	nextFilt = np.copy(filt)
	nextFiltSz = filtSz

	idxs = np.append(idxs, -1) # dummy idx so everything can go in the loop
	k = 1
	for i, idx in enumerate(idxs[2:]): # for every next idx

		# if filtSz > noiseSz: # TODO see if skipping bad stuff helps
		k += 1
		filtSums = nextFiltSums
		filt = np.copy(nextFilt)
		filtSz = nextFiltSz

		if idx >= 0: # ignore final dummy idx
			# compute how big next filter will be; if basically the
			# same size, there's probably another instance--we're looking
			# for a big gap here
			nextWindow = Xblur[:, idx:idx+windowLen]
			nextFiltSums = filtSums + nextWindow
			nextFilt = nextFiltSums / (k + 1)
			nextFiltSz = np.sum(nextFilt[nextFilt > minSim])
			enemySz = max(noiseSz, nextFiltSz)
		else:
			enemySz = noiseSz

		score = (filtSz - enemySz) * k
		if score > bestScore: # should we include filt as an instance?
			bestScore = score
			bsfLocs = idxs[:k]
			bsfFilt = np.copy(filt)
			if score > bsfScore:
				print("seed {}, k={}, score={}, locs={} is the new best!".format(
					seedStartIdx, k, score, bsfLocs))
			# print("------------------------")

		# elif filtSz + remainingSizes[k] <= bsfScore: # TODO check if abandoning helps
		# 	break
		elif nextFiltSz <= noiseSz:
			break

	if len(bsfLocs) >= 2 and bestScore > bsfScore:
		return bestScore, bsfLocs, bsfFilt * (bsfFilt > minSim)

	return -1, None, None


	# # try re-computing dot prods using this avg to get better idxs
	# # TODO does this help?
	# dotProds = sig.correlate2d(X, x0, mode='valid')
	# # compute best locations to try and sort them in decreasing order
	# bestIdxs = nonOverlappingMinima(dotProds, Lmin)
	# bestProds = dotProds[bestIdxs]
	# sortIdxs = np.argsort(bestProds)[::-1]
	# idxs = bestIdxs[sortIdxs]


def findAllInstancesMAP(X, Xblur, seedStartIdx, seedEndIdx, Lmin, Lmax, Lfilt,
	p0=-1, p0blur=-1, logs_0=None, bsfScore=0, **sink):

	assert(np.min(X) >= 0.)
	assert(np.max(X) <= 1.)
	assert(np.min(Xblur) >= 0.)
	assert(np.max(Xblur) <= 1.)
	assert(np.all(np.sum(X, axis=1) > 0))
	assert(np.all(np.sum(Xblur, axis=1) > 0))

	# ================================ variable initialization
	windowLen = seedEndIdx - seedStartIdx # assume end idx not inclusive

	if p0 <= 0.:
		p0 = np.mean(X) # fraction of entries that are 1 (roughly)
	if p0blur <= 0.:
		p0blur = np.mean(Xblur)
	if logs_0 is None:
		# theta_0 = np.zeros(windowShape) + p0blur
		featureMeans = np.mean(Xblur, axis=1, keepdims=True)
		# featureSums = np.sum(Xblur, axis=1, keepdims=True) + 1. # regularize
		# featureMeans = featureSums / X.shape[1]
		theta_0 = np.ones((Xblur.shape[0], windowLen)) * featureMeans
		logs_0 = np.log(theta_0)

	lamda = 0

	# ================================ candidate location generation
	x0 = Xblur[:, seedStartIdx:seedEndIdx]
	if np.sum(x0) <= 0.:
		print "map() {}-{}: empty".format(seedStartIdx, seedEndIdx)
		return -1, None, None
	dotProds = dotProdsWithAllWindows(x0, X)

	# compute best locations to try and then sort them in decreasing order
	bestIdxs = nonOverlappingMaxima(dotProds, Lmin)

	bestProds = dotProds[bestIdxs]
	sortIdxs = np.argsort(bestProds)[::-1]
	idxs = bestIdxs[sortIdxs]

	# ================================ now figure out which idxs should be instances

	# initialize counts
	idx = idxs[0]
	counts = np.copy(X[:, idx:idx+windowLen])
	countsBlur = np.copy(Xblur[:, idx:idx+windowLen])

	bestOdds = -np.inf
	bestFilt = None
	bestLocs = None
	for i, idx in enumerate(idxs[1:]):
		k = i + 2.

		# update counts
		window = X[:, idx:idx+windowLen]
		windowBlur = Xblur[:, idx:idx+windowLen]
		counts += window
		countsBlur += windowBlur

		# our params
		theta_1 = countsBlur / k
		logs_1 = np.log(theta_1)
		logs_1[np.isneginf(logs_1)] = -999 # any non-inf number--will be masked by counts

		logDiffs = (logs_1 - logs_0)
		gains = counts * logDiffs # *must* use this so -999 is masked
		threshMask = gains > lamda
		# threshMask = logDiffs > 0
		threshMask *= theta_1 > .5
		# threshMask = theta_1 > .5
		# gains += (k - counts) * (logs_1c - logs_0c)
		filt = logDiffs * threshMask

		logOdds = np.sum(counts * filt)

		randomOdds = np.sum(filt) * p0blur * k
		nextWindowOdds = -np.inf
		if k < len(idxs):
			idx = idxs[k]
			nextWindow = X[:, idx:idx+windowLen]
			# nextWindowOdds = np.sum(filt * nextWindow)
			nextWindowOdds = np.sum(filt * nextWindow) * k
		penalty = max(randomOdds, nextWindowOdds)
		logOdds -= penalty

		if logOdds > bestOdds:
			bestOdds = logOdds
			bestFilt = np.copy(filt)
			bestLocs = idxs[:k]
			# print("k={}; log odds {}".format(k, logOdds))

	# print "map() {}: odds, locs {} {}".format(seedStartIdx, bestOdds, len(bestLocs))
	return bestOdds, bestLocs, bestFilt

# this just got too cluttered...
def old_findAllInstancesMAP(X, Xblur, seedStartIdx, seedEndIdx, Lmin, Lmax, Lfilt,
	p0=-1, p0blur=-1, logs_0=None, bsfScore=0, **sink):

	# print "map(): X, Xblur shape", X.shape, Xblur.shape

	# ------------------------ stats
	windowLen = seedEndIdx - seedStartIdx # assume end idx not inclusive

	if p0 <= 0.:
		p0 = np.mean(X) # fraction of entries that are 1 (roughly)
	if p0blur <= 0.:
		p0blur = np.mean(Xblur)
		p0blur *= 2
	if logs_0 is None:
		# featureMeans = np.mean(Xblur, axis=1).reshape((-1, 1))
		# featureMeans = np.mean(Xblur, axis=1).reshape((-1, 1)) * 2
		# theta_0 = np.ones((Xblur.shape[0], windowLen)) * featureMeans
		theta_0 = np.zeros((Xblur.shape[0], windowLen)) + p0blur
		# theta_0 = np.zeros((Xblur.shape[0], windowLen)) + 2 * p0blur
		theta_0c = 1. - theta_0
		logs_0 = np.log(theta_0)
		logs_0c = np.log(theta_0c)
		# logs_0 = np.zeros((Xblur.shape[0], windowLen)) + np.log(p0blur) # overzealous
		# logs_0c = np.zeros((Xblur.shape[0], windowLen)) + np.log(1. - p0blur)
		# logs_0 = np.zeros((Xblur.shape[0], windowLen)) + np.log(p0) # works
	# print "logs_0 nan at:", np.where(np.isnan(logs_0))[0]

	# lamda = -2 * np.log(p0blur) - .001 # just below what we get with 2 instances
	# lamda = -np.log(p0blur) - .001 # actually, including log(theta_0), should be this
	lamda = 0
	# print "lambda", lamda

	# minSim = p0
	# expectedOnesPerWindow = p0blur * X.shape[0] * windowLen
	# noiseSz = p0blur * expectedOnesPerWindow # num ones to begin with

	# ------------------------ candidate location generation
	x0 = Xblur[:, seedStartIdx:seedEndIdx]
	if np.sum(x0) <= 0.:
		print "map() {}-{}: empty".format(seedStartIdx, seedEndIdx)
		return -1, None, None
	# dotProds = dotProdsWithAllWindows(x0, X)
	dotProds = dotProdsWithAllWindows(x0, Xblur)
	# assert(np.min(dotProds) >= 0.)
	# dotProds[dotProds < noiseSz] = 0. # don't even consider places worse than noise

	# compute best locations to try and then sort them in decreasing order
	bestIdxs = nonOverlappingMaxima(dotProds, Lmin)
	# bestIdxs = sub.optimalAlignment(dotProds, Lmin)

	bestProds = dotProds[bestIdxs]
	# keepIdxs = np.where(bestProds > noiseSz)[0]
	# bestIdxs = bestIdxs[keepIdxs]
	# bestProds = bestProds[keepIdxs]
	sortIdxs = np.argsort(bestProds)[::-1]
	idxs = bestIdxs[sortIdxs]

	# ------------------------ now figure out which idxs should be instances
	# avg together the 2 best windows and compute the score
	idx1, idx2 = idxs[:2]
	window1 = X[:, idx1:idx1+windowLen]
	window2 = X[:, idx2:idx2+windowLen]
	counts = window1 + window2
	window1blur = Xblur[:, idx1:idx1+windowLen]
	window2blur = Xblur[:, idx2:idx2+windowLen]
	countsBlur = window1blur + window2blur

	# compute best pair filter compared to noise
	k = 2.
	theta_1 = countsBlur / k
	# logs_1 = np.zeros(theta_1.shape)
	logs_1 = np.log(theta_1)
	logs_1[np.isneginf(logs_1)] = -99999. # big negative num
	# logs_1c = np.log(1. - theta_1)
	# logs_1c[np.isneginf(logs_1c)] = -99999.
	# gains = k * (logs_1 - logs_0) - lamda
	# gains = counts * (logs_1 - logs_0) - lamda
	# filt = gains * (gains > 0)
	# logOdds = np.sum(filt)
	logDiffs = (logs_1 - logs_0)
	gains = counts * logDiffs
	threshMask = gains > lamda
	# gains += (k - counts) * (logs_1c - logs_0c)
	filt = logDiffs * threshMask
	# logOdds = np.sum(filt) - lamda * np.count_nonzero(filt)
	# logOdds = np.sum(filt)

	# subtract1 = np.minimum(window1, theta_1)
	# subtract2 = np.minimum(window1, theta_1)
	# subtracts = [subtract1, subtract2]
	# for idx, subVal in zip(idxs[:2], subtracts):
	# 	Xblur[:, idx:idx+windowLen] -= subVal

	# # compute best pair filter compared to nearest enemy
	# if len(idxs) > 2:
	# 	idx = idxs[k]
	# 	nextWindow = Xblur[:, idx:idx+windowLen]
	# 	# theta_e = nextWindow
	# 	theta_e = np.maximum(nextWindow, theta_0)
	# 	logs_e = np.log(theta_e)
	# 	logs_e[np.isneginf(logs_e)] = -99999.
	# 	# logs_ec = np.log(1. - theta_e)
	# 	# logs_ec[np.isneginf(logs_ec)] = -99999.
	# 	gains_e = counts * (logs_1 - logs_e)
	# 	# gains_e = nextWindow * (logs_1 - logs_e)
	# 	filt_e = gains * (gains_e > lamda)
	# 	# logOdds_e = np.sum(filt_e) - lamda * np.count_nonzero(filt_e)
	# 	logOdds_e = np.sum(filt_e)
	# 	# if logOdds_e < logOdds:
	# 	if True:
	# 		# print("k=2; enemy log odds {} < noise log odds {}".format(
	# 		# 	logOdds_e, logOdds))
	# 		logOdds = logOdds_e

	# logOdds = np.sum(filt * window1) - lamda * np.count_nonzero(filt)
	# logOdds = np.sum(filt * window1)

	logOdds = np.sum(filt * X[:, idx1:idx1+windowLen])
	# compute best pair filter compared to nearest enemy
	if k < len(idxs):
		idx = idxs[k]
		# nextWindow = Xblur[:, idx:idx+windowLen]
		nextWindow = X[:, idx:idx+windowLen]
		nextWindowOdds = np.sum(filt * nextWindow)
		# nextWindowOdds -= lamda * np.count_nonzero(filt)
		logOdds -= nextWindowOdds

	# logOdds = np.sum(filt * counts) - lamda * np.count_nonzero(filt)
	# # compute best pair filter compared to nearest enemy
	# if k < len(idxs):
	# 	idx = idxs[k]
	# 	nextWindowBlur = Xblur[idx:idx+windowLen]
	# 	theta_e = nextWindowBlur
	# 	logs_e = np.log(theta_e)
	# 	logs_e[np.isneginf(logs_e)] = -99999.
	# 	filt = logs_e
	# 	# logDiffs = (logs_1 - logs_0)
	# 	# gains = counts * logDiffs
	# 	# threshMask = gains > lamda
	# 	# # gains += (k - counts) * (logs_1c - logs_0c)
	# 	# filt = logDiffs * threshMask
	# 	gains_e =
	# 	logOdds_e =


	# if np.mean(x0) > (p0blur * 2):
	# 	fig, axes = plt.subplots(2, 5, figsize=(10,8))
	# 	axes = axes.flatten()
	# 	plt.suptitle("{}-{}".format(idx1, idx2))
	# 	axes[0]
	# 	viz.imshowBetter(window1, ax=axes[0])
	# 	viz.imshowBetter(window2, ax=axes[1])
	# 	viz.imshowBetter(counts, ax=axes[2])
	# 	viz.imshowBetter(gains, ax=axes[3])
	# 	viz.imshowBetter(filt, ax=axes[4])
	# 	axes[4].set_title("{}".format(logOdds))

	# 	viz.imshowBetter(X[:, idx1:idx1+windowLen], ax=axes[5])
	# 	viz.imshowBetter(X[:, idx2:idx2+windowLen], ax=axes[6])

	bestOdds = logOdds
	bestFilt = filt
	bestLocs = idxs[:2]

	# print "map() {} -> {} {}:".format(seedStartIdx, idx1, idx2), logOdds, bestLocs
	# print "map() {}: k=2, total prob: {}".format(seedStartIdx, np.sum(theta_1))
	# return logOdds, bestLocs, bestFilt # TODO remove after debug



	# Alright, so what I want the right answer to be is the biggest patch
	# of black I can get, subtracting off the next-biggest patch of black
	# (weighting each by the filter or something)
	# -so maybe that's the log odds of the Nth loc - the log odds of the Nth loc
	#	-and note that these are log odds *of the loc being an instance*, not
	# 	the log odds of the filter as a whole
	#		-although pretty sure there's some formulation of filter log odds
	#		such that these are equivalent--it just isn't my current logOdds var



	# Xblur = np.copy(Xblur)
	# Xblur[:, idx1:idx1+windowLen] = np.maximum(0, Xblur[:, idx1:idx1+windowLen] - theta_1)
	# Xblur[:, idx2:idxs2+windowLen] = np.maximum(0, Xblur[:, idx2:idx2+windowLen] - theta_1)

	# try adding the rest of the idxs
	for i, idx in enumerate(idxs[2:]):
		k += 1
		window = Xblur[:, idx:idx+windowLen]

		# filter and odds vs noise
		counts += window
		theta_1 = counts / k
		logs_1 = np.log(theta_1)
		logs_1[np.isneginf(logs_1)] = -99999. # big negative num
		# logs_1c = np.log(1. - theta_1)
		# logs_1c[np.isneginf(logs_1c)] = -99999.
		# gains = k * (logs_1 - logs_0) - lamda
		# gains = counts * (logs_1 - logs_0) - lamda
		# filt = gains * (gains > 0)
		# logOdds = np.sum(filt)
		logDiffs = (logs_1 - logs_0)
		threshMask = gains > lamda
		# gains += (k - counts) * (logs_1c - logs_0c)
		filt = logDiffs * threshMask
		# logOdds = np.sum(filt) - lamda * np.count_nonzero(filt)
		# logOdds = np.sum(filt)

		# logOdds = np.sum(filt * window) - lamda * np.count_nonzero(filt)
		# logOdds = np.sum(filt * window)

		logOdds = np.sum(filt * X[:, idx:idx+windowLen])
		# compute best pair filter compared to nearest enemy
		# randomOdds = np.sum(filt) # Wait, was this even intentional
		randomOdds = np.sum(filt) * p0blur
		nextWindowOdds = 0
		if k < len(idxs):
			idx = idxs[k]
			# nextWindow = Xblur[:, idx:idx+windowLen]
			nextWindow = X[:, idx:idx+windowLen]
			nextWindowOdds = np.sum(filt * nextWindow)
			# nextWindowOdds -= lamda * np.count_nonzero(filt)
			# logOdds -= nextWindowOdds
		penalty = max(randomOdds, nextWindowOdds)
		logOdds -= penalty

		# subtract = np.minimum(window, theta_1)
		# subtracts.append(subtract)
		# Xblur[:, idx:idx+windowLen] -= subtract

		# # filter and odds vs nearest enemy
		# if k < len(idxs):
		# 	idx = idxs[k]
		# 	nextWindow = Xblur[:, idx:idx+windowLen]
		# 	# theta_e = nextWindow
		# 	theta_e = np.maximum(nextWindow, theta_0)
		# 	logs_e = np.log(theta_e)
		# 	logs_e[np.isneginf(logs_e)] = -99999.
		# 	# logs_ec = np.log(1. - theta_e)
		# 	# logs_ec[np.isneginf(logs_ec)] = -99999.
		# 	gains_e = counts * (logs_1 - logs_e)
		# 	# gains_e = nextWindow * (logs_1 - logs_e)
		# 	# gains_e += (k - counts) * (logs_1c - logs_ec)
		# 	filt_e = gains * (gains_e > lamda)
		# 	# logOdds_e = np.sum(filt_e) - lamda * np.count_nonzero(filt_e)
		# 	logOdds_e = np.sum(filt_e)
		# 	# logOdds = min(logOdds, logOdds_e)
		# 	# if logOdds_e < logOdds:
		# 	if True:
		# 		# print("k={}; enemy log odds {} < noise log odds {}".format(
		# 		# 	k, logOdds_e, logOdds))
		# 		logOdds = logOdds_e

		# Xblur[:, idx:idx+windowLen] = np.maximum(0, Xblur[:, idx:idx+windowLen] - theta_1)

		# print "map() {}: k={}, total prob: {}".format(seedStartIdx, k, np.sum(theta_1))

		if logOdds > bestOdds:
			bestOdds = logOdds
			bestFilt = np.copy(filt)
			bestLocs = idxs[:k]
			# print("k={}; log odds {}".format(k, logOdds))

	# print "map() {}: odds, locs {} {}".format(seedStartIdx, bestOdds, len(bestLocs))

	# for idx, subVal in zip(idxs[:len(subtracts)], subtracts):
	# 	Xblur[:, idx:idx+windowLen] += subVal

	return bestOdds, bestLocs, bestFilt


def findAllInstancesFromSeedLoc(X, Xblur, seedStartIdx, seedEndIdx, Lmin, Lmax,
	Lfilt=0, generalizeSeedsAlgo=None, **kwargs):

	if generalizeSeedsAlgo == 'submodular':
		return findAllInstancesSubmodular(X, Xblur, seedStartIdx, seedEndIdx,
			Lmin, Lmax, Lfilt, **kwargs)
	elif generalizeSeedsAlgo == 'avg':
		return findAllInstancesAvgBlur(X, Xblur, seedStartIdx, seedEndIdx,
			Lmin, Lmax, Lfilt, **kwargs)
	elif generalizeSeedsAlgo == 'map':
		return findAllInstancesMAP(X, Xblur, seedStartIdx, seedEndIdx,
			Lmin, Lmax, Lfilt, **kwargs)
	else:
		raise ValueError("instance generalization method "
			"{} not recognized".format(generalizeSeedsAlgo))


def findInstancesUsingSeedLocs(X, Xblur, seedStartIdxs, seedEndIdxs, Lmin,
	Lmax, Lfilt, windowLen=None, **kwargs):

	if (len(seedStartIdxs) != len(seedEndIdxs)) or (not len(seedStartIdxs)):
		raise ValueError("must supply same number of start and end indices > 0!"
			"Got {} and {}".format(len(seedStartIdxs), len(seedEndIdxs)))

	print "findInstancesUsingSeedLocs(): got seedStartIdxs: ", seedStartIdxs

	if windowLen <= 0:
		windowLen = seedEndIdxs[0] - seedStartIdxs[0]
	elif windowLen < Lmax:
		raise ValueError("windowLen {} < Lmax {}".format(windowLen, Lmax))

	# precompute feature mat stats so that evaluations of each seed
	# don't have to duplicate work
	p0 = np.mean(X)
	p0blur = np.mean(Xblur)
	featureMeans = np.mean(Xblur, axis=1, keepdims=True)
	theta_0 = np.ones((Xblur.shape[0], windowLen)) * featureMeans
	logs_0 = np.log(theta_0)

	# hack: if using submodular generalization algo, precompute pairwise
	# dot prods of windows using convolution
	useSubmodular = kwargs.get('generalizeSeedsAlgo') == 'submodular' and windowLen
	# useSubmodular = False # yeah, definitely slower when using all possible seeds
	if useSubmodular:
		print "computing all windowSims"
		colSims = np.dot(X.T, Xblur)
		filt = np.zeros((windowLen, windowLen)) + np.diag(np.ones(windowLen)) # zeros except 1s on diag
		windowSims = sig.convolve2d(colSims, filt, mode='valid')
		print "computed all windowSims"

	bsfScore = 0
	bsfFilt = None
	bsfLocs = None
	bsfSeedIdx = -1
	for i in range(len(seedStartIdxs)):
		startIdx, endIdx = seedStartIdxs[i], seedEndIdxs[i]
		# if i % 10 == 0:
		# 	print "trying seed startIdx, endIdx ", startIdx, endIdx

		if useSubmodular: # hack to make this run faster for prototyping
			kwargs['dotProds'] = windowSims[i]

		score, locs, filt = findAllInstancesFromSeedLoc(X, Xblur, startIdx, endIdx,
			Lmin, Lmax, Lfilt, p0=p0, p0blur=p0blur, logs_0=logs_0,
			bsfScore=bsfScore, **kwargs)
		# if i % 10 == 0:
		# 	print "idx {}: got score, locs, filtShape: {} {} {}".format(
		# 		startIdx, score, locs, filt.shape if filt is not None else None)

		if score > bsfScore:
			bsfScore = score
			bsfFilt = np.copy(filt)
			bsfLocs = np.copy(locs)
			bsfSeedIdx = startIdx
			print "findInstancesUsingSeedLocs(): got new best score", bsfScore

	print "findInstancesUsingSeedLocs(): got best seed, score", bsfSeedIdx, bsfScore
	print "findInstancesUsingSeedLocs(): got best locs ", bsfLocs
	# print "findInstancesUsingSeedLocs(): X shape, Xblur shape ", X.shape, Xblur.shape

	return bsfScore, bsfLocs, bsfFilt


def extractTrueLocs(X, Xblur, bsfLocs, bsfFilt, windowLen, Lmin, Lmax,
	extractTrueLocsAlgo=DEFAULT_EXTRACT_LOCS_ALGO, **sink):

	if extractTrueLocsAlgo == 'none':
		return bsfLocs, bsfLocs + Lmax

	# determine expected value of an element of X (or, alternatively, Xblur)
	if extractTrueLocsAlgo == 'x':
		p0 = np.mean(X)
	else:
		p0 = np.mean(Xblur)

	if bsfFilt is None:
		print "WARNING: extractTrueLocs(): received None as filter"
		return np.array([0]), np.array([1])

	print "extractTrueLocs(): bsf locs", bsfLocs
	print "extractTrueLocs(): bsfFilt shape", bsfFilt.shape

	# compute the total filter weight in each column, ignoring low values
	bsfFiltWindow = np.copy(bsfFilt)
	# minSim = p0
	# bsfFiltWindow *= bsfFiltWindow >= minSim
	sums = np.sum(bsfFiltWindow, axis=0)

	# subtract off the amount of weight that we'd expect in each column by chance
	kBest = len(bsfLocs)
	expectedOnesFrac = np.power(p0, kBest-1) # this is like 0; basically no point
	expectedOnesPerCol = expectedOnesFrac * X.shape[0]
	sums -= expectedOnesPerCol

	# # at least for a couple msrc examples, these are basically flat--which makes sense
	# plt.figure()
	# plt.plot(sums)
	# plt.plot(np.zeros(len(sums)) + expectedOnesPerCol)
	# # from ..utils.misc import nowAsString
	# # plt.savefig('/Users/davis/Desktop/ts/figs/msrc/sums-{}.pdf'.format(nowAsString()))
	# plt.show()
	# plt.close()

	# pick the optimal set of indices to maximize the sum of sequential column sums
	start, end, _ = maxSubarray(sums)

	# ensure we picked at least Lmin points
	sumsLength = len(sums)
	while end - start < Lmin:
		nextStartVal = sums[start-1] if start > 0 else -np.inf
		nextEndVal = sums[end] if end < sumsLength else -np.inf
		if nextStartVal > nextEndVal:
			start -= 1
		else:
			end += 1
	# ensure we picked at most Lmax points
	while end - start > Lmax:
		if sums[start] > sums[end-1]:
			end -= 1
		else:
			start += 1

	locs = np.sort(np.asarray(bsfLocs))
	startIdxs = locs + start
	endIdxs = locs + end

	# # reconcile overlap; we first figure out how much we like the start vs end
	# # for different amounts of overlap
	# startSums = np.cumsum(sums)
	# endSums = np.cumsum(sums[::-1])
	# # gaps = startIdxs[1:] - startIdxs[:-1]
	# for i in range(len(startIdxs) - 1):
	# 	te1, ts2 = endIdxs[i], startIdxs[i+1]
	# 	gap = ts2 - te1
	# 	if gap > 0:
	# 		continue

	# 	# figure out best amount by which to crop start and end indices
	# 	overlap = -gap + 1
	# 	bestSplitCost = np.inf
	# 	bestMoveStart = -1
	# 	for moveStartThisMuch in range(0, overlap):
	# 		moveEndThisMuch = overlap - moveStartThisMuch
	# 		startCost = startSums[moveStartThisMuch-1] if moveStartThisMuch else 0.
	# 		endCost = endSums[moveEndThisMuch-1] if moveEndThisMuch else 0.
	# 		cost = startCost + endCost
	# 		if cost < bestSplitCost:
	# 			bestSplitCost = cost
	# 			bestMoveStart = moveStartThisMuch

	# 	startIdxs[i+1] += bestMoveStart
	# 	endIdxs[i] -= (overlap - bestMoveStart)

	if len(startIdxs) > 2:
		lengths = endIdxs - startIdxs
		maxInternalLength = np.max(lengths[1:-1])
		startIdxs[0] = max(startIdxs[0], endIdxs[0] - maxInternalLength)
		endIdxs[-1] = min(endIdxs[-1], startIdxs[-1] + maxInternalLength)

	print "extractTrueLocs(): startIdxs, endIdxs", startIdxs, endIdxs

	return startIdxs, endIdxs

def computeAllSeedIdxsFromPair(seedIdxs, numShifts, stepLen):
	for idx in seedIdxs[:]:
		i = idx
		j = idx
		for shft in range(numShifts):
			i -= stepLen
			j += stepLen
			seedIdxs += [i, j]

	seedIdxs = np.unique(seedIdxs)
	return seedIdxs

# TODO set algo to 'pair' after debug
def learnFF(seq, X, Xblur, Lmin, Lmax, Lfilt,
	generateSeedsAlgo=DEFAULT_SEEDS_ALGO,
	generalizeSeedsAlgo=DEFAULT_GENERALIZE_ALGO,
	extractTrueLocsAlgo=DEFAULT_EXTRACT_LOCS_ALGO,
	generateSeedsStep=.1, padBothSides=True, **generalizeKwargs):

	padLen = (len(seq) - X.shape[1]) // 2
	if padBothSides:
		X = ar.addZeroCols(X, padLen, prepend=True)
		X = ar.addZeroCols(X, padLen, prepend=False)
		Xblur = ar.addZeroCols(Xblur, padLen, prepend=True)
		Xblur = ar.addZeroCols(Xblur, padLen, prepend=False)

	tStartSeed = time.clock()

	# find seeds; i.e., candidate instance indices from which to generalize
	numShifts = int(1. / generateSeedsStep) + 1
	stepLen = int(Lmax * generateSeedsStep)
	windowLen = Lmax + stepLen
	print "learnFF(): stepLen, numShifts", stepLen, numShifts

	if generateSeedsAlgo == 'pair':
		searchLen = (Lmin + Lmax) // 2
		motif = findMotifOfLengthFast([seq], searchLen)
		seedIdxs = [motif.idx1, motif.idx2]
		print "seedIdxs from motif: ", seedIdxs
		seedIdxs = computeAllSeedIdxsFromPair(seedIdxs, numShifts, stepLen)

	elif generateSeedsAlgo == 'all':
		seedIdxs = np.arange(X.shape[1] - windowLen) # TODO remove after debug

	elif generateSeedsAlgo == 'random':
		seedIdxs = list(np.random.choice(np.arange(len(seq) - Lmax), 2))
		seedIdxs = computeAllSeedIdxsFromPair(seedIdxs, numShifts, stepLen)

	elif generateSeedsAlgo == 'walk':
		# score all subseqs based on how much they don't look like random walks
		# when examined using different sliding window lengths
		scores = np.zeros(len(seq))
		for dim in range(seq.shape[1]):
			# compute these just once, not once per length
			dimData = seq[:, dim].ravel()
			diffs = dimData[1:] - dimData[:-1]
			std = np.std(diffs)
			for divideBy in [1, 2, 4, 8]:
				partialScores = windowScoresRandWalk(dimData, Lmin // divideBy,
					std=std)
				scores[:len(partialScores)] += partialScores

		# figure out optimal seeds based on scores of all subseqs
		bestIdx = np.argmax(scores)
		start = max(0, bestIdx - Lmin)
		end = min(len(scores), start + Lmin)
		scores[start:end] = -1
		secondBestIdx = np.argmax(scores)

		seedIdxs = [bestIdx, secondBestIdx]
		seedIdxs = computeAllSeedIdxsFromPair(seedIdxs, numShifts, stepLen)
	else:
		raise NotImplementedError("Only algo 'pair' supported to generate seeds"
			"; got unrecognized algo {}".format(generateSeedsAlgo))

	# compute start and end indices of seeds to try
	seedStartIdxs = np.sort(np.array(seedIdxs))
	seedStartIdxs = seedStartIdxs[seedStartIdxs >= 0]
	seedStartIdxs = seedStartIdxs[seedStartIdxs < X.shape[1] - windowLen]
	seedEndIdxs = seedStartIdxs + windowLen

	print "learnFF(): seedIdxs after removing invalid idxs: ", seedStartIdxs
	print "learnFF(): fraction of idxs used as seeds: {}".format(
		len(seedStartIdxs) / float(len(seq)))

	tEndSeed = time.clock()

	generalizeKwargs['windowLen'] = windowLen # TODO remove after prototype

	bsfScore, bsfLocs, bsfFilt = findInstancesUsingSeedLocs(X, Xblur,
		seedStartIdxs, seedEndIdxs, Lmin, Lmax, Lfilt,
		generalizeSeedsAlgo=generalizeSeedsAlgo,
		**generalizeKwargs)

	# print "learnFF(): got bsfFilt shape", bsfFilt.shape

	startIdxs, endIdxs = extractTrueLocs(X, Xblur, bsfLocs, bsfFilt, windowLen,
		Lmin, Lmax, extractTrueLocsAlgo=extractTrueLocsAlgo)

	tEndFF = time.clock()
	print "learnFF(): seconds to find seeds, locs, total =\n\t{:.3f}\t{:.3f}\t{:.3f}".format(
		tEndSeed - tStartSeed, tEndFF - tEndSeed, tEndFF - tStartSeed)

	return startIdxs, endIdxs, bsfFilt


def learnFFfromSeq(seq, Lmin, Lmax, Lfilt=0, extendEnds=True, **kwargs):
	Lmin = int(len(seq) * Lmin) if Lmin < 1. else Lmin
	Lmax = int(len(seq) * Lmax) if Lmax < 1. else Lmax
	Lfilt = int(len(seq) * Lfilt) if Lfilt < 1. else Lfilt

	if not Lfilt or Lfilt < 0:
		Lfilt = Lmin

	print "learnFFfromSeq(): seqShape {}; using Lmin, Lmax, Lfilt= {}, {}, {}".format(
		seq.shape, Lmin, Lmax, Lfilt)

	# extend the first and last values out so that features using
	# longer windows are present for more locations (if requested)
	padLen = 0
	origSeqLen = len(seq)
	if extendEnds:
		padLen = Lmax # TODO uncomment after debug
		# padLen = Lmax - Lfilt # = windowLen
		seq = extendSeq(seq, padLen, padLen)
		# print "learnFFfromSeq(): extended seq to shape {}".format(seq.shape)

	# plt.figure()
	# plt.plot(seq)
	# print kwargs

	X = buildFeatureMat(seq, Lmin, Lmax, **kwargs)

	# plt.figure()
	# viz.imshowBetter(X)
	# plt.figure()
	# viz.imshowBetter(Xblur)

	X, Xblur = preprocessFeatureMat(X, Lfilt, **kwargs)
	if extendEnds: # undo padding after constructing feature matrix
		X = X[:, padLen:-padLen]
		Xblur = Xblur[:, padLen:-padLen]
		seq = seq[padLen:-padLen]

	# catch edge case where all nonzeros were in the padding
	keepRowIdxs = ar.nonzeroRows(X, thresh=1.)
	X = X[keepRowIdxs]
	Xblur = Xblur[keepRowIdxs]

	assert(np.min(X) >= 0.)
	assert(np.max(X) <= 1.)
	assert(np.min(Xblur) >= 0.)
	assert(np.max(Xblur) <= 1.)
	assert(np.all(np.sum(X, axis=1) > 0))
	assert(np.all(np.sum(Xblur, axis=1) > 0))

	# plt.figure()
	# viz.imshowBetter(X)
	# plt.figure()
	# viz.imshowBetter(Xblur)

	# these are identical to ff8's if we use padLen = Lmax - Lfilt
	# print "sums:", np.sum(seq), np.sum(X), np.sum(Xblur)

	startIdxs, endIdxs, bsfFilt = learnFF(seq, X, Xblur, Lmin, Lmax, Lfilt, **kwargs)

	# print "learnFFfromSeq(): got bsfFilt shape", bsfFilt.shape

	# account for fact that X is computed based on middle of data
	offset = (origSeqLen - X.shape[1]) // 2
	print "learnFFfromSeq(): X offset = ", offset
	# startIdxs += offset
	# endIdxs += offset
	# startIdxs -= offset
	# endIdxs = offset

	return startIdxs, endIdxs, bsfFilt, X, Xblur


if __name__ == '__main__':
	from doctest import testmod
	testmod()

	# x = np.ones((2,2))
	# X = np.arange(12).reshape((2, -1))
	# print dotProdsWithAllWindows(x, X)

