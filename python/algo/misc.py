#!/usr/env/python

import numpy as np
import matplotlib.pyplot as plt
# from scipy.ndimage import filters

from scipy.misc import logsumexp
# from scipy.stats import describe
from sklearn.decomposition import TruncatedSVD

from ..datasets import datasets
from ..datasets import synthetic as synth
from ..utils import arrays as ar
from ..utils import sliding_window as window

from numba import jit

def describe(x, name=""):
	return "{}: shape={}, min, mean, max = {:.3g}, {:.3g}, {:.3g}".format(name, np.shape(x),
		np.min(x), np.mean(x), np.max(x))

def notNoiseScore(x):
	# abs(sum(diffs(x)) / sum(|diffs(x)|) )
	diffs = x[1:] - x[:-1]
	absDiffs = np.abs(diffs)
	return np.abs(np.sum(diffs) / np.sum(absDiffs))

def cumDiffsAndAbsDiffs(seq, square=False):
	diffs = seq[1:] - seq[:-1]

	absDiffs = np.abs(diffs)
	if square:
		diffs *= absDiffs
		absDiffs *= absDiffs

	cumDiffs = np.cumsum(diffs, axis=0)
	cumAbsDiffs = np.cumsum(absDiffs, axis=0)

	return cumDiffs, cumAbsDiffs


def old_slidingNotNoiseScoreAtLength(seq, length, cumDiffs=None, cumAbsDiffs=None,
	useVariance=True):
	if cumDiffs is None or cumAbsDiffs is None:
		cumDiffs, cumAbsDiffs = cumDiffsAndAbsDiffs(seq)

	windowDiffs = cumDiffs[length:] - cumDiffs[:-length]
	windowAbsDiffs = cumAbsDiffs[length:] - cumAbsDiffs[:-length]
	windowRatios = windowDiffs / windowAbsDiffs
	windowRatios = np.nan_to_num(windowRatios)
	windowScores = np.abs(windowRatios)

	if useVariance and False:
		# # compute variance in each sliding window position
		# cumSums = np.cumsum(seq, axis=0)
		# windowSums = cumSums[length:] - cumSums[:-length]
		# windowMeans = windowSums / length

		# cumSumsSq = np.cumsum(seq*seq, axis=0)
		# windowSumsSq = cumSumsSq[length:] - cumSumsSq[:-length]
		# windowMeansSq = windowSumsSq / length

		# windowVariances = windowMeansSq - windowMeans * windowMeans

		# # weight window positions by relative variance
		# windowStds = np.sqrt(windowVariances)
		# windowStds = np.maximum(windowStds, 0)
		windowStds = np.sqrt(slidingVariance(seq, length))
		windowWeights = windowStds / np.max(windowStds, axis=0)
		windowScores *= windowWeights[:-1]

	print np.argmin(windowScores), np.min(windowScores)
	assert(np.min(windowScores) > -.001)
	assert(np.max(windowScores) < 1.0001)

	print "seq shape", seq.shape
	print "diffs shape", cumDiffs.shape
	print "window diffs shape", windowDiffs.shape
	print "window scores shape", windowScores.shape

	# return windowScores

	# filt = np.ones(2 * length + 1)
	# scores = filters.convolve1d(windowScores, weights=filt, axis=1, mode='constant')

	# stuff in the first

	cumWindowScores = np.cumsum(windowScores, axis=0)
	scores = np.zeros(seq.shape)
	numWindows = len(windowScores)
	assert(len(cumWindowScores) == len(windowScores))
	for i in range(len(seq)):
		firstWindowIncludingThis = max(0, i - length + 1)
		firstWindowIncludingThis = min(numWindows - 1, firstWindowIncludingThis)
		lastWindowIncludingThis = min(numWindows - 1, i)
		startScore = cumWindowScores[firstWindowIncludingThis]
		endScore = cumWindowScores[lastWindowIncludingThis]
		scores[i] = endScore - startScore

	scores /= length

	# # add up the scores from each window a given time step is part of
	# scores = np.zeros(seq.shape)
	# # for i in range(length, 2 * length - 1):
	# for i in range(length):
	# 	startIdx, endIdx = i, i + len(windowScores)
	# 	scores[startIdx:endIdx] += windowScores
	# 	# startIdx += length
	# 	# endIdx += length
	# 	# scores[startIdx:endIdx] += windowScores[:-length]

	# scores /= length

	# for i in range(length, 2 * length - 1):
	# 	startIdx, endIdx = i + length, i + length + len(windowScores)
	# 	scores[startIdx:endIdx] += windowScores

	# scores /= 2 * length - 1

	return scores


def slidingVariance(seq, length):
	cumSums = np.cumsum(seq, axis=0)
	windowSums = cumSums[length:] - cumSums[:-length]
	windowMeans = windowSums / length

	cumSumsSq = np.cumsum(seq*seq, axis=0)
	windowSumsSq = cumSumsSq[length:] - cumSumsSq[:-length]
	windowMeansSq = windowSumsSq / length

	windowVariances = windowMeansSq - windowMeans * windowMeans

	# print "variances stats", describe(windowVariances)

	return np.maximum(0, windowVariances) # deal with numerical instability


def windowScores2(seq, length):
	# windowVariances = slidingVariance(seq, length)
	diffs = seq[1:] - seq[:-1]
	diffs_2 = diffs[1:] - diffs[:-1]
	absDiffs_2 = np.abs(diffs_2)
	expectedAbsDiffs_2 = diffs[:-1]

	sigma_diff = np.std(diffs)

	print "seq shape, length", seq.shape, length
	print "sigma diff", sigma_diff

	# compute log prob of each window under noise model (iid gaussian
	# first derivatives, 0 mean and global sigma)
	firstProbs_noise = -.5 * (diffs / sigma_diff)**2
	secondProbs_noise = -.5 * ((absDiffs_2 - expectedAbsDiffs_2) / sigma_diff)**2
	sigmaProbs_noise = -.5 * np.log(2*np.pi) - np.log(sigma_diff)
	sigmaProbs_noise *= 2 # using this sigma for firstProbs and secondProbs
	logProbs_noise = firstProbs_noise[:-1] + secondProbs_noise + sigmaProbs_noise

	print "firstProbs stats", describe(firstProbs_noise)
	print "secondProbs stats", describe(secondProbs_noise)
	print "sigmaProbs stats", describe(sigmaProbs_noise)
	print "raw noise log probs stats", describe(logProbs_noise)

	cumLogProbs_noise = np.cumsum(logProbs_noise, axis=0)
	logProbs_noise = cumLogProbs_noise[length:] - cumLogProbs_noise[:-length]

	# compute log prob of each window under "pattern" model (gaussian
	# first derivs with gaussian difference between successive first
	# derivs, MLE variance for both first derivs and differences therein)
	# Note that we use a simplification to compute log prob of the data
	# under MLE params, which lets us just use the number of points, not
	# actual squared values
	diffsVariances = slidingVariance(diffs, length)
	diffsVariances_2 = slidingVariance(absDiffs_2, length)
	# diffsVariances[diffsVariances == 0.] = np.inf
	firstProbs_pat = - length * (.5 + (.5 * np.log(2*np.pi)) - .5 * np.log(diffsVariances))
	secondProbs_pat = -length * (.5 + (.5 * np.log(2*np.pi)) - .5 * np.log(diffsVariances_2))

	logProbs_pat = (firstProbs_pat[:-1] + secondProbs_pat)
	ignoreIdxs = np.isinf(logProbs_pat)
	# logProbs_pat[np.isnan(logProbs_pat)] = -1e6 # 0 variance -> flat signal -> noise
	# logProbs_pat[np.isinf(logProbs_pat)] = -1e6 # 0 variance -> flat signal -> noise
	logProbs_pat[np.isinf(logProbs_pat)] = 0 # 0 variance -> flat signal -> noise

	# compute prob of being a pattern (ignoring priors); this is just
	# P(pat) / (P(pat) + P(noise)). For numerical stability, we
	# compute this by taking the difference in log probs of the
	# numerator and denominator and then exponentiating that
	logDenominators = logsumexp((logProbs_noise, logProbs_pat))
	logProbsPat = logProbs_pat - logDenominators
	probsPat = np.exp(logProbsPat)
	# probsPat[np.isnan(probsPat)] = 0
	probsPat[ignoreIdxs] = 0

	print "noise log probs stats", describe(logProbs_noise)
	print "pat log probs stats", describe(logProbs_pat)
	print "probs stats,", describe(probsPat)

	# okay, so something here is broken but I have no idea what

	return probsPat

from scipy.stats import norm

def windowScores3(seq, length):
	diffs = seq[1:] - seq[:-1]
	absDiffs = np.abs(diffs)
	diffs_2 = diffs[1:] - diffs[:-1]
	absDiffs_2 = np.abs(diffs_2)

	# variance_diff = np.var(diffs)
	# expectedDiff = np.mean(absDiffs)
	expectedDiff = np.std(diffs)
	expectedDiffs_2 = absDiffs[:-1] + expectedDiff

	scores = np.zeros(seq.shape)
	actualDiffs = diffs[:-1]
	actualDiffs_2 = absDiffs_2

	# want (actual diff / expected diff) * (expected diff_2 / actual diff_2)
	firstProbs = norm.pdf(actualDiffs / expectedDiff)
	secondProbs = norm.pdf((actualDiffs_2 - expectedDiffs_2) / expectedDiff)
	# scores[1:-1] = (actualDiffs / expectedDiff) * (expectedDiffs_2 / actualDiffs_2)
	scores[1:-1] = firstProbs * secondProbs
	# scores[1:-1] = firstProbs
	# scores[1:-1] = secondProbs

	print describe(scores, 'scores')

	return scores

# Notes on this one:
# -finds big, smooth sections (even if sinusoidal or otherwise not just 1 slope)
#	-actually finds stuff in chlorineConcentration and some other ones
#	-owns at finding spikes in ecg stuff
# -fails to completely reject a white noise signal
# -ignores flat sections of actual patterns (even if right between steep sections)
# -only looks at how similar successive pairs of diffs are--ignores overall
# trends; consequently, stupid about finding gradual but consistent slopes
def windowScores4(seq, length):
	# diffs = seq[1:] - seq[:-1]
	# absDiffs = np.abs(diffs)
	# diffs_2 = diffs[1:] - diffs[:-1]
	# absDiffs_2 = np.abs(diffs_2)

	# var_diff = np.variance(diffs)
	# diffs_sq = diffs * diffs
	# diffs_sq_signed = diffs_sq * np.sign(diffs)

	# scores = np.zeros(seq.shape)

	# avgs of adjacent pairs of first derivatives
	diffs = seq[1:] - seq[:-1]
	avgs = (diffs[1:] + diffs[:-1]) / 2
	avgs = np.abs(avgs)
	# avgs = (seq[2:] - seq[:-2]) / 2 # = (diffs[1:] - diffs[:-1]) / 2

	# sigma = length * np.mean(np.abs(diffs), axis=0)
	# sigma = np.mean(np.abs(diffs), axis=0)
	# avgs /= sigma

	cumAvgs = np.cumsum(avgs, axis=0)
	windowAvgs = cumAvgs[length:] - cumAvgs[:-length]
	# cumDiffs = np.cumsum(np.abs(diffs[:-1]))
	# windowDiffs = cumDiffs[length:] - cumDiffs[:-length]

	# return 1. - np.exp(-windowAvgs**2)
	# absDiffs = np.abs(diffs)
	# sigma = np.std(diffs) * np.sqrt(2)
	# expectedTotalDiff = length * np.mean(absDiffs)
	# stdDiff = np.std(absDiffs)
	# return np.exp(-(windowAvgs/expectedTotalDiff)**2)
	# return np.exp(-(windowAvgs/sigma)**2)

	sigma = length * np.mean(np.abs(diffs), axis=0) * np.sqrt(2)
	# sigma =
	return 1. - np.exp(-(windowAvgs / sigma)**2)

	# return windowAvgs / np.max(windowAvgs) # good peaks, but needs to saturate
	# return windowAvgs / windowDiffs # always around .8

	# expectedDiff = np.mean(np.abs(diffs))
	# probs = 1. - np.exp(-avgs / expectedDiff)

	# print describe(probs, "probs stats")

	# return probs

	# cumProbs = np.cumsum(probs)
	# return cumProbs[length:] - cumProbs[:-length]


def windowScores5(seq, length, square=True, useVariance=False):
	cumDiffs, cumAbsDiffs = cumDiffsAndAbsDiffs(seq, square=True)

	windowDiffs = cumDiffs[length:] - cumDiffs[:-length]
	windowAbsDiffs = cumAbsDiffs[length:] - cumAbsDiffs[:-length]
	windowRatios = windowDiffs / windowAbsDiffs
	windowRatios = np.nan_to_num(windowRatios)
	windowScores = np.abs(windowRatios)

	if useVariance:
		windowStds = np.sqrt(slidingVariance(seq, length))
		windowWeights = windowStds / np.max(windowStds, axis=0)
		windowScores *= windowWeights[:-1]

	return windowScores


# _walks = {}
def createRandWalks(num, length, walkStd):
	walks = np.random.randn(num, length) * walkStd
	np.cumsum(walks, axis=1, out=walks)
	walks -= np.mean(walks, axis=1, keepdims=True)

	return walks

	# key = num * length
	# if key in _walks:
	# 	walks, oldStd = _walks[key]
	# 	walks *= (walkStd / oldStd) # scale to correct std deviation
	# else:
	# 	walks = np.random.randn(num, length) * walkStd
	# 	np.cumsum(walks, axis=1, out=walks)
	# 	walks -= np.mean(walks, axis=1, keepdims=True)

	# _walks[key] = (walks, walkStd) # memoize this

	# return walks


def windowScoresRandWalk(seq, length, numRandWalks=100, std=-1):
	numSubseqs = len(seq) - length + 1

	if length < 4:
		# print("WARNING: windowScoresRandWalk(): returning zero since "
		# 	"requested length {} < 4".format(length))
		if length <= 0: # n - m + 1 is > n in this case, which is bad
			numSubseqs = len(seq)
		return np.zeros(numSubseqs)

	if std <= 0:
		diffs = seq[1:] - seq[:-1]
		std = np.std(diffs)

	walks = createRandWalks(numRandWalks, length, std)

	windowScores = np.zeros(numSubseqs)
	subseqs = window.sliding_window_1D(seq, length)
	# for i in range(numSubseqs):
	# 	subseq = seq[i:i+length]
	for i, subseq in enumerate(subseqs):
		diffs = walks - (subseq - np.mean(subseq)) # combine mean norm with temp copying
		dists = np.sum(diffs * diffs, axis=1) / length
		# windowScores[i] = np.mean(dists)
		windowScores[i] = np.min(dists)
		# distScore = np.min(dists)
		# windowScores[i] = 1. - np.exp(distScore)

	windowScores /= np.max(windowScores)
	return windowScores

def slidingNotNoiseScoreAtLength(seq, length, windowScoreFunc=windowScoresRandWalk,
	zeroPad=False):
	windowScores = windowScoreFunc(seq, length)

	if len(windowScores) == len(seq):
		print "returning window scores directly"
		return windowScores

	if zeroPad:
		return ar.centerInMatOfSize(windowScores, len(seq))

	cumWindowScores = np.cumsum(windowScores, axis=0)
	scores = np.zeros(seq.shape)
	numWindows = len(windowScores)
	for i in range(len(seq)):
		firstWindowIncludingThis = max(0, i - length + 1)
		firstWindowIncludingThis = min(numWindows - 1, firstWindowIncludingThis)
		lastWindowIncludingThis = min(numWindows - 1, i)
		startScore = cumWindowScores[firstWindowIncludingThis]
		endScore = cumWindowScores[lastWindowIncludingThis]
		scores[i] = endScore - startScore

	return scores / length


def mainHighlight():
	howMany = 2
	np.random.seed(12345)
	# saveDir = 'figs/highlight/tidigits/'
	# tsList = datasets.loadDataset('tidigits_grouped_mfcc', whichExamples=range(howMany))
	tsList = datasets.loadDataset('ucr_short', whichExamples=range(1), instancesPerTs=5)
	# tsList = datasets.loadDataset('dishwasher_groups', whichExamples=range(howMany), instancesPerTs=5)
	# tsList = datasets.loadDataset('msrc', whichExamples=range(howMany), instancesPerTs=5)
	print '------------------------'
	for ts in tsList:
		ts.data = ts.data[:, 0]
		# ts.data = ts.data[:, 1]

		# ts.data = ts.data[:500]
		# _, axes = plt.subplots(5, figsize=(8, 10))
		# ts.plot(ax=axes[0])
		# # first deriv
		# from ..viz import viz_utils as viz
		# ts.data = ts.data[1:] - ts.data[:-1]
		# ts.plot(ax=axes[1])
		# axes[1].set_title('1st deriv')
		# axes[2].hist(ts.data, 50, normed=True) # looks a lot like a laplace distro
		# axes[2].set_title('1st deriv histogram')
		# viz.plotVertLine(np.median(np.abs(ts.data)), ax=axes[2], color='g')
		# viz.plotVertLine(np.mean(np.abs(ts.data)), ax=axes[2], color='k')
		# # second deriv
		# ts.data = ts.data[1:] - ts.data[:-1]
		# ts.plot(ax=axes[3])
		# axes[3].set_title('2nd deriv')
		# axes[4].hist(ts.data, 50, normed=True) # looks a lot like a laplace distro
		# axes[4].set_title('2nd deriv histogram')
		# viz.plotVertLine(np.median(np.abs(ts.data)), ax=axes[4], color='g')
		# viz.plotVertLine(np.mean(np.abs(ts.data)), ax=axes[4], color='k')
		# plt.tight_layout()
		# continue

		# ts.data = np.random.randn(len(ts.data)) # white noise
		# ts.data = np.cumsum(np.random.randn(len(ts.data))) # random walk
		# scores = slidingNotNoiseScoreAtLength(ts.data, len(ts.data) // 64)
		# scores = slidingNotNoiseScoreAtLength(ts.data, len(ts.data) // 32)
		scores = slidingNotNoiseScoreAtLength(ts.data, len(ts.data) // 16,
			zeroPad=True)
		# scores = slidingNotNoiseScoreAtLength(ts.data, 8)
		# scores = slidingNotNoiseScoreAtLength(ts.data, 16)
		# scores = slidingNotNoiseScoreAtLength(ts.data, 32)
		# return
		_, axes = plt.subplots(3)
		ts.plot(ax=axes[0])
		axes[1].plot(scores)
		axes[1].set_title("Scores")
		if len(scores.shape) > 1:
			means = np.mean(scores, axis=1)
			axes[2].plot(means)
		else:
			axes[2].plot(scores * scores)
			axes[2].set_title("Squared Scores")
		for ax in axes[1:]:
			ax.set_ylim([0, 1])
			ax.set_xlim([0, len(scores)])

		# ts.plot(saveDir)
		# print ts.name, ts.labels
		plt.tight_layout()
	plt.show()
	return

# ================================================================ PCA


def makeSine():
	return np.sin(np.linspace(0, 6, 100))


def pcaPlot(seq):
	_, axes = plt.subplots(2)
	ax0, ax1 = axes
	ax0.plot(seq)
	ax0.set_title("Time Series")

	vect = TruncatedSVD(n_components=1).fit_transform(seq)
	ax1.plot(vect)
	ax1.set_title("1st Principle Component")

	ylimits = [np.min(seq), np.max(seq)]
	[ax.autoscale(tight=True) for ax in axes]
	[ax.set_ylim(ylimits) for ax in axes]
	plt.tight_layout()
	plt.show()


def sineWhiteNoise(noiseStd=.1): # basically ignores the small noise dim
	v1 = makeSine()
	v2 = synth.randconst(len(v1)) * noiseStd
	seq = np.vstack((v1, v2)).T
	pcaPlot(seq)


def sineWhiteNoiseOffset(noiseStd=.1, noiseMean=5.):
	v1 = makeSine()
	v2 = synth.randconst(len(v1)) * noiseStd + noiseMean
	seq = np.vstack((v1, v2)).T
	pcaPlot(seq)


def sines():
	X = []
	for i in range(5):
		v = makeSine()
		X.append(v)
		X.append(v + np.random.randn(*v.shape) / 2.)
	X = np.vstack(X).T # each col of x is 1 dimension
	pcaPlot(X)


def embeddedSines(noiseStd=.1):
	v1, m = synth.sinesMotif()
	v2 = synth.randconst(len(v1)) * noiseStd
	seq = np.vstack((v1, v2)).T
	pcaPlot(seq)


def multishapes():
	seq, m = synth.multiShapesMotif()
	pcaPlot(seq)


def cancellingSeq(): # illustrate how PCA gets owned by negative correlation
	(seq, start1, start2), m = synth.multiShapesMotif(returnStartIdxs=True)
	seq = np.c_[(seq[:, 0], seq[:, 2])] # just up and down triangles
	for start in (start1, start2):
		seq[start:start+m] -= np.mean(seq[start:start+m], axis=0)
	pcaPlot(seq)


def embeddedSinesRandWalk(walkStd=.05):
	v1, m = synth.sinesMotif()
	v2 = synth.randwalk(len(v1), std=walkStd)
	v2 -= np.mean(v2)
	seq = np.vstack((v1, v2)).T
	pcaPlot(seq)


def embeddedSinesInRandWalk(walkStd=.1, patternInBothDims=False):
	inst1 = makeSine()
	inst2 = makeSine()
	v1 = synth.randwalk(500, std=walkStd)
	v2 = synth.randwalk(len(v1), std=walkStd)
	v1 -= np.mean(v1)
	v2 -= np.mean(v2)

	if patternInBothDims:
		# always preserves sines cuz any linear combination of the
		# sines is still a sine
		v1, start1, start2 = synth.createMotif(v1, inst1, inst2,
			sameMean=True, returnStartIdxs=True)
		synth.embedSubseq(v2, inst1, start1, sameMean=True)
		synth.embedSubseq(v2, inst2, start2, sameMean=True)
	else:
		v1 = synth.createMotif(v1, inst1, inst2, sameMean=True)
	seq = np.vstack((v1, v2)).T
	pcaPlot(seq)

def embeddedTrianglesInRandWalk(walkStd=.05, patternInBothDims=True):
	inst1 = synth.bell(100)
	inst1 -= np.mean(inst1)
	inst2 = synth.bell(100)
	inst2 -= np.mean(inst2)
	v1 = synth.randwalk(500, std=walkStd)
	v2 = synth.randwalk(len(v1), std=walkStd)
	v1 -= np.mean(v1)
	v2 -= np.mean(v2)

	if patternInBothDims:
		# always preserves sines cuz any linear combination of the
		# sines is still a sine
		v1, start1, start2 = synth.createMotif(v1, inst1, inst2,
			sameMean=True, returnStartIdxs=True)
		inst1 = synth.funnel(100)
		inst1 -= np.mean(inst1)
		inst2 = synth.funnel(100)
		inst2 -= np.mean(inst2)
		synth.embedSubseq(v2, inst1, start1, sameMean=True)
		synth.embedSubseq(v2, inst2, start2, sameMean=True)
	else:
		v1 = synth.createMotif(v1, inst1, inst2, sameMean=True)
	seq = np.vstack((v1, v2)).T
	pcaPlot(seq)

def mainPCA():
	# sines()
	# sineWhiteNoise(noiseStd=.1) # basically ignores noise
	# sineWhiteNoise(noiseStd=.5) # noisier sine wave of same noiseStd
	# sineWhiteNoise(noiseStd=1.) # basically just the noise
	# sineWhiteNoiseOffset() # doesn't subtract mean; dominated by high noise offset
	multishapes() # just sine wave; + and - triangles cancel; good example of PCA breaking
	embeddedSines() # noise dim goes away
	# cancellingSeq() # doesn't actually cancel, nevermind
	# embeddedSinesRandWalk() # either listens to sines or walk, depending on amplitudes
	# embeddedSinesInRandWalk() # same as above
	embeddedTrianglesInRandWalk() # often partial (but never exact) cancellation of triangles


# so basically, PCA is figuring out a linear combo of dims and then just adding
# the dims up with these weights. It listens to stuff with larger amplitude
# (as defined by distance from 0--no mean normalization) vastly more
	# amplitude has double effect since high-amplitude dims:
	# 1) add more to signal for a given weight
	# 2) get assigned higher weights by PCA
#
# Aside from irrelevant dimensions having higher amplitude, the other
# pathological case is when two dims are correlated overall, but pattern
# data in those dims are negatively correlated, and so they cancel out
#  -or vice versa

if __name__ == '__main__':
	mainHighlight()
	# mainPCA()






