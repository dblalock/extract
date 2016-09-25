#!/usr/env/python

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.decomposition import TruncatedSVD

from ..algo import motif
from ..algo import outcast
from ..algo.ff8 import logSpacedLengths
from ..datasets import synthetic as synth
from ..datasets.datasets import DataLoader
from ..utils import arrays as ar
from ..utils import learn
from ..utils import evaluate
from ..utils.files import ensureDirExists
from ..utils.misc import nowAsString
from ..viz import viz_utils as viz
from ..viz.viz_utils import plotRect


def labeledTimeSeriesToPatternInstances(ts):
	instances = []
	for i in range(len(ts.startIdxs)):
		start, end, lbl = ts.startIdxs[i], ts.endIdxs[i], ts.labels[i]
		instance = evaluate.createPatternInstance(start, end, lbl, fromSeq=ts.id)
		instances.append(instance)
	return instances


def labeledTimeSeriesCollectionToPatternInstances(collection):
	instances = []
	for i, ts in enumerate(collection):
		instances += labeledTimeSeriesToPatternInstances(ts)
	return instances


# ================================ Downsampler

def downsampleLabeledTs(ts, by):
	ts2 = ts.clone()
	ts2.data = ar.downsampleMat(ts.data, rowsBy=by)
	ts2.startIdxs /= by
	ts2.endIdxs /= by
	return ts2


class Downsampler(BaseEstimator, TransformerMixin):

	def __init__(self, by=5):
		self.by = by

	def fit(self, X, y=None, **params):
		return self

	def transform(self, X):
		"""assumes X is a collection of LabeledTimeSeries"""

		if self.by:
			Xt = map(lambda ts: downsampleLabeledTs(ts, self.by), X)
		return Xt


# ================================ DimsSelector

class DimsSelector(BaseEstimator, TransformerMixin):

	def __init__(self, whichDims=None, randomSubsetSize=None):
		self.whichDims = whichDims
		self.randomSubsetSize = randomSubsetSize

	def fit(self, X, y=None, **params):
		return self

	def transform(self, X):
		"""assumes X is a collection of LabeledTimeSeries"""
		if (not self.whichDims) and (not self.randomSubsetSize):
			return X

		Xt = []
		for ts in X:
			if len(ts.data.shape) < 2: # no dim selection when 1D
				continue

			ts2 = ts.clone()
			if self.whichDims is not None: # try using explicit set of dims
				print "DimsSelector: selecting {}/{} dims".format(len(self.whichDims),
					ts.data.shape[1])
				ts2.data = ts2.data[:, self.whichDims]
			elif self.randomSubsetSize: # pick a random subset of dims
				nDims = ts.data.shape[1]
				subsetSize = self.randomSubsetSize
				if self.randomSubsetSize < 1.:
					subsetSize = int(round(nDims * self.randomSubsetSize))
				whichDims = np.random.choice(np.arange(nDims), subsetSize)
				print "DimsSelector: using random dims: ", whichDims
				ts2.data = ts2.data[:, whichDims]
			Xt.append(ts2)

		return Xt


# ================================ DerivativeDimsAppender

def addDerivativeDims(X, first=True, second=False):
	X = ar.ensure2D(X)

	if (not first) and (not second):
		return X

	nSamples, nDims = X.shape
	newNumDims = nDims * 2
	if first and second:
		newNumDims = nDims * 3

	Xnew = np.empty((nSamples, newNumDims))
	Xnew[:2, nDims:] = 0 # zero out first 2 rows of derivatives

	if first:
		firstDiffs = np.diff(X, axis=0)
		Xnew[1:, nDims:] = firstDiffs
	if second:
		secondDiffs = np.diff(X, axis=0, n=2)
		if first:
			Xnew[2:, (2*nDims):] = secondDiffs
		else:
			Xnew[2:, nDims:] = secondDiffs

	return Xnew


class DerivativeDimsAppender(BaseEstimator, TransformerMixin):

	def __init__(self, first=True, second=False):
		self.first = first
		self.second = second

	def fit(self, X, y=None, **params):
		return self

	def transform(self, X):
		"""assumes X is a collection of LabeledTimeSeries"""

		Xt = []
		for ts in X:
			ts2 = ts.clone()
			ts2.data = addDerivativeDims(ts2.data, first=self.first, second=self.second)
			Xt.append(ts2)
		return Xt


# ================================ Adding noise + adversarial dims

class DimsAppender(BaseEstimator, TransformerMixin):

	def __init__(self, fractionAdversarialDims=0., numAdversarialDims=0,
		fractionNoiseDims=0., numNoiseDims=0, noiseType=None):

		self.fractionAdversarialDims = max(fractionAdversarialDims, 0)
		self.numAdversarialDims = int(numAdversarialDims)
		self.fractionNoiseDims = max(fractionNoiseDims, 0)
		self.numNoiseDims = int(numNoiseDims)
		self.noiseType = noiseType

	def fit(self, X, y=None, **params):
		return self

	def transform(self, X):
		"""assumes X is a collection of LabeledTimeSeries"""

		noNoiseDims = self.numNoiseDims <= 0 and self.fractionNoiseDims <= 0.
		noAdverDims = self.numAdversarialDims <= 0 and self.fractionAdversarialDims <= 0.
		if noNoiseDims and noAdverDims:
			return X

		numAdversarialDims = self.numAdversarialDims
		numNoiseDims = self.numNoiseDims
		Xt = []
		for ts in X:
			numDataDims = ts.data.shape[1]
			if numAdversarialDims < 1:
				numAdversarialDims = int(numDataDims * self.fractionAdversarialDims)
			if numNoiseDims < 1:
				numNoiseDims = int(numDataDims * self.fractionNoiseDims)

			ts2 = ts.clone()
			data = ts2.data
			if numAdversarialDims > 0:
				data = synth.addAdversarialDims(data, numToAdd=numAdversarialDims,
					startIdxs=ts.startIdxs, endIdxs=ts.endIdxs)
			if numNoiseDims > 0:
				data = synth.addNoiseDims(data, numToAdd=numNoiseDims,
					noiseType=self.noiseType)
			ts2.data = data

			Xt.append(ts2)
		return Xt

# class NoiseDimsAppender(DimsAppender): # TODO composition over inheritence here

# 	def __init__(self, fractionNoiseDims=0., numNoiseDims=0, noiseType=None):
# 		# super(NoiseDimsAppender, self).__init__(fractionNoiseDims, numNoiseDims)
# 		self.fractionNoiseDims = max(fractionNoiseDims, 0)
# 		self.numNoiseDims = int(numNoiseDims)
# 		self.noiseType = noiseType

# 	def transform(self, X):
# 		"""assumes X is a collection of LabeledTimeSeries"""

# 		if self.numNoiseDims == 0 and self.fractionNoiseDims == 0.:
# 			return X

# 		numToAdd = self.numNoiseDims
# 		Xt = []
# 		for ts in X:
# 			if numToAdd < 1:
# 				numToAdd = int(ts.data.shape[1] * self.fractionNoiseDims)

# 			ts2 = ts.clone()
# 			ts2.data = synth.addNoiseDims(ts2.data, numToAdd=numToAdd,
# 				noiseType=self.noiseType)
# 			Xt.append(ts2)
# 		return Xt


# class AdversarialDimsAppender(BaseEstimator, TransformerMixin):

# 	def __init__(self, fractionAdversarialDims=0., numAdversarialDims=0, noiseType=None):
# 		# super(NoiseDimsAppender, self).__init__(fractionAdversarialDims, numAdverarialDims)
# 		self.fractionAdversarialDims = max(fractionAdversarialDims, 0)
# 		self.numAdversarialDims = int(numAdversarialDims)

# 	def transform(self, X):
# 		"""assumes X is a collection of LabeledTimeSeries"""

# 		if self.numAdversarialDims == 0 and self.fractionAdversarialDims == 0.:
# 			return X

# 		numToAdd = self.numAdversarialDims
# 		Xt = []
# 		for ts in X:
# 			if numToAdd < 1:
# 				numToAdd = int(ts.data.shape[1] * self.fractionAdversarialDims)

# 			ts2 = ts.clone()
# 			ts2.data = synth.addAdversarialDims(ts2.data, numToAdd=numToAdd,
# 				startIdxs=ts.startIdxs, endIdxs=ts.endIdxs)
# 			Xt.append(ts2)
# 		return Xt

# class DimsAppender()

# ================================ MotifExtractor

def reduceTokDims(seq, numDims=1):
	if numDims < 1:
		return seq
	seq -= np.mean(seq, axis=0) # mean normalize cols
	return TruncatedSVD(n_components=numDims).fit_transform(seq)


def addLineBreakAtMidPoint(string):
	breakIdx = len(string)/2
	return string[:breakIdx] + '\n' + string[breakIdx:]


def plotMotifs(X, y, y_hat, showMotifs, saveMotifsDir, fileNameFunc, ax=None,
	plotSeq=True, capYLim=1000):
	"""X is a collection of labeledTimeSeries, y and y_hat are collections
	of PatternInstances. TODO this assumes same y and y_hat for all x in X
	(or, really, that there's only one x in X)"""
	if not (showMotifs or saveMotifsDir):
		return
	ensureDirExists(saveMotifsDir)
	origAx = ax
	for i, ts in enumerate(X):
		if origAx is None:
			plt.close()
			_, ax = plt.subplots(figsize=(10, 6))
		if plotSeq:
			ax.plot(ts.data, lw=2)
		ax.set_xlim((0, len(ts.data)))
		ax.set_ylim([ts.data.min(), min(capYLim, ts.data.max())])

		print "plotMotifs: plotting ts with name: {}".format(ts.name)

		fileName = fileNameFunc(i, ts)
		plotTitle = fileName.replace('_', ' ')
		plotTitle = plotTitle[:-4] # remove '.pdf'
		plotTitle = addLineBreakAtMidPoint(plotTitle)
		# breakIdx = len(plotTitle)/2
		# plotTitle = plotTitle[:breakIdx] + '\n' + plotTitle[breakIdx:]
		ax.set_title(plotTitle)

		for inst in y_hat[i]:
			plotRect(ax, inst.startIdx, inst.endIdx, hatch='///', alpha=.2)
		# if len(np.unique(y)) > 1:
		# 	for inst in y[i]:
		# 		try:
		# 			lbl = int(inst.label)
		# 		except:
		# 			lbl = hash(inst.label)
		# 		color = viz.colorForLabel(lbl)
		# 		plotRect(ax, inst.startIdx, inst.endIdx, color=color)
		# else:
		if True:
			for inst in y[i]:
				# plotRect(ax, inst.startIdx, inst.endIdx, color='green', hatch='///', alpha=.1)
				plotRect(ax, inst.startIdx, inst.endIdx, color='none', hatch='---', alpha=.3)
		if saveMotifsDir:
			plt.savefig(os.path.join(saveMotifsDir, fileName))
		if showMotifs:
			plt.show()


def computeIOUStats(y_hat, y, matchUpLabels):

	allIous = []
	iouTotals = np.zeros(4, dtype=np.float) # intersection-over-union

	# ------------------------ iou (overlap) stats
	for predicted, truth in zip(y_hat, y):
		# intersectionSize, unionSize, iou = evaluate.subseqIOUStats(
		iouStats = evaluate.subseqIOUStats(predicted, truth,
			matchUpLabels=matchUpLabels, returnMoreStats=True)
		intersectionSize, unionSize, iou, reportedSize, truthSize = iouStats
		iouTotals[0] += intersectionSize
		iouTotals[1] += unionSize
		iouTotals[2] += reportedSize
		iouTotals[3] += truthSize
		allIous.append(iou)

	iou = iouTotals[0] / iouTotals[1] if iouTotals[1] > 0. else 0.
	std_iou = np.std(allIous) if len(allIous) else 0.

	# if no ground truth starts/stops, don't divide by 0
	if not iouTotals[1]:
		iouTotals[1] += .001

	reportedLengths = [(0 if not insts else len(insts)) for insts in y_hat]
	numReported = sum(reportedLengths)
	# numTrue = sum([len(insts) for insts in y]) # all classes, so useless

	return {'iou': iou,
			'std_iou': std_iou,
			'numReported': numReported,
			# 'numTrue': numTrue,
			'intersectSz': iouTotals[0],
			'unionSz': iouTotals[1],
			'reportedSize': iouTotals[2],
			'truthSize': iouTotals[3],
			'reportedFrac': iouTotals[0] / iouTotals[2],
			'truthFrac': iouTotals[0] / iouTotals[3]}


def computePrecRecF1Stats(y_hat, y, ignorePositions, matchUpLabels,
	requireContainment, onlyReturnPair, iouThresholds=None):

	if iouThresholds is None or not len(iouThresholds):
		iouThresholds = np.arange(.05, 1., .05) # .05,.1,...,.95

	numThreshs = len(iouThresholds)

	# compute (precision, recall, f1 score) at a given IOU threshold
	if numThreshs == 1:
		thresh = iouThresholds[0]
		totals = np.zeros(3, dtype=np.int)
		precRecF1s = []

		for predicted, truth in zip(y_hat, y):
			numReported, numTrue, numMatches = evaluate.subseqMatchStats(
				predicted, truth, ignorePositions=ignorePositions,
				matchUpLabels=matchUpLabels, minOverlapFraction=thresh,
				requireContainment=requireContainment)
			if onlyReturnPair:
				assert(numReported == 2) # TODO remove
				numTrue = 2 # pretend there were only 2 instances to find
			totals += np.array((numReported, numTrue, numMatches), dtype=np.int)

			precRecF1 = evaluate.precisionRecallF1(numReported, numTrue, numMatches)
			precRecF1s.append(precRecF1)

		precRecF1 = evaluate.precisionRecallF1(totals[0],
			totals[1], totals[2])

		# return dict of stat name (with thresh as suffix) -> stat value
		suffix = '_{}'.format(int(thresh * 100))
		keys = ['prec', 'rec', 'f1', 'numTrue', 'numMatches']
		keys = [k + suffix for k in keys]
		vals = list(precRecF1) + list(totals[1:])
		stats = dict(zip(keys, vals))

		# standard deviations of precision, recall, f1
		if len(y) > 1:
			matchStats = np.array(precRecF1s, dtype=np.float)
			stds = np.std(matchStats, axis=0)
			keys = ['std_prec', 'std_rec', 'std_f1']
			keys = [k + suffix for k in keys]
			stats.update(dict(zip(keys, stds)))

		return stats

	# if more than one threshold, recurse and store stats for each
	stats = {}
	for i, thresh in enumerate(iouThresholds):
		statsForThresh = computePrecRecF1Stats(y_hat, y, ignorePositions,
			matchUpLabels, requireContainment, onlyReturnPair, [thresh])
		stats.update(statsForThresh)

	return stats


class BaseMotifExtractor(BaseEstimator, TransformerMixin, ClassifierMixin):

	def __init__(self, showMotifs=False, saveMotifsDir=None,
		onlyReturnPair=False, ignorePositions=False, matchUpLabels=False,
		requireContainment=False):
		self.showMotifs = showMotifs
		self.saveMotifsDir = saveMotifsDir
		self.onlyReturnPair = onlyReturnPair
		self.ignorePositions = ignorePositions
		self.matchUpLabels = matchUpLabels
		self.requireContainment = requireContainment

	def fit(self, X, y=None, **sink):
		return self

	def predict(self, X, **sink):
		Xt, instances = self.transform(X)
		return instances

	def plotMotifs(self, X, y, y_hat):
		raise NotImplementedError("Must be overridden by a subclass!")

	def score(self, X, y, sample_weight=None):
		"""assumes X is a collection of LabeledTimeSeries; all other
		arguments are ignored"""
		# TODO this method has become bloated and terrible

		# ================================ compute (and plot) y_hat

		tstart = time.clock()
		X, y_hat = self.transform(X)
		tElapsed = time.clock() - tstart

		# y is a collection of sets of instances, one instance
		# from each LabeledTimeSeries in the input
		y = []
		for i, ts in enumerate(X):
			instances = labeledTimeSeriesToPatternInstances(ts)
			y.append(instances)

		self.plotMotifs(X, y, y_hat)

		# ================================ compute stats

		stats = {}
		iouStats = computeIOUStats(y_hat, y, matchUpLabels=self.matchUpLabels)
		stats.update(iouStats)
		matchStats = computePrecRecF1Stats(y_hat, y,
			ignorePositions=self.ignorePositions,
			matchUpLabels=self.matchUpLabels,
			requireContainment=self.requireContainment,
			onlyReturnPair=self.onlyReturnPair)
		stats.update(matchStats)

		# print stats
		sortedKeys = sorted(stats.keys())
		for k in sortedKeys:
			print('{}: {}'.format(k, stats[k]))

		# print info about how well the algo did
		print("score(): got {}/{} matches, {} f1 overall @.5".format(
			stats['numMatches_50'], stats['numTrue_50'], stats['f1_50']))
		print("score(): got {}/{} matches, {} f1 overall @.25".format(
			stats['numMatches_25'], stats['numTrue_25'], stats['f1_25']))
		print("score(): got {} iou overall".format(
			stats['intersectSz'] / stats['unionSz']))
		print("score(): elapsed time = {}s".format(tElapsed))

		# rename stuff so that it's all together in the csv
		prependStr = '_'
		newStats = {}
		for k, v in stats.items():
			newStats[prependStr + k] = v
		stats = newStats

		# store main stat (f1 score at IOU of .5) as score
		stats[learn.SCORE_NAME] = stats[prependStr + 'f1_50']

		# store runtime
		stats['time'] = tElapsed

		return stats


class MotifExtractor(BaseMotifExtractor):

	def __init__(self, lengths=None, threshAlgo=None, addData=None,
		addDataFractionOfLength=0., lengthNormFunc=None, lengthStep=0,
		onlyReturnPair=False, ignorePositions=False, matchUpLabels=True,
		downsampleBy=None, forceNumDims=None, mdlBits=8, mdlAbandonCriterion=None,
		mdlSearchEachTime=True, maxOverlapFraction=0., requireContainment=False,
		seedShiftStep=None, showMotifs=False, saveMotifsDir=None):
		self.lengths = lengths
		self.lengthStep = lengthStep
		self.threshAlgo = threshAlgo
		self.addData = addData
		self.addDataFractionOfLength = addDataFractionOfLength
		self.lengthNormFunc = lengthNormFunc
		self.onlyReturnPair = onlyReturnPair
		self.ignorePositions = ignorePositions
		self.matchUpLabels = matchUpLabels
		self.requireContainment = requireContainment
		self.downsampleBy = downsampleBy # having this here allows auto-rescaling lengths
		self.forceNumDims = int(forceNumDims) if forceNumDims else -1
		self.mdlBits = mdlBits
		self.mdlAbandonCriterion = mdlAbandonCriterion
		self.mdlSearchEachTime = mdlSearchEachTime
		self.maxOverlapFraction = maxOverlapFraction
		self.seedShiftStep = seedShiftStep

		self.showMotifs = showMotifs
		self.saveMotifsDir = saveMotifsDir

	def transform(self, X):
		"""
		Parameters
		----------
		X: a collection of LabeledTimeSeries

		Returns
		-------
		X: the original collection of LabeledTimeSeries
		instances: A collection of PatternInstance objects
		"""

		self.lengths = np.asarray(self.lengths)

		if self.downsampleBy:
			X = map(lambda ts: downsampleLabeledTs(ts, self.downsampleBy), X)
			if np.max(self.lengths) > 1.: # absolute, not fractional lengths
				self.lengths /= self.downsampleBy

		print "MotifExtractor(): ts[0] shape", X[0].data.shape
		# print "MotifExtractor(): ", X[0].startIdxs
		# print "MotifExtractor(): ", X[0].endIdxs
		# print "MotifExtractor(): ", X[0].labels

		if self.onlyReturnPair:
			func = motif.findMotifPatternInstances
		else:
			func = motif.findAllMotifPatternInstances

		instances = []
		for i, ts in enumerate(X):
			print("MotifExtractor(): transforming ts {} / {}".format(i+1, len(X)))

			# possibly reduce dimensionality of ts
			if self.forceNumDims > 0:
				ts.data = ar.ensure2D(reduceTokDims(ts.data))

			# convert fractional lengths into absolute lengths
			lengths = self.lengths
			if np.max(lengths) <= 1.:
				lengths = lengths * len(ts.data)
			lengths = np.array(lengths, dtype=np.int)
			lengths = np.unique(lengths) # might yield identical ints

			# just try one in every N lengths, not every last one
			minLen = np.min(lengths)
			maxLen = np.max(lengths)
			lengths = np.arange(minLen, maxLen + 1, self.lengthStep)

			print "MotifExtractor(): searching lengths for this ts: ", lengths

			# print "MotifExtractor(): mdl bits = ", self.mdlBits

			instancesInTs = func(ts.data, lengths,
					threshAlgo=self.threshAlgo, addData=self.addData,
					addDataFractionOfLength=self.addDataFractionOfLength,
					lengthNormFunc=self.lengthNormFunc, mdlBits=self.mdlBits,
					maxOverlapFraction=self.maxOverlapFraction,
					mdlAbandonCriterion=self.mdlAbandonCriterion,
					searchEachTime=self.mdlSearchEachTime,
					shiftStep=self.seedShiftStep)
			# instances.append(instancesInTs)
			# make sure that ts id is set right or match comparisons will fail
			instances.append(map(lambda inst: inst._replace(fromSeq=ts.id),
				instancesInTs))

		return X, instances

	def plotMotifs(self, X, y, y_hat):
		plotMotifs(X, y, y_hat, self.showMotifs, self.saveMotifsDir, self._generateFileName)

	def _generateFileName(self, i, ts):
		fName = ts.name
		fName += '_ds=%d' % self.downsampleBy
		if self.forceNumDims > 0:
			fName += '_dims=%d' % self.forceNumDims
		fName += '_onlyPair=%s' % str(self.onlyReturnPair)
		fName += '_Lmin=%g' % np.min(self.lengths)
		fName += '_Lmax=%g' % np.max(self.lengths)
		fName += '_thresh=%s' % str(self.threshAlgo)
		fName += '_addData=%s' % str(self.addData)
		fName += '_addFrac=%g' % self.addDataFractionOfLength
		fName += '_overlap=%g' % self.maxOverlapFraction
		if self.threshAlgo == 'mdl':
			fName += '_bits=%d' % self.mdlBits
		fName += '.pdf'

		return fName


# ================================ Outcast

class OutcastExtractor(BaseMotifExtractor):

	def __init__(self, Lmin=0, Lmax=0, downsampleBy=1, lengthStep=0,
		ignorePositions=False, matchUpLabels=True, requireContainment=False,
		showMotifs=False, saveMotifsDir=None):

		self.Lmin = Lmin
		self.Lmax = Lmax
		self.downsampleBy = downsampleBy

		# stuff for base class to compute scores
		self.ignorePositions = ignorePositions
		self.matchUpLabels = matchUpLabels
		self.requireContainment = requireContainment

		# stuff to plot answers
		self.showMotifs = showMotifs
		self.saveMotifsDir = saveMotifsDir

	def transform(self, X):
		self.lengths = logSpacedLengths(self.Lmin, self.Lmax, logStep=.25,
			round=False, ints=False)

		print "OutcastExtractor(): log spaced lengths for this ts: ", self.lengths

		if self.downsampleBy:
			X = map(lambda ts: downsampleLabeledTs(ts, self.downsampleBy), X)
			if np.max(self.lengths) > 1.: # absolute, not fractional lengths
				self.lengths /= self.downsampleBy

		instances = []
		for i, ts in enumerate(X):
			print("OutcastExtractor(): transforming ts {} / {}".format(i+1, len(X)))

			# convert fractional lengths into absolute lengths
			lengths = np.asarray(self.lengths)
			if np.max(lengths) <= 1.:
				lengths = lengths * len(ts.data)
			lengths = lengths.astype(np.int)
			lengths = np.unique(lengths) # might yield identical ints

			if self.lengthStep > 1: # enforce spacing between lengths for small ones
				diffs = lengths[1:] - lengths[:-1]
				okayIdxs = np.where(diffs >= self.lengthStep)
				lengths = lengths[okayIdxs]

			print "OutcastExtractor(): Lmin, Lmax for this ts: ", \
				self.Lmin * len(ts.data), self.Lmax * len(ts.data)
			print "OutcastExtractor(): searching lengths for this ts: ", lengths

			instancesInTs = outcast.findAllOutcastInstances(ts.data, lengths)
			instances.append(map(lambda inst: inst._replace(fromSeq=ts.id),
				instancesInTs))

		return X, instances

	def plotMotifs(self, X, y, y_hat):
		plotMotifs(X, y, y_hat, self.showMotifs, self.saveMotifsDir, self._generateFileName)

	def _generateFileName(self, i, ts):
		fName = ts.name
		fName += '_ds=%d' % self.downsampleBy
		fName += '_Lmin=%g' % np.min(self.lengths)
		fName += '_Lmax=%g' % np.max(self.lengths)
		fName += '.pdf'

		return fName


# ================================ FFs

from ..algo.ff8 import buildFeatureMat, plotSeqAndFeatures
# from ..algo.ff8 import learnFFfromSeq
from ..algo.ff10 import learnFFfromSeq

# this class is unused; functionality rolled into FFExtractor
class SparseBinarizer(BaseEstimator, TransformerMixin):

	def __init__(self, featureLmin=0, featureLmax=0, cardinality=8,
		includeLocalZnorm=False, includeLocalSlope=False):
		self.featureLmin = featureLmin
		self.featureLmax = featureLmax
		self.cardinality = cardinality
		self.includeLocalZnorm = includeLocalZnorm
		self.includeLocalSlope = includeLocalSlope

	def fit(self, X, y=None, **params):
		return self

	def transform(self, X):
		"""assumes X is a collection of LabeledTimeSeries"""
		Xt = []
		for ts in X:
			featureMat = buildFeatureMat(ts.data, self.featureLmin,
				self.featureLmax, cardinality=self.cardinality,
				includeLocalZnorm=self.includeLocalZnorm,
				includeLocalSlope=self.includeLocalSlope)
			ts2 = ts.clone()
			ts2.data = featureMat.T
			Xt.append(ts2)

		return Xt


class FFExtractor(BaseMotifExtractor):

	def __init__(self, Lmin=0, Lmax=0, Lfilt=0, cardinality=4, downsampleBy=0,
		includeLocalZnorm=False, includeLocalSlope=False, includeSaxHashes=False,
		includeMaxFilteredSlope=False, includeNeighbors=False,
		includeVariance=False, extendEnds=True, detrend=False, ignoreFlat=False,
		saxCardinality=4, saxWordLen=4,
		logX=False, logXblur=False,
		ignorePositions=False, matchUpLabels=True, requireContainment=False,
		saveMotifsDir=None, showMotifs=False):
		super(FFExtractor, self).__init__(
			ignorePositions=ignorePositions, matchUpLabels=matchUpLabels,
			requireContainment=requireContainment,
			saveMotifsDir=saveMotifsDir, showMotifs=showMotifs)

		self.Lmin = Lmin
		self.Lmax = Lmax
		self.Lfilt = Lfilt
		self.cardinality = cardinality
		self.downsampleBy = downsampleBy
		self.includeLocalZnorm = includeLocalZnorm
		self.includeLocalSlope = includeLocalSlope
		self.includeSaxHashes = includeSaxHashes
		self.includeMaxFilteredSlope = includeMaxFilteredSlope
		self.includeNeighbors = includeNeighbors
		self.includeVariance = includeVariance
		self.extendEnds = extendEnds
		self.detrend = detrend
		self.ignoreFlat = ignoreFlat
		self.saxCardinality = saxCardinality
		self.saxWordLen = saxWordLen
		self.logX = logX
		self.logXblur = logXblur

		self.filters = []
		self.featureMats = []
		# use an internal SparseBinarizer so Lmin, Lmax, cardinality
		# tied together and so that we can store what the seqs looked like
		# before the transformation (for plotting)
		# self.featureConstructor = SparseBinarizer(Lmin, Lmax, cardinality,
		# 	includeLocalZnorm=includeLocalZnorm,
		# 	includeLocalSlope=includeLocalSlope)

		# print "FF init(): featureConstructor: including znorm, slope?"
		# print self.featureConstructor.includeLocalZnorm
		# print self.featureConstructor.includeLocalSlope

	def transform(self, X):
		"""
		Parameters
		----------
		X: a collection of LabeledTimeSeries

		Returns
		-------
		X: the original collection of LabeledTimeSeries
		instances: A collection of PatternInstance objects
		"""

		if self.downsampleBy and self.downsampleBy != 1:
			X = map(lambda ts: downsampleLabeledTs(ts, self.downsampleBy), X)
			if self.Lmin > 1.: # absolute, not fractional lengths
				self.Lmin = self.Lmin // self.downsampleBy
			if self.Lmax > 1.:
				self.Lmax = self.Lmax // self.downsampleBy
			if self.Lfilt > 1.:
				self.Lfilt = self.Lfilt // self.downsampleBy

		print "FF transform(): X[0] data shape = ", X[0].data.shape

		# self.Xorig = X # hack to let us plot both raw ts and feature rep

		# featureConstructor = SparseBinarizer(self.Lmin, self.Lmax,
		# 	cardinality=self.cardinality, includeLocalZnorm=self.includeLocalZnorm,
		# 	includeLocalSlope=self.includeLocalSlope)

		# X = featureConstructor.transform(X)

		instanceLists = []
		self.filters = []
		self.featureMats = []
		self.featureMatsBlur = []
		for i, ts in enumerate(X):
			print("FFExtractor(): transforming ts {} / {}".format(i, len(X)))

			startIdxs, endIdxs, filt, featureMat, featureMatBlur = learnFFfromSeq(
				ts.data, self.Lmin, self.Lmax, self.Lfilt,
				cardinality=self.cardinality,
				includeLocalZnorm=self.includeLocalZnorm,
				includeLocalSlope=self.includeLocalSlope,
				includeSaxHashes=self.includeSaxHashes,
				includeMaxFilteredSlope=self.includeMaxFilteredSlope,
				includeNeighbors=self.includeNeighbors,
				includeVariance=self.includeVariance,
				extendEnds=self.extendEnds,
				detrend=self.detrend,
				ignoreFlat=self.ignoreFlat,
				saxCardinality=self.saxCardinality,
				saxWordLen=self.saxWordLen,
				logX=self.logX,
				logXblur=self.logXblur)

			# print "ff: startIdxs, endIdxs = "
			# print np.c_[startIdxs, endIdxs]

			self.filters.append(filt)
			self.featureMats.append(featureMat)
			self.featureMatsBlur.append(featureMatBlur)

			instancesInTs = []
			for start, end in zip(startIdxs, endIdxs):
				instancesInTs.append(evaluate.createPatternInstance(start, end,
					fromSeq=ts.id))

			instanceLists.append(instancesInTs)

		return X, instanceLists

	def plotMotifs(self, X, y, y_hat):
		if not any((self.showMotifs, self.saveMotifsDir)):
			return
		if self.saveMotifsDir:
			ensureDirExists(self.saveMotifsDir)

		# print "plotMotifs: yhat = ", y_hat

		# TODO factor out dup code with plotMotifs
		for i, ts in enumerate(X):
			# hack assuming we just transformed X
			featureMat = self.featureMats[i]
			# featureMat = self.featureMatsBlur[i]

			# plt.figure() # so rects getting plotted here seem to be past end...
			# filt = self.filters[i]
			# viz.imshowBetter(filt)
			# plt.show()
			# windowLen = filt.shape[1]

			axSeq, axSim, axFilt = plotSeqAndFeatures(ts.data, featureMat,
				createFiltAx=True, padBothSides=True)
			fileName = self._generateFileName(i, ts)
			title = fileName[:-4] # no .pdf
			title = title.replace('_', ' ')
			axSeq.set_title(title)
			for inst in y_hat[i]:
				plotRect(axSeq, inst.startIdx, inst.endIdx)
				# TODO this is a total hack to plot where instances end in
				# the feature matrix; this info should come directly from
				# the learning function
				matOffset = 0 # 0, not below line, since we told it pad both sides
				# matOffset = (len(ts.data) - featureMat.shape[1]) // 2
				matStartIdx = inst.startIdx - matOffset
				# matEndIdx = inst.endIdx - matOffset - self.Lfilt
				matEndIdx = inst.endIdx - matOffset
				plotRect(axSim, matStartIdx, matEndIdx, alpha=.2)
			for inst in y[i]:
				plotRect(axSeq, inst.startIdx, inst.endIdx, color='none', hatch='---', alpha=.3)
				# try:
				# 	lbl = int(inst.label)
				# except:
				# 	lbl = hash(inst.label)
				# color = viz.colorForLabel(lbl + 1) # hack so lbl 0 -> green
				# plotRect(axSeq, inst.startIdx, inst.endIdx, color=color)

			filt = self.filters[i]
			if filt is not None:
				axFilt.imshow(filt, interpolation='nearest', aspect='auto')
			else:
				print("WARNING: attempted to plot null filter for ts {}".format(i))
			# from ..algo import ff10
			# filtLen = len(ts.data) * self.Lmin
			# filtBlur = ff10.filterRows(filt, filtLen)
			# axFilt.imshow(filtBlur, interpolation='nearest', aspect='auto')

			plt.tight_layout()
			if self.saveMotifsDir:
				plt.savefig(os.path.join(self.saveMotifsDir, fileName))
			if self.showMotifs:
				plt.show()

	def _generateFileName(self, i, ts):
		fName = ts.name
		fName += '_ds=%d' % self.downsampleBy
		fName += '_Lmin=%g' % self.Lmin
		fName += '_Lmax=%g' % self.Lmax
		fName += '_Lfilt=%g' % self.Lfilt
		fName += '.pdf'

		return fName


# ================================================================ Main

def main():
	# from ..datasets import synthetic as synth
	# (data, start1, start2), m = synth.multiShapesMotif(returnStartIdxs=True)
	# print data.shape

	# y = [evaluate.createPatternInstance(startIdx=idx, endIdx=idx + m)
	# 	for idx in (start1, start2)]

	# ------------------------ try out estimators on their own
	# data = [data]

	# dimsel = DimsSelector(whichDims=[0, 2])
	# ds = Downsampler(by=4)

	# data = dimsel.fit_transform(data)
	# data = ds.fit_transform(data)

	# # data = data[0]

	# print data.shape
	# import matplotlib.pyplot as plt
	# plt.plot(data)
	# plt.show()

	# ------------------------  try out pipeline
	# dimsel = DimsSelector(whichDims=[0, 2])
	# ds = Downsampler(by=4)
	# from sklearn.pipeline import Pipeline, make_pipeline
	# p = make_pipeline(dimsel, ds)

	# data = p.fit_transform(data)

	# print data.shape
	# import matplotlib.pyplot as plt
	# plt.plot(data)
	# plt.show()

	# ------------------------  try out learn functions

	# dimSelecting = ("DimSelect", DimsSelector(whichDims=[0,2]))
	# # downSampling = ("Downsample", Downsampler(by=1))
	# motifFinding = ("FindMotif", MotifExtractor(lengths=[6], onlyReturnPair=True, downsampleBy=4))

	# # d = [dimSelecting, downSampling, motifFinding]
	# d = [dimSelecting, motifFinding]
	# pipes, pipeParams = learn.makePipelines(d, cacheBlocks=False)

	# df = learn.tryParams(d, data, y, crossValidate=False)
	# print df
	# # print pipes, pipeParams

	# from ..utils.misc import nowAsString
	# from ..utils.files import ensureDirExists
	# import os
	# SAVE_DIR = './results/motif/'
	# ensureDirExists(SAVE_DIR)
	# df.to_csv(os.path.join(SAVE_DIR, nowAsString() + '.csv'))

	# ------------------------  try out moar learn functions

	saveDir = 'figs/motif/experiments/'
	ensureDirExists(saveDir)

	# datasetParams = [{'datasetName': ('triangles', 'sines')}]
	datasetParams = [{'datasetName': ['msrc'],
		# 'whichExamples':[[1,2]],
		'whichExamples':[range(3)],
		# 'seed': [np.random.rand(2)]
	}]
	dimSelectingParams = [{'whichDims': [None]}]
	# dimSelectingParams = [{'randomSubsetSize': [.0625, .25]}]
	# extractorParams = [{'lengths': [25],
	extractorParams = [{'lengths': [np.arange(1./20, 1./8, .005)],
	# extractorParams = [{'lengths': [20],
		# 'downsampleBy': [5, 10],
		'downsampleBy': [5],
		# 'onlyReturnPair': [True],
		'onlyReturnPair': [False],
		'threshAlgo': ['minnen'],
		'saveMotifsDir': [saveDir]
	}]
	# 'showMotifs': [True]}]

	dataLoading = [DataLoader(), datasetParams]
	dimSelecting = (DimsSelector(), dimSelectingParams)
	motifFinding = (MotifExtractor(), extractorParams)

	d = [("LoadData", [dataLoading]),
		("SelectDims", [dimSelecting]),
		("FindMotif", [motifFinding])
	]

	df = learn.tryParams(d, None, None, crossValidate=False)

	print df

	# from ..utils.misc import nowAsString
	# from ..utils.files import ensureDirExists
	# SAVE_DIR = './results/motif/'
	# ensureDirExists(SAVE_DIR)
	# df.to_csv(os.path.join(SAVE_DIR, nowAsString() + '.csv'))

if __name__ == '__main__':
	main()





