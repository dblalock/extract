#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import arrays as ar
from extract import extract

# NOTE: this file does not "stand alone" as far as loading datasets
from ..python.datasets import datasets

try: # slightly prettier plots if you have seaborn installed
	import seaborn as sb
	# don't draw white lines through the data
	sb.set_style("whitegrid", {'axes.grid': False})
except:
	pass


def plotVertLine(x, ymin=None, ymax=None, ax=None, **kwargs):
	if ax and (not ymin or not ymax):
		ymin, ymax = ax.get_ylim()
	if not ax:
		ax = plt

	kwargs.setdefault('color', 'k')
	kwargs.setdefault('linestyle', '--')
	if 'linewidth' not in kwargs:
		kwargs.setdefault('lw', 2)

	ax.plot([x, x], [ymin, ymax], **kwargs)


def plotRect(ax, xmin, xmax, ymin=None, ymax=None, alpha=.2,
	showBoundaries=True, color='grey', fill=True, hatch=None, **kwargs):
	if ax and (ymin is None or ymax is None):
		ymin, ymax = ax.get_ylim()
	if fill:
		patch = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
				facecolor=color, alpha=alpha, hatch=hatch)
		ax.add_patch(patch)
	if showBoundaries:
		plotVertLine(xmin, ymin, ymax, ax=ax, color=color, **kwargs)
		plotVertLine(xmax, ymin, ymax, ax=ax, color=color, **kwargs)


def removeEveryOtherXTick(ax):
	lbls = ax.get_xticks().astype(np.int).tolist()
	for i in range(1, len(lbls), 2):
		lbls[i] = ''
	ax.set_xticklabels(lbls)


def removeEveryOtherYTick(ax):
	lbls = ax.get_yticks().astype(np.int).tolist()
	for i in range(1, len(lbls), 2):
		lbls[i] = ''
	ax.set_yticklabels(lbls)


def plotSeqAndFeatures(seq, X, model, createModelAx=False, padBothSides=False, capYLim=1000):
	"""plots the time series above the associated feature matrix"""

	plt.figure(figsize=(10, 8))
	if createModelAx:
		nRows = 4
		nCols = 7
		axSeq = plt.subplot2grid((nRows,nCols), (0,0), colspan=(nCols-1))
		axSim = plt.subplot2grid((nRows,nCols), (1,0), colspan=(nCols-1), rowspan=(nRows-1))
		axPattern = plt.subplot2grid((nRows,nCols), (1,nCols-1), rowspan=(nRows-1))
		axes = (axSeq, axSim, axPattern)
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

	axSeq.set_title("Time Series", fontsize=28, y=1.04)
	axSim.set_title("Feature Matrix", fontsize=28, y=1.01)

	# plot the learned pattern model
	if createModelAx:
		axPattern.set_title("Learned\nPattern", fontsize=24, y=1.02)

		if model is not None:
			axPattern.imshow(model, interpolation='nearest', aspect='auto')
			tickLocations = plt.FixedLocator([0, model.shape[1]])
			axPattern.xaxis.set_major_locator(tickLocations)
			axPattern.yaxis.tick_right() # y ticks on right side
		else:
			print("WARNING: attempted to plot null feature weights!")

	for ax in axes:
		for tick in ax.get_xticklabels() + ax.get_yticklabels():
			tick.set_fontsize(20)
	removeEveryOtherYTick(axSeq)

	return axes


def plotExtractOutput(ts, startIdxs, endIdxs, featureMat, model,
	cleanOverlap=False, highlightInFeatureMat=False,
	showGroundTruth=False):

	axSeq, axSim, axPattern = plotSeqAndFeatures(ts.data, featureMat, model,
		createModelAx=True, padBothSides=True)

	# our algorithm is allowed to return overlapping regions, but
	# they look visually cluttered, so crudely un-overlap them
	if cleanOverlap:
		for i in range(1, len(startIdxs)):
			start, end = startIdxs[i], endIdxs[i-1]
			if start < end:
				middle = int((start + end) // 2)
				startIdxs[i] = middle + 1
				endIdxs[i-1] = middle

	# plot estimated regions
	for startIdx, endIdx in zip(startIdxs, endIdxs):
		plotRect(axSeq, startIdx, endIdx)

		if highlightInFeatureMat:
			# this is a hack to plot where instances are in the feature
			# matrix; ideally, this info should come directly from
			# the learning function
			matOffset = 0 # 0, not below line, since we told it to pad both sides
			matStartIdx = startIdx - matOffset
			matEndIdx = endIdx - matOffset
			plotRect(axSim, matStartIdx, matEndIdx, alpha=.2)

	# plot ground truth regions
	if showGroundTruth:
		for startIdx, endIdx in zip(ts.startIdxs, ts.endIdxs):
			plotRect(axSeq, startIdx, endIdx, color='none', hatch='---', alpha=.3)

	plt.tight_layout()


def main():
	np.random.seed(12345)

	# read in the first few time series from the TIDIGITS dataset; the return
	# value is a collection of LabeledTimeSeries (see datasets.utils). You
	# will of course need to have the relevant dataset on your machine, as
	# well as update datasets/paths.py to point to it. For TIDIGITS
	# specifically, you will also need to have librosa installed. For the
	# UCR datasets, the whichExamples argument takes this many examples from
	# all 20 datasets
	whichExamples = np.arange(2)
	tsList = datasets.loadDataset(datasets.TIDIGITS, whichExamples=whichExamples)

	# uncomment any of these to use a different dataset
	# tsList = datasets.loadDataset(datasets.DISHWASHER, whichExamples=whichExamples)
	# tsList = datasets.loadDataset(datasets.MSRC, whichExamples=whichExamples)
	# tsList = datasets.loadDataset(datasets.UCR, whichExamples=[0])

	Lmin, Lmax = 1./20, 1./10 # fractions of time series length
	for ts in tsList:
		startIdxs, endIdxs, model, featureMat, featureMatBlur = extract(
			ts.data, Lmin, Lmax)
		plotExtractOutput(ts, startIdxs, endIdxs, featureMat, model)

		# you can also call this if you just want to see what the data looks like
		# ts.plot()

		# plt.savefig(ts.name + '.pdf') # use this to save it

		plt.show()

if __name__ == '__main__':
	main()
