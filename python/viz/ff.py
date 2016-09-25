#!/usr/bin/env python

import matplotlib.pyplot as plt

from ..utils import arrays as ar
from viz_utils import plotRect, removeEveryOtherYTick

def plotSeqAndFeatures(seq, X, model, createModelAx=False, padBothSides=False, capYLim=1000):
	"""plots the time series above the associated feature matrix"""

	# mpl.rcParams['font.size'] = 30 # tick label size

	plt.figure(figsize=(10, 8))
	# plt.figure(figsize=(8, 10))
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
			print("WARNING: attempted to plot None as feature weights!")

	for ax in axes:
		for tick in ax.get_xticklabels() + ax.get_yticklabels():
			tick.set_fontsize(20)
	removeEveryOtherYTick(axSeq)

	return axes


def plotFFOutput(ts, startIdxs, endIdxs, featureMat, model,
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

	axSim.set_xlabel("Time")
	axSim.set_ylabel("Feature")
	plt.tight_layout()
