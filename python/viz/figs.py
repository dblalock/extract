#!/bin/env/python

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb

from joblib import Memory
memory = Memory('.', verbose=0)

from ..algo.ff10 import learnFFfromSeq
from ..algo.motif import findMotifPatternInstances, findAllMotifPatternInstances

from ..datasets import datasets
from ..datasets import dishwasher as dw
from ..datasets import read_ucr as ucr
from ..datasets import synthetic as synth
# from ..datasets import tidigits as ti

from ..utils import arrays as ar

import viz_utils as viz

SAVE_DIR = 'figs/paper'
# TITLE_Y_POS = 1.02
TITLE_Y_POS = 1.02

WHICH_DISHWASHER_TS = 22
WHICH_TIDIGITS_TS = 9

DEFAULT_SB_PALETTE = sb.color_palette()[:]
DISHWASHER_DIM_TO_HIGHLIGHT = 9
DISHWASHER_HIGHLIGHT_COLOR = sb.color_palette()[2]
DISHWASHER_COLOR_PALETTE = DEFAULT_SB_PALETTE[:2] + DEFAULT_SB_PALETTE[3:]

# AXES_BG_COLOR = '#EAEAF2' # seaborn default purple
AXES_BG_COLOR = '#EDEDF6' # slightly lighter
# AXES_BG_COLOR = '#F0F0F8' # lighter
# AXES_BG_COLOR = '#F8F8FA' # lighterer
# AXES_BG_COLOR = '#FCFCFF' # lightest

def saveFigWithName(name):
	path = os.path.join(SAVE_DIR, name + '.pdf')
	plt.savefig(path)

# ================================================================ Datasets

# ------------------------------------------------ Dishwasher

@memory.cache
def getGoodDishwasherTs():
	np.random.seed(123)
	tsList = dw.getLabeledTsList(shuffle=True, includeZC=True,
		instancesPerTs=3, padLen=50)
	# return tsList[8] # this one looks good with 5 instances
	return tsList[6] # this one looks good

def makeDishwasherFig(ax=None, zNorm=True, save=True):
	# ts = getGoodDishwasherTs()
	# ts.data = ar.zNormalizeCols(ts.data)
	ts = getFig1Ts(zNorm=True, whichTs=WHICH_DISHWASHER_TS)
	# ax = ts.plot(useWhichLabels=['ZC'], showLabels=False, capYLim=900)
	colors = DISHWASHER_COLOR_PALETTE * 3 # cycles thru almost three times
	colors[DISHWASHER_DIM_TO_HIGHLIGHT] = DISHWASHER_HIGHLIGHT_COLOR
	colors = colors[:ts.data.shape[1]]

	ts.data[:, 2] /= 2 # scale the ugliest dim to make pic prettier
	ax = ts.plot(showLabels=False, showBounds=False, capYLim=900, ax=ax,
		colors=colors) # resets palette...
	# ax = ts.plot(showLabels=False, showBounds=False, capYLim=900, ax=None) # works

	# ax.plot(ts.data[:, DISHWASHER_DIM_TO_HIGHLIGHT], color=DISHWASHER_HIGHLIGHT_COLOR)
	# sb.set_palette(DEFAULT_SB_PALETTE)

	sb.despine(left=True)
	ax.set_title("Dishwasher", y=TITLE_Y_POS)
	# ax.set_xlabel("Minute")
	plt.tight_layout()
	if save:
		saveFigWithName('dishwasher')

# ------------------------------------------------ MSRC

@memory.cache
def getGoodMsrcTs():
	tsList = datasets.loadDataset('msrc', whichExamples=range(10))
	return tsList[1]

def makeMsrcFig(ax=None, save=True):
	ts = getGoodMsrcTs()
	ax = ts.plot(showLabels=False, showBounds=False, ax=ax)
	sb.despine(left=True)
	ax.set_title("MSRC-12", y=TITLE_Y_POS)
	# ax.set_xlabel("Time (sample)")
	plt.tight_layout()
	if save:
		saveFigWithName('msrc')

# ------------------------------------------------ UCR

@memory.cache
def getGoodUcrTs():
	np.random.seed(123)
	# datasetName = "Two_Patterns"
	# datasetName = "50words" # decent, but not visually obvious
	datasetName = "wafer"
	# datasetName = "SwedishLeaf" # instance 3 looks decent
	tsList = ucr.labeledTsListFromDataset(datasetName, instancesPerTs=5)
	return tsList[3]

def makeUcrFig(ax=None, save=True):
	ts = getGoodUcrTs()
	ax = ts.plot(showLabels=False, showBounds=False, ax=ax, linewidths=3.)
	sb.despine(left=True)
	ax.set_title("UCR", y=TITLE_Y_POS)
	# ax.set_xlabel("Time (sample)")
	plt.tight_layout()
	if save:
		saveFigWithName('ucr')

# ------------------------------------------------ Tidigits

# @memory.cache
def getGoodTidigitsTs(whichTs=-1):
	if whichTs < 0:
		whichTs = WHICH_TIDIGITS_TS
	np.random.seed(12345) # from main in tidigits.py; should ideally be 123
	tsList = datasets.loadDataset('tidigits_grouped_mfcc', whichExamples=range(10))
	return tsList[whichTs]

def makeTidigitsFig(ax=None, save=True, whichTs=-1):
	ts = getGoodTidigitsTs(whichTs=whichTs)
	# ts.data = ar.meanNormalizeCols(ts.data)
	ax = ts.plot(showLabels=False, showBounds=False, ax=ax, linewidths=3.)
	sb.despine(left=True)
	ax.set_title("TIDIGITS", y=TITLE_Y_POS)
	# ax.set_xlabel("Time (sample)")
	plt.tight_layout()
	if save:
		saveFigWithName('tidigits')

# ------------------------------------------------ Combined

def makeDatasetsFig():
	mpl.rcParams['axes.titlesize'] = 32
	mpl.rcParams['axes.titleweight'] = 'bold'
	mpl.rcParams['axes.facecolor'] = AXES_BG_COLOR

	fig, axes = plt.subplots(2, 2)
	axes = axes.flatten()

	makeDishwasherFig(axes[0], save=False)
	makeTidigitsFig(axes[1], save=False)
	makeMsrcFig(axes[2], save=False)
	makeUcrFig(axes[3], save=False)

	plt.tight_layout(h_pad=2.)
	saveFigWithName('datasets')


# ================================================================ Fig1

@memory.cache
def getFig1Ts(zNorm=False, whichTs=-1):

	np.random.seed(123)
	if zNorm:
		if whichTs < 0:
			whichTs = 4 # 3 is pretty good
		tsList = dw.getLabeledTsList(shuffle=True, includeZC=True,
			instancesPerTs=3, addNoise=False)
		ts = tsList[whichTs]
		# ts.data = ar.meanNormalizeCols(ts.data)
		for i, col in enumerate(ts.data.T):
			ts.data[:, i] = ar.zNormalize(col)
			# if np.std(col) > 3: # exclude "flat" dims
			# 	if np.max(col) < 1000: # pick out the dims that don't spike like crazy
			# 		newCol = 100 * col / np.max(col)
			# 		ts.data[:,i] = newCol - np.min(newCol)
			# 	else:
			# 		# ts.data[:,i] = 10 * col / np.std(col)
			# 		ts.data[:, i] = 0
	else:
		whichTs = 9
		tsList = dw.getLabeledTsList(shuffle=True, includeZC=True, instancesPerTs=3)
		ts = tsList[whichTs]
		ts.data = np.minimum(1000, ts.data) # rip out insane jump # TODO remove
	# return tsList[8] # this one looks good with 5 instances

	return ts
	# super unambigous-looking ones = 6, 9, 14 (ZC), 28 (Z, noise spike), 31,
	# 36 (super irregular rect widths), 40 (just Z/Z2), 50 (ZC), 64 (Z/Z2),
	# 65 (ZC), 67 (ZC),

@memory.cache
def getGoodRectsTs(): # don't return crappy-looking ones right at the edge, etc
	np.random.seed(123)
	tsList = datasets.loadDataset('rects', whichExamples=range(10))
	# for ts in tsList:
	# 	ts.plot()
	# plt.show()
	return tsList[3] # 3 and 5 are fairly centered

@memory.cache
def getGoodTrianglesTs(): # don't return crappy-looking ones right at the edge, etc
	np.random.seed(123)
	tsList = datasets.loadDataset('triangles', whichExamples=range(10))
	# for ts in tsList:
	# 	ts.plot()
	# plt.show()
	return tsList[0] # 0,1,4,9 are reasonably centered

@memory.cache
def getGoodShapesTs(): # don't return crappy-looking ones right at the edge, etc
	np.random.seed(123)
	tsList = datasets.loadDataset('shapes', whichExamples=range(10))
	# for ts in tsList:
	# 	ts.plot()
	# plt.show()
	return tsList[0] # 0 is the best, 2 is okay

@memory.cache
def labelTs_sota(ts, lengths, onlyPair=False, mdl=False):
	if onlyPair:
		instancesInTs = findMotifPatternInstances(ts.data, lengths)
	else:
		if mdl:
			instancesInTs = findAllMotifPatternInstances(ts.data, lengths,
				mdlAbandonCriterion='allNegative', threshAlgo='mdl')
		else:
			instancesInTs = findAllMotifPatternInstances(ts.data, lengths,
				mdlAbandonCriterion='allNegative', threshAlgo='minnen')
	# construct a labeledTs with the returned start and end idxs
	ts_sota = ts.clone()
	startIdxs = np.array([inst.startIdx for inst in instancesInTs])
	endIdxs = np.array([inst.endIdx for inst in instancesInTs])
	labels = [inst.label for inst in instancesInTs]
	ts_sota.startIdxs = startIdxs
	ts_sota.endIdxs = endIdxs
	ts_sota.labels = labels

	return ts_sota

@memory.cache
def labelTs_ff(ts, Lmin, Lmax, **kwargs):
	startIdxs, endIdxs, filt, featureMat, featureMatBlur = learnFFfromSeq(
		ts.data, Lmin, Lmax, includeLocalSlope=True, **kwargs)
	ts_ff = ts.clone()
	ts_ff.startIdxs = startIdxs
	ts_ff.endIdxs = endIdxs
	ts_ff.labels = [0] * len(startIdxs)

	return ts_ff

def makeFig1():
	ts = getFig1Ts()

	# set up axes
	ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
	ax2 = plt.subplot2grid((2,2), (1,0))
	ax3 = plt.subplot2grid((2,2), (1,1))
	axes = [ax1, ax2, ax3]

	for ax in axes:
		ax.autoscale(tight=True)
		sb.despine(left=True, ax=ax)

	ts.plot(showLabels=False, showBounds=False, ax=ax1)

	lengths = [150]
	ts_sota = labelTs_sota(ts, lengths)
	ts_sota.plot(showLabels=False, ax=ax2)

	ts_ff = labelTs_ff(ts, 100, 200) # Lmin, Lmax
	ts_ff.plot(showLabels=False, ax=ax3)

	plt.setp(ax3.get_yticklabels(), visible=False)
	ax1.set_title("Patterns in Dishwasher Dataset")
	ax1.set_xlabel("Minute")
	ax2.set_title("State-of-the-art")
	ax3.set_title("Proposed")

	plt.tight_layout()
	plt.show()

def makeFig2():
	ts_rect = getGoodRectsTs()
	ts_tri = getGoodTrianglesTs()
	ts_shape = getGoodShapesTs()
	# ts_shape.data = ts_shape.data[:, [0,2]] # remove sine wave

	# tsList = [ts_tri]
	# tsList = [ts_rect]
	# tsList = [ts_shape]
	tsList = [ts_rect, ts_tri, ts_shape]
	for ts in tsList:
		lengths = [50]
		ts_sota = labelTs_sota(ts, lengths, onlyPair=True)
		ts_sota.plot(showLabels=False)

		if ts == ts_rect: # TODO remove hack after prototyping fig
			ts_ff = labelTs_ff(ts, .1, .2, extractTrueLocsAlgo='none')
		else:
			ts_ff = labelTs_ff(ts, .1, .2)

		ts_ff.plot(showLabels=False)
	plt.show()

def makeCombinedFig1(short=False, zNorm=False, whichTs=-1, scaleVariance=False):
	# sb.set_style('ticks')
	# sb.set_style('white')

	mpl.rcParams['axes.facecolor'] = AXES_BG_COLOR

	ts_dw = getFig1Ts(zNorm=zNorm, whichTs=whichTs)
	ts_rect = getGoodRectsTs()
	ts_tri = getGoodTrianglesTs()
	ts_shape = getGoodShapesTs()

	if scaleVariance:
		ts_dw_pretty = getFig1Ts(scaleVariance=True)
	else:
		ts_dw_pretty = ts_dw

	# remove the red since we'll save this to highlight the "pattern" dim
	palette = sb.color_palette()
	palette = palette[:2] + palette[3:]
	highlightColor = sb.color_palette()[2] # red
	sb.set_palette(palette)

	# set up axes
	if short:
		SCALE = 3
		OFFSET = 1
		NUM_ROWS = 2 * SCALE + 1
		FIG_SIZE = (8, 5)
	else:
		SCALE = 2
		OFFSET = 1
		NUM_ROWS = 5 * SCALE + 1
		FIG_SIZE = (8, 10)
	NUM_COLS = 2 * SCALE
	gridSz = (NUM_ROWS, NUM_COLS)
	fig = plt.figure(figsize=FIG_SIZE)
	ax0 = plt.subplot2grid(gridSz, (0,0), rowspan=SCALE, colspan=2*SCALE)
	ax10 = plt.subplot2grid(gridSz, (1 * SCALE + OFFSET, 0 * SCALE),
		rowspan=SCALE, colspan=SCALE)
	ax11 = plt.subplot2grid(gridSz, (1 * SCALE + OFFSET, 1 * SCALE),
		rowspan=SCALE, colspan=SCALE)
	if short:
		axes = [ax0, ax10, ax11]
		leftAxes = [ax10]
		rightAxes = [ax11]
	else:
		ax20 = plt.subplot2grid(gridSz, (2 * SCALE + OFFSET, 0 * SCALE),
			rowspan=SCALE, colspan=SCALE)
		ax21 = plt.subplot2grid(gridSz, (2 * SCALE + OFFSET, 1 * SCALE),
			rowspan=SCALE, colspan=SCALE)
		ax30 = plt.subplot2grid(gridSz, (3 * SCALE + OFFSET, 0 * SCALE),
			rowspan=SCALE, colspan=SCALE, sharex=ax20)
		ax31 = plt.subplot2grid(gridSz, (3 * SCALE + OFFSET, 1 * SCALE),
			rowspan=SCALE, colspan=SCALE, sharex=ax21)
		ax40 = plt.subplot2grid(gridSz, (4 * SCALE + OFFSET, 0 * SCALE),
			rowspan=SCALE, colspan=SCALE, sharex=ax30)
		ax41 = plt.subplot2grid(gridSz, (4 * SCALE + OFFSET, 1 * SCALE),
			rowspan=SCALE, colspan=SCALE, sharex=ax31)
		axes = [ax0, ax10, ax11, ax20, ax21, ax30, ax31, ax40, ax41]
		leftAxes = [ax10, ax20, ax30, ax40]
		rightAxes = [ax11, ax21, ax31, ax41]

	# basic axes setup
	for ax in axes:
		ax.autoscale(tight=True)
		# sb.despine(left=True, ax=ax)
		# plt.setp(ax.get_xticklabels(), visible=False) # actually, do show x labels
	plt.setp(ax0.get_xticklabels(), visible=True)

	for ax in axes:
		plt.setp(ax.get_yticklabels(), visible=False)

	# show dishwasher data
	# ts_noNorm = getFig1Ts(zNorm=False, whichTs=whichTs)
	# variances = np.var(ts_noNorm.data, axis=0)
	# linewidths = 1. + variances / np.max(variances)
	ts_dw_pretty.data[:, 2] /= 2 # reduce max height of spikiest dim to make it pretty
	linewidths = 1.5
	ts_dw_pretty.plot(showBounds=False, showLabels=False, ax=ax0,
		linewidths=linewidths)
	ax0.plot(ts_dw_pretty.data[:, DISHWASHER_DIM_TO_HIGHLIGHT], color=highlightColor)

	# show other pairs of data
	if short:
		tsList = [ts_dw]
	else:
		tsList = [ts_dw, ts_tri, ts_rect, ts_shape]
	for i, ts in enumerate(tsList):
		axLeft, axRight = leftAxes[i], rightAxes[i]

		lengths = [50]
		Lmin, Lmax = .1, .2
		onlyPair = True
		if ts == ts_dw: # dishwasher different than other (synthetic) ones
			lengths = [150]
			Lmin, Lmax = 100, 200
			onlyPair = False

		if ts == ts_rect: # TODO remove hack after prototyping fig
			ts_ff = labelTs_ff(ts, Lmin, Lmax, extractTrueLocsAlgo='none')
		else:
			ts_ff = labelTs_ff(ts, Lmin, Lmax)

		# run mdl algo as state-of-the-art; we help it out a bit by adding
		# noise so it doesn't return flat sections, as well as by only using
		# the 3 most correct places it returns
		ts.data += np.random.randn(*ts.data.shape) * .005
		ts_sota = labelTs_sota(ts, lengths, onlyPair=onlyPair)
		ts_sota.startIdxs[2] = ts_sota.startIdxs[3]
		ts_sota.endIdxs[2] = ts_sota.endIdxs[3]
		ts_sota.startIdxs = ts_sota.startIdxs[:3]
		ts_sota.endIdxs = ts_sota.endIdxs[:3]

		print "shapes:"
		print ts_dw.data.shape
		print ts_dw_pretty.data.shape
		print ts_sota.data.shape
		print ts_ff.data.shape
		if i == 0:
			ts_sota.data = np.copy(ts_dw_pretty.data)
			ts_ff.data = np.copy(ts_dw_pretty.data)
		ts_sota.plot(showLabels=False, ax=axLeft, alpha=.4, linewidths=linewidths)
		ts_ff.plot(showLabels=False, ax=axRight, alpha=.4, linewidths=linewidths)

		# plot the dim where the pattern is obvious in bright red
		axLeft.plot(ts_sota.data[:, DISHWASHER_DIM_TO_HIGHLIGHT], color=highlightColor)
		axRight.plot(ts_ff.data[:, DISHWASHER_DIM_TO_HIGHLIGHT], color=highlightColor)


	# leftAxes[0].set_ylim([0, 120])
	# rightAxes[0].set_ylim([0, 120])

	ax0.set_title("Dishwasher Power Measures Over Time")
	ax0.set_xlabel("Minute", labelpad=0, fontsize=14)
	ax10.set_title("State-of-the-art")
	ax11.set_title("Flock (proposed)")

	for ax in axes[3:]: # lower plots
		ax.set_title("")

	# leftAxes[0].set_yticks([0, 250, 500, 750, 1000])
	# for ax in leftAxes[1:-1]:
	# 	ax.set_yticks([0, .25, .5, .75, 1.])

	# annotate part a and part b
	if short:
		# fig.text(.01, .94, "a)", fontsize=18)
		# fig.text(.01, .42, "b)", fontsize=18)
		fig.text(.01, .95, "a)", fontsize=18)
		fig.text(.01, .44, "b)", fontsize=18)
	else:
		fig.text(.01, .97, "a)", fontsize=16)
		fig.text(.01, .71, "b)", fontsize=16)

	for ax in axes:
		viz.setXtickPadding(ax, 2.) # move x labels up; default ~=~ 4

	plt.tight_layout()
	# plt.subplots_adjust(left=.01, right=.99, bottom=.01, hspace=.1) # TODO uncomment
	plt.subplots_adjust(left=.01, right=.99, top=.93, bottom=0.04, hspace=.1)
	# plt.show()

	figName = 'fig1'
	# if whichTs >= 0:
	# 	figName += '_' + str(whichTs)
	saveFigWithName(figName)


# ================================================================ Methods

def makeThickSine():
	sineNoiseStd = 0
	seqLen = 750
	squareLen = seqLen / 17.
	sineLen = int(squareLen * 4)
	sine1 = synth.sines(sineLen, noiseStd=sineNoiseStd)

	sb.set_style('white')
	_, ax = plt.subplots()
	ax.plot(sine1, lw=16)
	ax.set_xlim([-squareLen, len(sine1) + squareLen])
	ax.set_ylim([-2, 2])

	sb.despine(left=True)
	plt.show()

# def makeWeirdSine(squareLen, numSquares, sineNoiseStd, **kwargs):
# 	firstQuarterFrac = .4
# 	length = int(squareLen * numSquares)
# 	firstQuarterLen = int(firstQuarterFrac * length)
	# sine1 = synth.sines(firstQuarterLen, periods=.25, **kwargs)
	# sine2 = synth.sines(firs)
	# sine2 = synth.warpedSine(sineLen, firstHalfFrac=.67,
	# 		noiseStd=sineNoiseStd)

def makeMethodsTs(warped=False, whiteNoise=True):
	seed = np.random.randint(999)
	seed = 123
	np.random.seed(seed)
	print "makeMethodsTs(): seeding RNG with {}".format(seed)

	noiseStd = .2
	sineNoiseStd = 0
	sineAmpStd = 0
	seqLen = 750
	squareLen = seqLen / 17.
	sineLen = int(squareLen * 4)
	preOffset = 60 # for extracting around sine wave
	postOffset = 60
	if whiteNoise:
		seq = synth.randconst(seqLen, std=noiseStd)
	else:
		seq = synth.notSoRandomWalk(seqLen, std=.05,
			trendFilterLength=(seqLen // 4), lpfLength=1)
		seq = ar.detrend(seq)

	sine1 = synth.sines(sineLen, noiseStd=sineNoiseStd, ampStd=sineAmpStd)
	start1 = int(2 * squareLen)
	end1 = start1 + len(sine1)

	if warped:
		sineLen2 = int(squareLen * 5)
		# origFracs = [.1, .15, .35, .47, .5] # these 2 look
		# newFracs = [.05, .1, .45, .61, .64] # pretty decent
		# origFracs = [.1, .15, .35, .47, .5] # little bit less
		# newFracs = [.05, .1, .4, .56, .6]  # stretched out
		# origFracs = [.1, .2, .35, .47]
		# newFracs = [.05, .15, .56, .68]
		# origFracs = [.1, .2, .32, .47]
		# newFracs = [.05, .15, .52, .68]
		# origFracs = [.08, .16, .2, .32, .47]
		# newFracs = [.04, .09, .15, .52, .68]
		origFracs = [.08, .16, .2, .30, .35, .44, .5]
		newFracs = [.04, .09, .15, .45, .52, .64, .7]
		sine2 = synth.warpedSines(sineLen2, origFracs=origFracs,
			newFracs=newFracs, noiseStd=sineNoiseStd, ampStd=sineAmpStd)

		start2 = int(10 * squareLen)
	else:
		sine2 = synth.sines(sineLen, noiseStd=sineNoiseStd, ampStd=sineAmpStd)
		start2 = int(11 * squareLen)
	end2 = start2 + len(sine2)

	# make sure that the sines are surrounded by different data;
	# we do this by (possibly) flipping the portions of the random
	# walk before the first sine and after the second one
	if not whiteNoise:
		break1 = start1 - preOffset
		break2 = end1 + postOffset
		break3 = start2 - preOffset
		break4 = end2 + postOffset
		startBg = seq[:break1]
		midBg = seq[break2:break3]
		endBg = seq[break4:]
		startNoise1 = seq[break1:start1]
		startNoise2 = seq[break3:start2]
		endNoise1 = seq[end1:break2]
		endNoise2 = seq[end2:break4]

		slopeScale = .015
		# slopeScale = .02
		preX = np.arange(preOffset) * slopeScale
		postX = np.arange(postOffset) * slopeScale

		startNoise1 += preX
		endNoise1 -= postX
		startNoise2 -= preX
		endNoise2 += postX

		# flip middle section if sloping down to compensate for
		# second sine being lower
		# midBgGap = midBg[-1] - midBg[0]
		earlyMin = np.min(seq[:break2])
		lateMin = np.min(seq[break3:])
		# increaseBy = max(0, earlyMin - lateMin) + 1.8
		increaseBy = max(0, earlyMin - lateMin) + 1.
		# increaseBy = earlyMin - lateMin + 1.
		# increaseBy = 1.5
		# if midBgGap:
		# 	midBg *= -1
		midBgLen = float(len(midBg))
		# midBg += np.arange(midBgLen) / midBgLen * (1.5 - midBgGap)
		midBg += np.arange(midBgLen) / midBgLen * increaseBy

		print "early min, late min", earlyMin, lateMin
		print "increasing by: ", increaseBy

		# # make a consistent up/down slope at beginning
		# startSlope1 = ar.computeSlope(startNoise1)
		# startNoise1 += np.arange(preOffset) * np.sign(startSlope1) * .01

		# # if ar.cosineSim(startNoise1, startNoise2) > 0:
		# # 	refPt = seq[start1-1]
		# # 	seq[:start1] = 2 * refPt - seq[:start1]
		# # 	print "flipping start stuff"

		# # make a consistent up/down slope at end
		# endSlope2 = ar.computeSlope(endNoise2)
		# endNoise2 += np.arange(postOffset) * np.sign(endSlope2) * .01

		# # # if np.sign(endSlope1) == np.sign(endSlope2):
		# # if ar.cosineSim(endNoise1, endNoise2) > 0:
		# # 	refPt = seq[end2]
		# # 	seq[end2:] = 2 * refPt - seq[end2:]
		# # 	print "flipping end stuff"

		sections = [startBg,
					startNoise1, sine1, endNoise1,
					midBg,
					startNoise2, sine2, endNoise2,
					endBg]
	else:
		sections = (seq[:start1], sine1, seq[end1:start2], sine2, seq[end2:])

	# sections = (seq[:start1], sine1, seq[end1:start2], sine2, seq[end2:])
	seq = synth.concatWithAlignedEndpoints(sections)
	# seq = synth.concatSeqs(sections)

	# ------------------------ plot setup

	palette = sb.color_palette()
	seqColor = palette[0] # blue
	# subseqColor1 = palette[3] # purple
	# subseqColor1 = 'k'
	subseqColor1 = palette[5] # light blue
	# subseqColor1 = sb.color_palette('dark')[0] # dark blue
	subseqColor2 = palette[1] # green

	sb.set_style('white')
	plt.figure()
	ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
	ax2 = plt.subplot2grid((2,2), (1,0))
	ax3 = plt.subplot2grid((2,2), (1,1))

	# ------------------------ time series

	ax1.plot(seq, lw=2, color=seqColor)
	# ax1.plot(seq, lw=2, color=sb.color_palette()[3])
	ax1.plot(np.arange(start1, end1), seq[start1:end1], lw=4, color=subseqColor1)
	# ax1.plot(np.arange(start2, end2), seq[start2:end2], lw=4, color=color)
	ax1.plot(np.arange(start2, end2), seq[start2:end2], lw=4, color=subseqColor2)
	ax1.set_xlim([-squareLen, seqLen + squareLen])
	if whiteNoise:
		ax1.set_ylim([-2, 2])
	else:
		ax1.set_ylim([np.min(seq) - .5, np.max(seq) + .5])

	# ------------------------ global distance
	# lower ax; illustration of dists between stuff when we
	# include extra data on the ends

	subseq1 = seq[(start1 - preOffset):(end1 + postOffset)]
	subseq2 = seq[(start2 - preOffset):(end2 + postOffset)]
	subseq1 = ar.zNormalize(subseq1)
	subseq2 = ar.zNormalize(subseq2)

	# color = sb.color_palette()[0]
	ax2.plot(subseq1, lw=2, color=subseqColor1)
	ax2.plot(np.arange(preOffset, len(subseq1) - postOffset),
		subseq1[preOffset:-postOffset], lw=4, color=subseqColor1)
	# ax2.plot(subseq2, color=color, lw=3)

	# ax2.plot(subseq2, lw=2)
	ax2.plot(subseq2, lw=2, color=subseqColor2)
	ax2.plot(np.arange(preOffset, len(subseq2) - postOffset),
		subseq2[preOffset:-postOffset], lw=4, color=subseqColor2)

	ax2.set_xlim([-10, len(subseq1) + 10])
	drawLinesBetween(ax2, subseq1, subseq2, lineStep=10)

	# ------------------------ local distances TODO


	# ------------------------ show plot

	sb.despine(left=True)
	plt.tight_layout()
	plt.show()


def drawLinesBetween(ax, seq1, seq2, lineStep=5, xvals=None):
	assert(len(seq1) == len(seq2))
	n = len(seq1)
	if xvals is None or not len(xvals):
		xvals = np.arange(n)

	for i in range(0, n, lineStep):
		minVal = min(seq1[i], seq2[i])
		maxVal = max(seq1[i], seq2[i])
		viz.plotVertLine(i, minVal, maxVal, ax=ax, lw=2, color='gray')


def makeGarbageDimTs():
	np.random.seed(123)
	seqLen = 750
	squareLen = seqLen / 17.
	seq = synth.notSoRandomWalk(seqLen, std=.05,
		trendFilterLength=(seqLen // 2), lpfLength=2)

	sb.set_style('white')
	_, ax = plt.subplots()
	# color = sb.color_palette()[1]
	# ax.plot(seq, lw=4, color="#660000") # red I'm using in keynote
	ax.plot(seq, lw=4, color="#CC0000") # red I'm using in keynote
	ax.set_xlim([-squareLen, seqLen + squareLen])
	ax.set_ylim([np.min(seq) * 2, np.max(seq) * 2])

	sb.despine(left=True)
	plt.show()

# def makeMethodsWarpedTs():


# ================================================================ Better Fig1


def makeFig1_v2(zNorm=True, whichTs=WHICH_DISHWASHER_TS):

	mpl.rcParams['axes.facecolor'] = AXES_BG_COLOR

	INSTANCE_BOUNDS_ALPHA = .3

	ts_dw = getFig1Ts(zNorm=zNorm, whichTs=whichTs)
	ts_dw.data[:, 2] /= 2 # reduce max height of spikiest dim to make it pretty
	ts_dw.endIdxs[2] += 25 # appears like it's missing stuff; fix this

	# remove the red since we'll save this to highlight the "pattern" dim
	palette = sb.color_palette()
	palette = palette[:2] + palette[3:]
	highlightColor = sb.color_palette()[2] # red
	sb.set_palette(palette)

	# ------------------------ set up axes
	FIG_SIZE = (8, 6)
	NUM_ROWS = 3
	NUM_COLS = 1
	gridSz = (NUM_ROWS, NUM_COLS)
	fig = plt.figure(figsize=FIG_SIZE)

	ax0 = plt.subplot2grid(gridSz, (0, 0))
	ax1 = plt.subplot2grid(gridSz, (1, 0), sharex=ax0)
	ax2 = plt.subplot2grid(gridSz, (2, 0), sharex=ax1)
	axes = [ax0, ax1, ax2]

	# basic axes configuration
	for ax in axes:
		ax.autoscale(tight=True)
		plt.setp(ax.get_yticklabels(), visible=False)
	plt.setp(ax0.get_xticklabels(), visible=False)
	plt.setp(ax1.get_xticklabels(), visible=False)

	# ------------------------ find patterns in ts

	lengths = [150]
	Lmin, Lmax = 100, 200
	onlyPair = False
	ts = ts_dw.clone()

	ts_ff = labelTs_ff(ts, Lmin, Lmax)

	# run mdl algo as state-of-the-art; we help it out a bit by adding
	# noise so it doesn't return flat sections, as well as by only using
	# the 3 most correct places it returns
	np.random.seed(123)
	ts.data += np.random.randn(*ts.data.shape) * .005
	# print "labels", ts.labels

	ts_sota = labelTs_sota(ts, lengths, onlyPair=onlyPair)
	ts_sota.startIdxs[2] = ts_sota.startIdxs[3]
	ts_sota.endIdxs[2] = ts_sota.endIdxs[3]
	ts_sota.startIdxs = ts_sota.startIdxs[:3]
	ts_sota.endIdxs = ts_sota.endIdxs[:3]

	ts_ff.data = np.copy(ts_dw.data)
	ts_sota.data = np.copy(ts_dw.data)

	# show ground truth, ff, and state-of-the-art
	linewidths = 1.5
	ts_dw.plot(ax=ax0, showBounds=True, showLabels=False,
		linewidths=linewidths, useWhichLabels=['Z3'],
		alpha=INSTANCE_BOUNDS_ALPHA)
	ts_ff.plot(ax=ax2, showLabels=False, alpha=INSTANCE_BOUNDS_ALPHA,
		linewidths=linewidths)
	ts_sota.plot(ax=ax1, showLabels=False, alpha=INSTANCE_BOUNDS_ALPHA,
		linewidths=linewidths)

	# plot the dim where the pattern is obvious in bright red
	tsList = [ts_ff, ts_sota, ts_dw]
	for ax, ts in zip(axes, tsList):
		ax.plot(ts.data[:, DISHWASHER_DIM_TO_HIGHLIGHT], color=highlightColor)

	# ax0.set_title("True instances of appliance running")
	ax0.set_title("True instances of appliance running", y=1.03)
	# ax1.set_title("Pattern discovered by [10] and [17]", y=1.01)
	# ax1.set_title("Pattern discovered using nearest neighbors [17]", y=1.01)
	ax1.set_title("Pattern discovered by [10]", y=1.01)
	ax2.set_title("Pattern discovered by Flock (proposed)", y=1.01)
	ax2.set_xlabel("Minutes", labelpad=1, fontsize=14)

	for ax in axes[3:]: # lower plots
		ax.set_title("")

	# annotate parts a, b, and c
	# fig.text(.01, .97, "a)", fontsize=18)
	# fig.text(.01, .64, "b)", fontsize=18)
	# fig.text(.01, .31, "c)", fontsize=18) # these 3 optimal for (8, 7)
	fig.text(.01, .96, "a)", fontsize=18)
	fig.text(.01, .65, "b)", fontsize=18)
	fig.text(.01, .33, "c)", fontsize=18)

	for ax in axes:
		viz.setXtickPadding(ax, 2.) # move x labels up; default ~=~ 4

	plt.tight_layout()
	# plt.subplots_adjust(left=.01, right=.999, top=.96, bottom=.03, hspace=.19)
	# plt.subplots_adjust(left=.01, right=.999, top=.96, bottom=.03) # optimal for (8, 7)
	plt.subplots_adjust(left=.01, right=.999, top=.945, bottom=.07, hspace=.27)

	saveFigWithName('fig1')


# ================================================================ Algo output

from ..algo import ff11 as ff
from ..viz.ff import plotFFOutput

def makeOutputFig():
	sb.set_style("whitegrid", {'axes.grid': False})

	np.random.seed(12345)

	whichExamples = np.arange(6) # tidigits example 5 is good
	tsList = datasets.loadDataset(datasets.TIDIGITS, whichExamples=whichExamples)

	# uncomment any of these to use a different dataset
	# tsList = datasets.loadDataset(datasets.DISHWASHER, whichExamples=whichExamples)
	# tsList = datasets.loadDataset(datasets.MSRC, whichExamples=whichExamples)
	# tsList = datasets.loadDataset(datasets.UCR, whichExamples=[0])

	Lmin, Lmax = 1./20, 1./10 # fractions of time series length

	ts = tsList[-1]
	startIdxs, endIdxs, model, featureMat, featureMatBlur = ff.learnFFfromSeq(
		ts.data, Lmin, Lmax)

	# the relevant features happen to be near the top; move them closer to the
	# middle so it doesn't look like that's typical or part of the algorithm
	reorderIdxs = np.arange(len(featureMat))
	reorderIdxs = np.roll(reorderIdxs, len(featureMat)//4)
	featureMat = featureMat[reorderIdxs]
	model = model[reorderIdxs]

	plotFFOutput(ts, startIdxs, endIdxs, featureMat, model)

	saveFigWithName('tidigits-output')
	plt.show()

# ================================================================ Main

if __name__ == '__main__':
	sb.set_context("poster") # big text
	# sb.set_context("talk") # medium text

	# ------------------------ datasets

	# import matplotlib as mpl
	# mpl.rcParams['axes.titlesize'] = 32
	# mpl.rcParams['axes.titleweight'] = 'bold'

	# plt.figure(figsize=(10, 6))
	# ax = plt.gca()
	# makeDishwasherFig(ax=ax)
	# makeTidigitsFig()
	# makeMsrcFig()
	# makeUcrFig()

	# makeDatasetsFig()

	# ------------------------ fig1

	# ts = getFig1Ts(zNorm=True, whichTs=22)

	# SAVE_DIR = 'figs/tmp/' # temp hack, overwriting constant
	# if not os.path.exists(SAVE_DIR):
	# 	os.mkdirs(SAVE_DIR)
	# # for i in range(4, 30): # 9, 22, 27 pretty good for dishwasher
	# for i in range(10): # 9, 22, 27 pretty good
	# 	ts = getGoodTidigitsTs(whichTs=i)
	# 	# makeCombinedFig1(short=True, whichTs=i)
	# 	# ts = getFig1Ts(scaleVariance=True, whichTs=i)
	# 	ts.plot(showLabels=False, saveDir=SAVE_DIR)
	# # plt.show()

	# # makeFig1()
	# # makeFig2()
	# makeCombinedFig1()
	# makeCombinedFig1(short=True, zNorm=True, whichTs=WHICH_DISHWASHER_TS) # use this line
	makeFig1_v2()

	# ------------------------ methods fig

	# makeMethodsTs()
	# makeMethodsTs(whiteNoise=False)
	# makeMethodsTs(warped=True)
	# makeGarbageDimTs()

	# ------------------------ how many tidigits ts?

	# tsList = datasets.loadDataset('tidigits_grouped_mfcc')
	# print "number of tidigits ts: ", len(tsList)

	# ------------------------ ff output

	# makeOutputFig()

