#!/usr/env/python

import os
import numpy as np

from ..datasets import read_msrc as msrc
from ..datasets import synthetic as synth
# from .utils.subseq import findMotif, findMotifOfLength
from ..algo.motif import findMotif
from ..utils import arrays as ar
from ..utils import files

from ..viz.motifs import plotMotif, findAndPlotMotifInstances

# ================================================================
# utility funcs
# ================================================================


# ================================================================
# synthetic data
# ================================================================

def tryFunc(motifFunc, *args, **kwargs):
	x, length = motifFunc(*args, **kwargs)
	motif = findMotif([x], length)
	plotMotif(x, motif)

def trySines(*args, **kwargs):
	tryFunc(synth.sinesMotif, *args, **kwargs)

def tryTriangles(*args, **kwargs):
	tryFunc(synth.trianglesMotif, *args, **kwargs)

def tryRects(*args, **kwargs):
	tryFunc(synth.rectsMotif, *args, **kwargs)

def tryShapes(*args, **kwargs):
	tryFunc(synth.multiShapesMotif, *args, **kwargs)

def syntheticMain():
	np.random.seed(1234)
	# tryShapes(noise=.02)
	trySines(noise=.1, backgroundNoise=.1)
	# tryTriangles(noise=0.02, backgroundNoise=.02)
	# tryRects(noise=.001) # still basically gets it right
	# tryRects(noise=.01) # gets most of the rect
	# tryRects(noise=.02) # gets only piece of the rect

# ================================================================
# real data
# ================================================================

def lengthNormalize(length, dist):
	# return dist / (length*length*length)
	# return dist / (length*length) # always finds max len motif
	return dist / length

# MSRC_SAVE_DIR = 'figs/motif/msrc/'

def tryMSRC(idxs=[0], whichDims=[], allInstances=False, saveDir=None):
	DOWNSAMPLE_BY = 5

	# dataFiles, tagFiles = msrc.getFileNames()
	# recordings = msrc.getRecordings(idxs=range(0, 500, 20))
	recordings = msrc.getRecordings(idxs=idxs)
	# recordings = msrc.getRecordings(idxs=[1])
	# recordings = msrc.getRecordings()
	for r in recordings:

		if len(r.data) < 5 * DOWNSAMPLE_BY:
			continue

		if len(whichDims) > 0:
			r.data = r.data[:, whichDims]

		print("downsampling MSRC data from %d samples to %d" %
			(len(r.data), len(r.data) / DOWNSAMPLE_BY))
		r.data = ar.downsampleMat(r.data, rowsBy=DOWNSAMPLE_BY)

		# TODO remove
		# r.sampleTimes = ar.downsampleMat(r.sampleTimes, rowsBy=DOWNSAMPLE_BY)
		# import matplotlib.pyplot as plt
		# plt.plot(r.sampleTimes, r.data)
		# # plt.plot(np.arange(len(r.sampleTimes)), r.sampleTimes)
		# plt.show()
		# return

		lMin = len(r.data) / 20
		lMax = len(r.data) / 8
		lStep = 2
		lengths = range(lMin, lMax+1, lStep)
		# lengths = len(r.data) / 10 # TODO remove
		# print "lengths", lengths
		# return

		fileName = ""
		if saveDir:
			fileName = os.path.join(fileName, "msrc", "downsample%d/" % DOWNSAMPLE_BY)
			if allInstances:
				fileName += 'all_'
			fileName += 'rec%d_downsamp%d_%d-%d-%d' % (r.id, DOWNSAMPLE_BY, lMin, lMax, lStep)
			fileName += '_normLenSq'
			fileName += '.pdf'
			fileName = os.path.join(saveDir, fileName)
			files.ensureDirExists(saveDir)

		if allInstances:
			labelIdxs = r.gestureIdxs / DOWNSAMPLE_BY
			findMotifKwargs = {}
			# findMotifKwargs['threshAlgo'] = 'minnen'
			findMotifKwargs['threshAlgo'] = 'maxsep'
			findMotifKwargs['addData'] = 'gauss'
			# findMotifKwargs['addData'] = 'freqMag'
			findMotifKwargs['addDataFractionOfLength'] = 5.
			findAndPlotMotifInstances(r.data, lengths,
				truthStartEndPairs=labelIdxs, findMotifKwargs=findMotifKwargs,
				saveas=fileName, title=r.gestureLabel, linewidth=2)
		else:
			motif = findMotif([r.data], lengths, lengthNormFunc=lengthNormalize, returnNormedSeqs=True)
			plotMotif(r.data, motif, saveas=fileName)


# ================================================================ Main

if __name__ == '__main__':
	# saveDir = 'figs/motif/'
	saveDir = None

	# syntheticMain()
	# tryMSRC(idxs=range(0, 550, 2), saveDir=None)
	# tryMSRC(idxs=[2], allInstances=False, saveDir=None)
	# tryMSRC(idxs=[2], allInstances=True, saveDir=saveDir, whichDims=[0])
	# tryMSRC(idxs=[2], allInstances=True, saveDir=saveDir)
	tryMSRC(idxs=range(5), allInstances=True, saveDir=None)
	# tryMSRC(idxs=range(1), allInstances=True, saveDir=None)
	# tryMSRC(idxs=range(1), allInstances=False, saveDir=None)
