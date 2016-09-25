#!/bin/env/python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

from ..algo.motif import findMotif, findAllMotifInstances
from ..utils.subseq import simMatFromDistTensor

from viz_utils import plotRect, plotRanges

def findAndPlotMotif(seq, lengths, **kwargs):
	motif = findMotif([seq], lengths)
	plotMotif(seq, motif, **kwargs)

def findAndPlotMotifInstances(seq, lengths, truthStartEndPairs=None,
	saveas=None, findMotifKwargs=None, **kwargs):
	# XXX this func will break if seq is a list of seqs, not just one ndarray

	if findMotifKwargs:
		startIdxs, instances, motif = findAllMotifInstances([seq], lengths,
			**findMotifKwargs)
	else:
		startIdxs, instances, motif = findAllMotifInstances([seq], lengths)

	# ------------------------ plot reported motif instances

	endIdxs = startIdxs + motif.length
	startEndPairs = np.c_[startIdxs, endIdxs]
	ax = plotMotifInstances(seq, startEndPairs, **kwargs)

	# ------------------------ plot ground truth
	if truthStartEndPairs is not None and len(truthStartEndPairs):
		try:
			if len(truthStartEndPairs[0]) == 1: # single points, not ranges
				color = 'k'
				truthStartEndPairs = np.asarray(truthStartEndPairs)
				truthStartEndPairs = np.c_[truthStartEndPairs, truthStartEndPairs]
			else:
				color = 'g'
		except: # elements are scalars and so len() throws
			color = 'k'
			truthStartEndPairs = np.asarray(truthStartEndPairs)
			truthStartEndPairs = np.c_[truthStartEndPairs, truthStartEndPairs]

		# make a vert line (spanning less than full graph height)
		# where the labels are
		yMin, yMax = np.min(seq), np.max(seq)
		yRange = yMax - yMin
		lineMin, lineMax = [yMin + frac * yRange for frac in (.4, .6)]
		plotMotifInstances(None, truthStartEndPairs, ax=ax,
			color=color, linestyle='-', lw=2,
			ymin=lineMin, ymax=lineMax)

	if saveas:
		plt.savefig(saveas)
	else:
		plt.show()

def plotMotif(seq, motif, showExtracted=True, color='gray',
	title=None, saveas=None):

	start1 = motif[3]
	start2 = motif[4]
	end1 = start1 + len(motif[0]) - 1
	end2 = start2 + len(motif[1]) - 1

	# just show where the motif is in the original signal
	if not showExtracted:
		_, ax = plt.subplots()
		ax.autoscale(False)
		ax.plot(seq)
		plotRect(ax, start1, end1, color=color)
		plotRect(ax, start2, end2, color=color)
		if saveas:
			plt.savefig(saveas)
		else:
			plt.show()
		return ax

	# set up axes
	ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
	ax2 = plt.subplot2grid((2,2), (1,0))
	ax3 = plt.subplot2grid((2,2), (1,1))
	ax1.autoscale(tight=True)
	ax2.autoscale(tight=True)
	ax3.autoscale(tight=True)

	# plot raw ts on top and motif instances on the bottom
	ax1.plot(seq, lw=2)
	ax2.plot(motif[0], lw=2)
	ax3.plot(motif[1], lw=2)
	ax1.set_title('Original Signal')
	ax2.set_title('Motif Instance at %d' % start1)
	ax3.set_title('Motif Instance at %d' % start2)

	# draw rects in the ts where the motif is
	plotRect(ax1, start1, end1, color=color)
	plotRect(ax1, start2, end2, color=color)

	plt.tight_layout()
	if saveas:
		plt.savefig(saveas)
	else:
		plt.show()

	return ax1, ax2, ax3

def plotMotifInstances(seq, startEndIdxPairs, title=None, ax=None,
	saveas=None, **kwargs):

	if ax is None:
		_, ax = plt.subplots()
		# ax.autoscale(False) # makes it not actually work...

	if seq is not None and len(seq): # plot original seq if one is provided
		ax.plot(seq, **kwargs)

	plotRanges(ax, startEndIdxPairs, **kwargs)

	if not title:
		title = "Motif Instances in Data"
	ax.set_title(title)
	if seq is not None and len(seq):
		ax.set_ylim([np.min(seq), np.max(seq)])
		ax.set_xlim([0, len(seq)])

	if saveas:
		plt.savefig(saveas)

	return ax

def showPairwiseSims(origSignal, m, simMat, clamp=True, pruneCorrAbove=-1,
	plotMotifs=True, showEigenVect=False, hasPadding=True, saveas=None):

	print "origSignal shape", origSignal.shape
	# padLen = len(origSignal) - simMat.shape[1]
	padLen = m - 1 if hasPadding else 0
	subseqLen = m

	plt.figure(figsize=(8,10))
	if showEigenVect:
		ax1 = plt.subplot2grid((20,10), (0,0), colspan=8, rowspan=5)
		ax2 = plt.subplot2grid((20,10), (5,0), colspan=8, rowspan=15)
		ax3 = plt.subplot2grid((20,10), (5,8), colspan=2, rowspan=15)
		ax3.autoscale(tight=True)
		ax3.set_title('Extracted')
	else:
		ax1 = plt.subplot2grid((4,2), (0,0), colspan=2)
		ax2 = plt.subplot2grid((4,2), (1,0), colspan=2, rowspan=3)
	ax1.autoscale(tight=True)
	ax2.autoscale(tight=True)
	ax1.set_title('Original Signal')
	ax1.set_ylabel('Value')

	if pruneCorrAbove > 0:
		ax2.set_title('Subsequence Cosine Similarities to Dictionary Sequences')
	else:
		ax2.set_title('Subsequence Pairwise Cosine Similarities')
	ax2.set_xlabel('Subsequence Start Index')
	ax2.set_ylabel('"Dictionary" Sequence Number')

	seq = origSignal
	imgMat = simMat

	print "imgMat shape: ", imgMat.shape

	# 	# show magnitude of similarities in each row in descending order; there are
	# 	# only about 60 entries > .01 in *any* row for msrc, and way fewer in most
	# 	# plt.figure()
	# 	# thresh = .5
	# 	# sortedSimsByRow = np.sort(imgMat, axis=1)
	# 	# sortedSimsByRow = sortedSimsByRow[:, ::-1]
	# 	# nonzeroCols = np.sum(sortedSimsByRow, axis=0) > thresh # ignore tiny similarities
	# 	# sortedSimsByRow = sortedSimsByRow[:, nonzeroCols]
	# 	# # plt.imshow(sortedSimsByRow)
	# 	# # plt.plot(np.mean(sortedSimsByRow, axis=1))
	# 	# plt.plot(np.sum(sortedSimsByRow > thresh, axis=1)) # entries > thresh per row

	# if pruneCorrAbove > 0.:
	# 	print "ImgMat Shape:"
	# 	print imgMat.shape
	# 	imgMat = removeCorrelatedRows(imgMat, pruneCorrAbove)
	# 	print imgMat.shape
	# 	print "NaNs at:", np.where(np.isnan(imgMat))[0]
	# 	print "Infs at:", np.where(np.isinf(imgMat))[0]

	# power iteration to see what we get
	if showEigenVect:
		width = int(subseqLen * 1.5)
		nRows, nCols = imgMat.shape
		nPositions = nCols - width + 1
		if nPositions > 1:
			elementsPerPosition = nRows * width # size of 2d slice
			dataMat = np.empty((nPositions, elementsPerPosition))
			# for i in range(nPositions): 		 	# step by 1
			for i in range(0, nPositions, width): 	# step by width, so non-overlapping
				startCol = i
				endCol = startCol + width
				data = imgMat[:, startCol:endCol]
				dataMat[i] = data.flatten()
			# ah; power iteration is for cov matrix, cuz needs a square mat
			# v = np.ones(elementsPerPosition) / elementsPerPosition # uniform start vect
			# for i in range(3):
			# 	v = np.dot(dataMat.T, v)
			svd = TruncatedSVD(n_components=1, random_state=42)
			svd.fit(dataMat)
			v = svd.components_[0]
			learnedFilt = v.reshape((nRows, width))
			ax3.imshow(learnedFilt) # seems to be pretty good
			# plt.show()

	ax1.plot(seq)
	ax2.imshow(imgMat, interpolation='nearest', aspect='auto')
	plt.tight_layout()

	if plotMotifs:
		searchSeq = seq
		print "searchSeq shape:", searchSeq.shape
		motif = findMotif([searchSeq], subseqLen) # motif of min length
		start1 = motif[3]
		start2 = motif[4]
		end1 = start1 + len(motif[0]) - 1
		end2 = start2 + len(motif[1]) - 1

		ax2.autoscale(False)
		color = 'grey'
		plotRect(ax1, start1, end1, color=color)
		plotRect(ax2, start1, end1, color=color)
		plotRect(ax1, start2, end2, color=color)
		plotRect(ax2, start2, end2, color=color)

		print "imgMat shape: ", imgMat.shape
		print "padLen: ", padLen
		if padLen:
			searchSeq = imgMat[:,:-padLen].T
		else:
			searchSeq = imgMat.T
		print "searchSeq shape:", searchSeq.shape
		print "subseqLen:", subseqLen
		motif = findMotif([searchSeq], subseqLen) # motif of min length
		start1 = motif[3]
		start2 = motif[4]
		end1 = start1 + len(motif[0]) - 1
		end2 = start2 + len(motif[1]) - 1
		print [start1, end1, start2, end2]

		color = 'm' # magenta
		plotRect(ax1, start1, end1, color=color)
		plotRect(ax2, start1, end1, color=color)
		plotRect(ax1, start2, end2, color=color)
		plotRect(ax2, start2, end2, color=color)

	if saveas:
		plt.savefig(saveas)
	else:
		plt.show()

	if showEigenVect:
		return ax1, ax2, ax3
	return ax1, ax2

def showPairwiseDists(origSignal, m, Dtensor, **kwargs):
	padLen = len(origSignal) - Dtensor.shape[1]
	simMat = simMatFromDistTensor(Dtensor, m, padLen)
	showPairwiseSims(origSignal, m, simMat, **kwargs)

