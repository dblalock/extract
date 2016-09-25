#!/usr/env/python

import numpy as np
from scipy.ndimage import filters

# from sklearn.decomposition import TruncatedSVD

from joblib import Memory
memory = Memory('./output', verbose=0)

from ..datasets import synthetic as synth
from ..datasets import read_msrc as msrc
from ..utils import sequence
from ..utils import arrays as ar
from ..utils import subseq as sub
from ..utils import kdtree as kd

from ..utils.arrays import downsampleMat
from ..utils.misc import randStrId
from ..viz.motifs import showPairwiseSims

# so this needs to:
# -learn a set of unique subseqs from the data
# -convert the data into a similarity mat using these seqs as a dictionary
# -find the best motif in the time domain as an initial guess
# -find everywhere this motif occurs (still in the time domain) based on dist
# 	-plot this and make sure it's always reasonable
# -initialize u and w to 0 and uniform
# -iteratively (prolly just like 5 times for now):
# 	-assign points to instances or none (maybe always an instance for now)
#	-find MLE gaussian for each dim
#	-update motif to maximize likelihood of data
#		-update position gaussian u and E
#		-update weights of each gaussian based on dominant eigenvect


def readDataset(dataset):
	dataId = ''
	if dataset == 'msrc':
		recordingNum = np.random.choice(np.arange(500), 1)
		r = list(msrc.getRecordings(idxs=[recordingNum]))[0]
		seq = downsampleMat(r.data, rowsBy=5)
		dims = np.random.choice(np.arange(seq.shape[1]), 5)
		seq = seq[:, dims]
		dataId = 'num=%d' % recordingNum
		dataId += '_dims=' + np.array_str(dims)
	elif dataset == 'shapes':
		seq, _ = synth.multiShapesMotif()
		dataId = randStrId()
	elif dataset == 'triangles':
		seq, _ = synth.trianglesMotif()
		dataId = randStrId()
	elif dataset == 'sines':
		seq, _ = synth.sinesMotif()
		dataId = randStrId()

	return seq, dataId


# def computeNeighborsForDims(seqs, length, maxDist, k=-1):
# 	occurIdxsForDims, XnormForDims = sub.uniqueSubseqs(seqs, length, maxDist, tieDims=False)
# 	return neighborsFromOccurrencesAndSubseqs(occurIdxsForDims, XnormForDims, k)

def neighborsFromOccurrencesAndSubseqs(occurIdxsForDims, XnormForDims, k=-1):
	"""returns a list of dicts, one for each dim. Each dict
	maps startIdx -> (featureIdx, dist)"""
	neighborsForDims = []

	if k < 1:
		k = XnormForDims[0].shape[1] # subseq length

	# find the k nearest neighbors for each subseq
	for dim in range(len(occurIdxsForDims)):
		idxs2positions = occurIdxsForDims[dim]
		Xnorm = XnormForDims[dim]
		print "dim %d has %d subseqs" % (dim, len(Xnorm))
		length = Xnorm.shape[1]
		nFeatures = len(idxs2positions)
		nNeighbors = min(k, nFeatures)

		# create kd tree full of dictionary seqs
		tree = kd.create(dimensions=length)
		sortedIdxs = sorted(idxs2positions.keys())
		print "findNeighbors(): numFeatures for dim %d: %d" % (dim, len(sortedIdxs))
		for i, idx in enumerate(sortedIdxs): # i = idx within dict seqs, idx = orig idx
			tree.add(Xnorm[idx], metadata=i)
		tree.rebalance()

		# subseq -> most similar dictionary seqs
		startIdxs2neighbors = {}
		for startIdx, row in enumerate(Xnorm):
			neighbors = tree.search_knn(row, nNeighbors)
			idxsAndDists = map(lambda nodeAndDist: # map to (feature idx, dist)
				(nodeAndDist[0].metadata, nodeAndDist[1]), neighbors)
			startIdxs2neighbors[startIdx] = idxsAndDists
		neighborsForDims.append(startIdxs2neighbors)
		print "dim %d: found neighbors for %d startIdxs" % (dim, len(startIdxs2neighbors))

	return neighborsForDims

# def simTensorFromNeighborsForDims(neighborsForDims, nFeatures, nSubseqs, length):
def simsFromNeighborsForDims(neighborsForDims, nFeatures, nSubseqs, length):
	"""returns an (nFeatures x nSubseqs) similarity matrix for each dimension"""
	nDims = len(neighborsForDims)
	# simTensor = np.zeros((nDims, nFeatures, nSubseqs)) # dim x feature x time
	simsForDims = []
	# print "simsFromNeighborsForDims(): nFeatures, nSubseqs = (%d, %d)" % (nFeatures, nSubseqs)
	for dim in range(nDims):
		neighborsDict = neighborsForDims[dim]
		simMat = np.zeros((nFeatures, nSubseqs))
		for seqIdx in np.arange(nSubseqs):
			idxsAndDists = neighborsDict[seqIdx]
			featureIdxs, dists = zip(*idxsAndDists)
			sims = sub.zNormDistsToCosSims(dists, length)
			sims = np.maximum(0, sims)
			# simTensor[dim, featureIdxs, seqIdx] = sims
			simMat[featureIdxs, seqIdx] = sims
		simsForDims.append(simMat)

	# return simTensor
	return simsForDims

def vstackTensorFirstDim(simTensor):
	return simTensor.reshape((-1, simTensor.shape[2]))

def allSimsForDims(seqs, length):
	"""seqs, length -> (nDims x p x p) tensor of cos sims, where p is total
	number of subsequences in all seqs provided"""
	if sequence.isListOrTuple(seqs):
		nDims = len(seqs[0].shape)
	else:
		nDims = len(seqs.shape)

	if nDims == 1:
		seqs = map(lambda s: s.reshape((-1, 1)), seqs) # make col vects

	# compute total number of subseqs; n - m + 1 for each seq
	nSubseqs = np.sum([len(seq) for seq in seqs]) 	# sum of n
	nSubseqs -= len(seqs) * (length-1) 				# sum of (m-1)
	sims = np.empty((nDims, nSubseqs, nSubseqs))
	for dim in range(nDims):
		signals = map(lambda s: s[:, dim], seqs)
		allSubs = sub.allZNormalizedSubseqs(signals, length)
		allSubs = ar.normalizeRows(allSubs)
		sims[dim] = np.dot(allSubs, allSubs.T)
		assert(np.max(sims[dim]) <= 1.00001)

	return sims

# @memory.cache
def computeSimMat(seqs, length, maxDist, k=-1, matForEachSeq=False,
	normFeatures=None, maximaOnly=False, removeSelfMatch=False, **kwargs):
	print "computeSimMat(): excecuting fresh"

	# create combined similarity matrix
	if k > 0: # use knn in each column
		occurIdxsForDims, XnormForDims = sub.uniqueSubseqs(seqs, length, maxDist, tieDims=False)
		nFeatures = max(map(lambda idxs: len(idxs), occurIdxsForDims)) # max for any dim
		nSubseqs = len(XnormForDims[0]) # assume all dims the same
		neighborsForDims = neighborsFromOccurrencesAndSubseqs(occurIdxsForDims, XnormForDims, k)
		simsForDims = simsFromNeighborsForDims(neighborsForDims, nFeatures, nSubseqs, length)
		simMat = np.vstack(simsForDims)
		# processs matrix to suck less
		simMat = ar.removeZeroRows(simMat) # TODO don't even allocate these
		counts = np.sum(simMat > 0.001, axis=1)
		simMat = simMat[counts > 0]
	else: # use all similarities in each column
		# simsForDims = allSimsForDims(seqs, length)
		# simMat = vstackTensorFirstDim(simsForDims)
		# simMat = np.maximum(simMat, 0) # clamp negative values at 0
		simMat = sub.similarityMat(seqs, length, pad=False, **kwargs)

	if maximaOnly:
		simMat = localMaxFilterSimMat(simMat)

	if removeSelfMatch:
		# set self-matches (which always have the maximum value of 1)
		# to the 2nd-highest value in the row instead; we don't just
		# use the diagonal because we may have removed many features
		rowIdxs = np.arange(simMat.shape[1])
		maxIdxs = np.argmax(simMat, axis=1)
		maxIdxs = (rowIdxs, maxIdxs) # 2d indices

		simMat[maxIdxs] = 0
		secondMaxIdxs = np.argmax(simMat, axis=1)
		secondMaxIdxs = (rowIdxs, secondMaxIdxs)
		simMat[maxIdxs] = simMat[secondMaxIdxs]

	if normFeatures == 'mean':
		simMat = ar.meanNormalizeRows(simMat)
	elif normFeatures == 'z':
		simMat = ar.zNormalizeRows(simMat)
	elif normFeatures == 'std':
		simMat = ar.stdNormalizeRows(simMat)

	if not matForEachSeq: # return combined similarity mat for all seqs
		return simMat
	else: 	# return similarity mat for each individual seq passed in
		# split cols (samples) by origin
		fromSeq = sub.computeFromSeq(seqs, length)
		assert(len(fromSeq) == simMat.shape[1])
		seqNum2Mat = sequence.splitElementsBy(lambda i, _: fromSeq[i], simMat.T)
		# combine rows from each origin
		keys = sorted(seqNum2Mat.keys())
		mats = map(lambda k: np.vstack(seqNum2Mat[k]).T, keys)

		# print [len(seqNum2Mat[k]) for k in keys]

		return mats

def localMaxFilterSimMat(simMat, allowEq=True):
	# zero everything but the relative maxima in each row

	# compute discrete 1st deriv
	# diffs = simMat[:, 1:] - simMat[:,:-1]
	# diffs = np.sign(diffs)

	# if this diff is positive and prev one was negative, it's a rel max;
	# except that actually, we only require one to be strictly > or < 0
	# isMax = diffs[1:] - diffs[:-1] # want diffs to be >0, <=0 or >=0, <0

	# nevermind, just use vanilla rel maxima
	idxs = ar.idxsOfRelativeExtrema(simMat, maxima=True, allowEq=allowEq, axis=1)
	maximaOnly = np.zeros(simMat.shape)
	maximaOnly[idxs] = simMat[idxs]
	return maximaOnly

def filterSimMat(simMat, filtLength, filtType, scaleFilterMethod='max1'):
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
	# print filt
	# filt = np.tile(filt, (simMat.shape[0], 1))
	# print filt.shape
	return filters.convolve1d(simMat, weights=filt, axis=1, mode='constant')

def l1Project(A, desiredNorm=1.):
	# we want to find lambda such that, when we subtract lambda from
	# every entry and clip at 0, the sum is desiredNorm = 1.0
	desiredNorm = 1.
	# vectorize A
	v = A.flatten()
	n = len(v)
	# sort absolute values in descending order and compute cumulative sum
	vAbs = np.abs(v)
	vSort = np.sort(vAbs)[::-1]
	cumAbs = np.cumsum(vSort)
	# determine lambda based on highest value that would still be above 0
	aboveZeros = np.nonzero(vSort * np.arange(1, n+1) > (cumAbs - desiredNorm))[0]
	highestIdx = aboveZeros[-1]
	lamda = (cumAbs[highestIdx] - desiredNorm) / (highestIdx + 1.)
	# subtract off lambda from A
	vNorm = (vAbs - lamda).clip(min=0.)
	vNorm *= np.sign(v) # undo absolute value
	return vNorm.reshape(A.shape)

def lurnPatternInSimMat(simMat, width):
	nRows, nCols = simMat.shape
	nPositions = nCols - width + 1
	assert(nPositions > 0)

	elementsPerPosition = nRows * width # size of 2d slice
	dataMat = np.empty((nPositions, elementsPerPosition))
	for i in range(nPositions): 		 	# step by 1
		# for i in range(0, nPositions, width): # step by width, so non-overlapping
		startCol = i
		endCol = startCol + width
		data = simMat[:, startCol:endCol]
		dataMat[i] = data.flatten()

	v = lurnPattern(dataMat)
	# svd = TruncatedSVD(n_components=1, random_state=42)
	# svd.fit(dataMat)
	# v = svd.components_[0]
	learnedFilt = v.reshape((nRows, width))
	# ax3.imshow(learnedFilt) # seems to be pretty good

def lurnPattern(X, w=None): # TODO pass in avg of stuff around best motif as w0
	"""learns to weight features in each row of 2D data matrix X"""
	nRows, nFeatures = X.shape

	# TODO use minibatches if doing this as grad descent
	if w is None:
		w = np.zeros(nFeatures) + 1./nFeatures # uniform initial weights

	X = ar.meanNormalizeCols(X) # TODO do this before passing X in, cuz this is wrong

	for ep in range(10): # train for fixed number of epochs for now
		Y = np.dot(X, w) # excitation vect for all examples
		dW = (X - w) * Y # levy's rule dW--each row is gradient towards dominant eigenvect

		# now weight the dW by some function of the excitation reflecting how
		# sure we are that each example is an instance of the pattern
		Z = np.exp(Y*Y)
		Z = Z / np.sum(Z)
		dW_weighted = dW * Z.reshape((-1,1)) # make Z a col vect to scale each row

		dw_flat = np.sum(dW_weighted, axis=0)
		lurn = .2/(1 + ep) # learning rate
		w += lurn * dw_flat

		# for i, x in enumerate(X):
		# 	y = np.dot(w, x)
		# 	dw = (x - w)*y  # levy's rule; no "x - E[x]"" because we killed mean
			# lurn_w =


def mainShowSimMat():
	seq, dataId = readDataset('msrc')
	# seq, dataId = readDataset('shapes')
	# seq, dataId = readDataset('triangles')
	# seq, dataId = readDataset('sines')
	print seq.shape

	length = 12
	# k = 20
	k = -1
	maxDist = .2 * length
	# maxDist = 0.
	simMat = computeSimMat(seq, 8, maxDist, k)
	padding = np.zeros((simMat.shape[0], length-1))
	simMat = np.hstack((simMat, padding)) # padding
	showPairwiseSims(seq, length, simMat, plotMotifs=True, showEigenVect=False)

def mainTryLurn():
	seq, dataId = readDataset('msrc')
	# seq, dataId = readDataset('shapes')
	# seq, dataId = readDataset('triangles')
	# seq, dataId = readDataset('sines')
	print seq.shape

	length = 12
	# k = 20
	k = -1
	maxDist = .2 * length
	# maxDist = 0.
	simMat = computeSimMat(seq, 8, maxDist, k)
	padding = np.zeros((simMat.shape[0], length-1))
	simMat = np.hstack((simMat, padding)) # padding
	showPairwiseSims(seq, length, simMat, plotMotifs=True, showEigenVect=False)

if __name__ == '__main__':
	mainTryLurn()
	# seq, dataId = readDataset('msrc')
	# # seq, dataId = readDataset('shapes')
	# # seq, dataId = readDataset('triangles')
	# # seq, dataId = readDataset('sines')
	# print seq.shape

	# length = 12
	# # k = 20
	# k = -1
	# maxDist = .2 * length
	# # maxDist = 0.
	# simMat = computeSimMat(seq, 8, maxDist, k)
	# padding = np.zeros((simMat.shape[0], length-1))
	# simMat = np.hstack((simMat, padding)) # padding
	# showPairwiseSims(seq, length, simMat, plotMotifs=True, showEigenVect=False)

