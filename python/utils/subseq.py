#!/usr/env/python

import numpy as np

import sliding_window as window
import kdtree as kd
from sequence import isListOrTuple, asListOrTuple, flattenListOfLists
from arrays import colsAsList, zNormalizeRows, normalizeRows, nonzeroRows
from arrays import isScalar, meanNormalizeRows
from files import ensureDirExists
from misc import randStrId

'''
def subseqDistsFFT(x, y):
	"""Find the L2^2 distances between y and every subseq of x;
	Untested port of data dicts code
	"""
	y = (y - np.mean(y)) / np.std(y)
	n = len(x)
	m = len(y)

	# get dot products via convolution theorem
	# x_pad = np.r_[x, np.zeros(n)]
	x_pad = np.r_[x, np.zeros(m)]
	y_pad = y[::-1]
	# y_pad = np.r_[y_pad, np.zeros(2*n - m)]
	y_pad = np.r_[y_pad, np.zeros(n)]
	X = fft(x_pad)
	Y = fft(y_pad)
	Z = X * Y
	z = ifft(Z)
	x_dot_y = z[m-1:n] # only middle part of convolution

	# y stats
	sum_y = np.sum(y)
	sum_y2 = np.sum(y * y)

	# x stats
	sum_x = np.cumsum(x)
	sum_x2 = np.cumsum(x * x)
	sum_x[m:n] -= sum_x[0:n-m]
	sum_x2[m:n] -= sum_x2[0:n-m]
	sum_x = sum_x[m-1:n]
	sum_x2 = sum_x2[m-1:n]
	mean_x = sum_x / m
	mean_x_sq = mean_x*mean_x
	sig_x_sq = sum_x2 / m - mean_x_sq
	sig_x = np.sqrt(sig_x_sq)

	# distance
	term1 = (sum_x2 - 2*sum_x * mean_x + m*mean_x_sq) / sig_x_sq
	term2 = -2*(x_dot_y - sum_y * mean_x) / sig_x
	return term1 + term2 + sum_y2
'''

def subseqDists(x, y):
	"""find the L2^2 distances between y and every subseq of x"""
	y = y.flatten()
	y = (y - np.mean(y)) / np.std(y)

	# flatten Nd input seqs
	origDims = len(x.shape)
	stride = origDims # TODO allow stepping in more than one direction
	x = x.flatten()

	subseqs = window.sliding_window(x, len(y), stride)
	subseqs = zNormalizeRows(subseqs)

	return distsToRows(subseqs, y)

def nearestSubseqStartIdx(x, y):
	dists = subseqDists(x, y)
	return np.argmin(dists)

def distsToRows(X, y):
	diffs = X - y
	return np.sum(diffs*diffs, axis=1)


# ================================================================
# ================================================================
# Begin garbagey experimentation code
# ================================================================
# ================================================================

def uniqueRowIdxs(Xnorm, maxDist):
	# minSim = zNormDistToCosSim(maxDist, Xnorm.shape[1])
	minSim = zNormDistsToCosSims(maxDist, 1) # assume dist normalized by length
	assert(0. <= minSim <= 1.)

	keepIdxs = []
	for i, query in Xnorm:
		sims = np.dot(Xnorm[keepIdxs], query)
		assert(np.max(sims) <= 1.)
		if np.max(sims) >= minSim:
			continue
		keepIdxs.append(i)

	return keepIdxs

def allZNormalizedSubseqs(seqs, length):
	X, _, _ = window.flattened_subseqs_of_length(seqs, length, norm='each')
	return zNormalizeRows(X, removeZeros=False)

def uniqueSubseqsInSignals(signal, length, maxDist, norm='each', tree=None):
	X, _, _ = window.flattened_subseqs_of_length(signal, length, norm=norm)
	Xnorm = zNormalizeRows(X, removeZeros=False)

	# print("subseqsInSignals: signal has %d subseqs" % (len(Xnorm)))

	# init kd tree--we can't give it any data yet because we only want to
	# search through seqs that have been added to the dictionary
	if tree is None:
		width = Xnorm.shape[1]
		tree = kd.create(dimensions=width)

	signalOccurIdxs = {}
	tree.add(Xnorm[0], 0)
	for startIdx, subseq in enumerate(Xnorm[1:]):
		if np.sum(subseq*subseq) < .001: # ignore zero seqs
			continue

		startIdx += 1 # since we skipped Xnorm[0]
		neighbors = tree.search_knn(subseq, 2)
		neighborIdx = -1
		neighborDist = np.inf
		# pull out whichever neighbor isn't the query
		for node, dist in neighbors:
			idx = node.metadata
			if idx != startIdx:
				neighborIdx = idx
				neighborDist = dist
		if neighborIdx < 0:
			print "ERROR: knn returned <2 neighbors..."
			print "Neighbors returned:", neighbors
			assert(0)

		# print "neighborDist", neighborDist, maxDist
		if neighborDist < maxDist:
			# store that the subseq happened at this idx too
			l = signalOccurIdxs.get(neighborIdx, [])
			l.append(startIdx)
			# signalOccurIdxs[neighborIdx] = l
		else:
			# ah, so this can overwrite crap and yield too few features
			signalOccurIdxs[startIdx] = [startIdx]
			tree.add(subseq, startIdx)

		# rebalance if startIdx is a power of 2, so we do so log(N) times
		if 2**int(np.log2(startIdx)) == startIdx:
			# print "rebalancing at start idx %d" % (startIdx,)
			tree.rebalance()

		# signalOccurIdxs[neighborIdx] = [startIdx]
		# if res:
		# 	nn, dist = res
		# 	if dist <= maxDist:
		# 		# store that the subseq happened at this idx too
		# 		neighborID = nn.metadata
		# 		signalOccurIdxs[neighborID].append(startIdx)
		# 		continue
		# neighborID = startIdx
		# signalOccurIdxs[neighborID] = [startIdx]
		# tree.add(subseq, neighborID)

	return signalOccurIdxs, Xnorm # return Xnorm for convenience, although confusing...


def uniqueSubseqs(seqs, length, maxDist, tieDims=False):
	"""return a set of subseqs such that no subseq is within maxDist of any
	other subseq and all original subseqs are within maxDist of one of the
	subseqs returned; basically a greedy unique operator where uniqueness
	is defined by L2^2 distance within maxDist"""

	seqs = asListOrTuple(seqs)
	is1D = len(seqs[0].shape) == 1

	if tieDims or is1D:
		# zip trick so that we always return two lists (occurIdxs and Xnorms)
		return zip(*[uniqueSubseqsInSignals(seqs, length, maxDist, norm='each_mean')])
	else:
		nDims = seqs[0].shape[1]
		occurIdxs = []
		Xnorms = []
		for dim in range(nDims): # for each dimension
			print("finding unique seqs in dim %d" % (dim,))
			signals = map(lambda seq: seq[:, dim], seqs)
			signalOccurIdxs, Xnorm = uniqueSubseqsInSignals(signals, length, maxDist, norm='each_mean')
			print("found %d unique seqs" % (len(signalOccurIdxs),))
			occurIdxs.append(signalOccurIdxs)
			Xnorms.append(Xnorm)

		return occurIdxs, Xnorms

def repeatedSubseqOccurIdxs(seqs, length, maxDist, tieDims=False):
	# TODO dont just mash everything together
	# X = np.vstack(seqs)
	# nSamples, nDims = X.shape
	occurIdxsForDims, _ = uniqueSubseqs(seqs, length, maxDist, tieDims)

	# print occurIdxsForDims[0]
	# print occurIdxsForDims[1]
	# print occurIdxsForDims[2]

	# repeatedSubseqsForDims = []
	repeatedOccurIdxsForDims = []
	for i in range(len(occurIdxsForDims)): # for each (effective) dimension
		idx2positions = occurIdxsForDims[i]
		idx2positions_repeated = {}
		idxs_repeated = []
		for idx, positions in idx2positions.iteritems():
			if len(positions) > 1:
				idx2positions_repeated[idx] = positions
				idxs_repeated.append(idx)
		idxs_repeated = sorted(idxs_repeated)
		# repeatedSubseqs = seq[idxs_repeated]

		# store which subseqs were repeated and where they happened
		# repeatedSubseqsForDims.append(repeatedSubseqs)
		repeatedOccurIdxsForDims.append(idx2positions_repeated)

	# print repeatedOccurIdxsForDims[0]
	# print repeatedOccurIdxsForDims[1]
	# print repeatedOccurIdxsForDims[2]

	return repeatedOccurIdxsForDims

def simMatFromDistTensor(Dtensor, length, padLen, clamp=True, pruneCorrAbove=.9):
	imgMat = Dtensor.reshape((-1, Dtensor.shape[2]))

	if padLen:
		distsNoPad = imgMat[:,:-padLen]
		# similarities = zNormDistsToCosSims(distsNoPad, length)
		similarities = zNormDistsToSims(distsNoPad, length)
		imgMat[:,:-padLen] = similarities
		endCol = imgMat.shape[1]
		imgMat[:,(endCol-padLen):endCol] = 0
	else:
		distsNoPad = imgMat
		# similarities = zNormDistsToCosSims(distsNoPad, length)
		similarities = zNormDistsToSims(distsNoPad, length)
		imgMat = similarities

	if clamp:
		imgMat = np.maximum(0, imgMat)

	if pruneCorrAbove > 0.:
		imgMat = removeCorrelatedRows(imgMat, pruneCorrAbove)

	return imgMat

def similarityMat(seqs, length, clamp=True, pruneCorrAbove=-1, **kwargs):
	Dtensor, _ = pairwiseDists(seqs, length, **kwargs)
	# padLen = (length - 1) if kwargs.get('padLen') else 0
	padLen = length - 1

	return simMatFromDistTensor(Dtensor, length, padLen,
		clamp=clamp, pruneCorrAbove=pruneCorrAbove)

# def pairwiseDists(seqs, length, norm='each_mean', tieDims=False, pad=True,
def pairwiseDists(seqs, length, norm='each', tieDims=False, pad=True,
	removeZeros=True, k=-1):

	seqs = asListOrTuple(seqs)
	nDims = 1
	if len(seqs[0].shape) < 2 or tieDims:
		Xnorm, _, _ = window.flattened_subseqs_of_length(seqs, length, norm=norm)
	else:
		nDims = seqs[0].shape[1]
		# bypass flattening--each dim of each seq is treated as a separate
		# 1D seq; we end up with a long list whose elements are 1D vectors,
		# each of which was originally a column within some ND array in seqs
		#
		# note that this may do weird things if there's more than one seq
		# because the dims for each seq are sequential, rather than the seqs
		# for each dim
		separatedByDim = map(lambda X: colsAsList(X), seqs)
		flatSeqs = flattenListOfLists(separatedByDim)
		flatSeqs = map(lambda v: v.flatten(), flatSeqs) # col vects -> 1D arrays
		Xnorm, _, _ = window.flattened_subseqs_of_length(flatSeqs, length, norm='each')

	nSamples, m = Xnorm.shape
	rowsPerDim = nSamples / nDims
	print "----- pairwiseDists"
	print "length", length
	print "origSeqs[0] shape", seqs[0].shape
	print "nsamples, m, rowsPerDim", Xnorm.shape, rowsPerDim
	print "-----"

	if pad:
		paddingLen = length - 1
	else:
		paddingLen = 0

	# print "Xnorm stats:", np.mean(Xnorm, axis=1), np.std(Xnorm, axis=1)

	# D = np.zeros((nSamples, nSamples+paddingLen*nDims)) # 0 pad at end so samples line up
	Dtensor = np.zeros((nDims, rowsPerDim, rowsPerDim+paddingLen))
	# D = np.zeros((nSamples, nSamples))

	maxPossibleDist = 2**2 * m
	maxIdx = 0
	for dim in range(nDims):
		# extract subseqs associated with this dim
		minIdx = maxIdx
		maxIdx += rowsPerDim
		Xdim = Xnorm[minIdx:maxIdx]
		# compute dists to each one
		for i, row in enumerate(Xdim):
			if removeZeros:
				if np.sum(row*row) < 1.e-6:
					Dtensor[dim, i, :rowsPerDim] = maxPossibleDist
					continue

			diffs = Xdim - row
			diffs_sq = diffs * diffs
			# dMinIdx = minIdx + dim*paddingLen
			# dMaxIdx = dMinIdx + rowsPerDim
			dists = np.sum(diffs_sq, axis=1)

			# D[minIdx + i, dMinIdx:dMaxIdx] = dists
			Dtensor[dim, i,:rowsPerDim] = dists
		# only keep k lowest dists
		if k > 0:
			for j in np.arange(rowsPerDim):
				col = Dtensor[dim, :, j]
				highestIdxs = np.argsort(col)[k:]
				Dtensor[dim, highestIdxs, j] = maxPossibleDist

	# return Dtensor, D, Xnorm
	return Dtensor, Xnorm

def zNormDistsToSims(dists, seqLength): # not cosine sims, just useful similarity
	return 1. - np.sqrt(np.asarray(dists) / seqLength) # assumes L2^2 dist

def zNormDistsToCosSims(dists, seqLength):
	# return 1. - np.sqrt(np.asarray(dists) / seqLength) # wrong
	return 1. - np.asarray(dists) / (2*seqLength) # right (assuming L2^2 dist)

# @jit
def removeCorrelatedRows(X, thresh, accumulate=False):
	# for some reason, this occasionally removes all rows but the first,
	# even when it clearly shouldn't
	X = X[nonzeroRows(X)]
	Xnorm = meanNormalizeRows(X)

	Xnorm = normalizeRows(Xnorm)
	# print "mean row norm: ", np.mean(np.linalg.norm(Xnorm, axis=1)) # exactly 1.0...looks good
	keepIdxs = np.array([0])
	multipliers = np.array([1])
	for i, row in enumerate(Xnorm[1:]):
		Xkeep = Xnorm[keepIdxs]
		sims = np.dot(Xkeep, row)
		if np.max(sims) >= 1.0001:
			print "wtf, max too high!"
			print "sims", sims
			print "max", np.max(sims)
			assert(0)
		if np.min(sims) <= -1.0001:
			print "wtf, min too low!"
			print "sims", sims
			print "min", np.min(sims)
			assert(0)
		if np.all(sims < thresh):
			keepIdxs = np.r_[keepIdxs, i]
			multipliers = np.r_[multipliers, 1]
		elif accumulate:
			bestMatchIdx = np.argmax(sims)
			multipliers[bestMatchIdx] += 1 # weight by num times it happens

		# elif len(sims) == 1:
			# print i, sims.shape, np.max(sims)
			# print np.std(row), np.mean(row)

	return X[keepIdxs] * multipliers.reshape((-1,1))
	# return X[keepIdxs]

def computeFromSeq(seqs, subseqLength=1):
	"""Given a collection of sequences, returns an array of which seq an
	element i came from in the concatenation of all the seqs

	>>> s = [[1,2],[3,4,5]]
	>>> computeFromSeq(s)
	array([0, 0, 1, 1, 1])
	>>> computeFromSeq(s, subseqLength=2)
	array([0, 1, 1])
	>>> computeFromSeq([1,2])
	array([0, 0])
	"""
	# just one seq -> array of all 0s
	if isScalar(seqs[0]):
		return np.zeros(len(seqs), dtype=np.int)
	if len(seqs) == 1:
		return np.zeros(len(seqs[0]), dtype=np.int)

	seqLens = np.array(map(lambda seq: len(seq) - subseqLength + 1, seqs))
	cumLen = np.cumsum(seqLens)
	combinedLength = cumLen[-1]
	startIdxs = np.r_[0, cumLen[:-1]]
	endIdxs = np.r_[startIdxs[1:], combinedLength]
	fromSeq = np.zeros(combinedLength, dtype=np.int)
	for i in range(len(startIdxs)):
		startIdx, endIdx = startIdxs[i], endIdxs[i]
		fromSeq[startIdx:endIdx] = i
	return fromSeq


def optimalAlignK(scores, m, k):
	"""
	Given an array of scores, return the indices I of the k best scores such
	that for all i, j in I, i !=j -> |i - m| >= m; in other words, the indices
	must be m apart

	Parameters
	----------
	scores: 1D, array-like
		an ordered collection of scores
	m: int
		minimum spacing between reported indices
	k: int or array-like of int
		number of indices to return

	Returns
	-------
	idxs: an array of indices for each k value specified (a single array
		or a list thereof, depending on whether k is an int or a collection)

	>>> s = [2,1,4,3]
	>>> optimalAlignK(s, 2, 1)
	array(2)
	>>> optimalAlignK(s, 2, 2)
	array([0, 2])
	>>> optimalAlignK(s, 3, 2)
	array([0, 3])
	>>> optimalAlignK(s, 4, 2)
	array(2)
	>>> optimalAlignK(s, 2, [1, 2])
	[array([2]), array([0, 2])]
	>>> s2 = [2,1,4,3,1,7,1]
	>>> optimalAlignK(s2, 2, [2, 3])
	[array([2, 5]), array([0, 2, 5])]
	>>> optimalAlignK(s2, 3, [1, 2, 3])
	[array([5]), array([2, 5]), array([0, 3, 6])]
	>>> s3 = [2,1,4,3,1,7,-99]
	>>> optimalAlignK(s3, 3, [3])
	[]
	"""
	# ------------------------ err handling and arg munging
	if scores is None or not len(scores):
		raise RuntimeError("No scores given!")
	if k is None:
		raise RuntimeError("Number of locations to return must be >= 1")
	if isScalar(k):
		k = (k,)
	k = np.sort(np.asarray(k))
	kmax = np.max(k)
	if kmax < 1:
		raise RuntimeError("Number of locations to return must be >= 1")

	n = len(scores)
	if n <= m or k[-1] == 1:
		return np.array(np.argmax(scores), dtype=np.int)

	scores = np.asarray(scores)
	if np.all(scores <= 0.):
		print("Warning: optimalAlignK(): all scores <= 0")
		return [[] for kk in k]

	# ------------------------ find best score and parent for each k at each idx

	# initialize first m points
	historyShape = (len(scores), kmax)
	c = np.zeros(historyShape) - 1							# cumulative score
	c[:m, 0] = scores[:m]
	parentIdxs = np.zeros(historyShape, dtype=np.int) - 1 	# previous best idx

	# compute scores and parent idxs
	bestScores = np.zeros(kmax)
	bestIdxs = np.zeros(kmax) - 1
	for i in range(m, n):
		oldIdx = i - m
		betterForTheseK = c[oldIdx] > bestScores
		# print i, bestScores, bestIdxs
		if np.any(betterForTheseK): # check not really needed; will just do nothing
			bestScores[betterForTheseK] = c[oldIdx, betterForTheseK]
			bestIdxs[betterForTheseK] = oldIdx
		parentIdxs[i, 1:] = bestIdxs[:-1]
		c[i, 1:] = bestScores[:-1] + scores[i]
		c[i, 1:] *= parentIdxs[i, 1:] >= 0 # only valid parents
		c[i, 0] = scores[i] # TODO? seemingly no point if < bestScores[0]

	# print np.c_[np.arange(n), scores, c]
	# print np.c_[np.arange(n), scores, parentIdxs]

	# compute best set of idxs for each value of k
	allParents = []
	for kk in k:
		kIdx = kk - 1
		parents = []
		parent = np.argmax(c[:, kIdx])
		if c[parent, kIdx] <= 0.:
			allParents.append([])
			continue
		while parent >= 0:
			parents.append(parent)
			parent = parentIdxs[parent, kIdx]
			kIdx -= 1
		parents = np.array(parents, dtype=np.int)[::-1]

		allParents.append(parents)

	if len(k) == 1:
		return allParents[0]
	return allParents

def optimalAlignment(scores, m, scoresForEndIdxs=True):
	"""
	Given an array of scores for the end positions of subsequences of length m
	(where higher scores correspond to better alignments and negative scores
	indicate locations that should be "skipped"), returns the end indices of
	the optimal placement of subsequences.

	Parameters
	------
	scores: a vector of scores
	m: length of the previous values with which overlap is disallowed

	>>> s = [2,1,3,4]
	>>> optimalAlignment(s, 2)
	array([0, 3])
	>>> s = [1,2,3,2,5,1]
	>>> optimalAlignment(s, 2)
	array([0, 2, 4])
	>>> s = [-1,2,3,2,5,1]
	>>> optimalAlignment(s, 2)
	array([2, 4])
	>>> optimalAlignment(s, 2, scoresForEndIdxs=False)
	array([2, 4])
	"""
	if scores is None or not len(scores):
		return None

	scores = np.asarray(scores)
	n = len(scores)

	if n <= m:
		return np.argmax(scores)

	# if the scores are associate with start indices, rather than end
	# indices, we need to ensure things don't overlap with positions after
	# them, not positions before them; the easiest way to do this is just
	# to reverse the order of everything
	#
	# TODO almost positive we can remove this param--algo guarantees that
	# scores taken have gaps of at least m-1 between them; doesn't care
	# about not overlapping before vs after
	if not scoresForEndIdxs:
		scores = scores[::-1]

	c = np.empty(scores.shape) 							# cumulative score
	c[:m] = scores[:m]
	parentIdxs = np.empty(scores.shape, dtype=np.int)	# previous best idx
	parentIdxs[:m] = -1

	idx_bsf = -1	# best-so-far
	score_bsf = 0.

	# compute best parent at least m time steps ago for every idx;
	# a "parent" is a previous cumulative score we can add to
	for i in range(m, n):
		oldIdx = i - m
		if c[oldIdx] > score_bsf:
			idx_bsf = oldIdx
			score_bsf = c[oldIdx]
		c[i] = scores[i] + score_bsf
		parentIdxs[i] = idx_bsf

	# compute lineage of best score
	parent = np.argmax(c)
	if scores[parent] < 0.: # edge case: all scores are negative
		return None
	parents = []
	while parent >= 0:
		parents.append(parent)
		parent = parentIdxs[parent]

	parents = np.array(parents, dtype=np.int)
	if scoresForEndIdxs:
		return parents[::-1] # appended in reverse order above
	else:
		return (n-1) - parents # we reversed the order at the start


def tryDataset(dataset, save=True):
	from viz.motifs import showPairwiseDists
	from ..datasets import synthetic as synth
	from ..datasets import read_msrc as msrc
	from ..utils.arrays import downsampleMat

	saveDir = "figs/subseq/similarities/"
	saveDir += dataset + '/'
	ensureDirExists(saveDir)

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

	# tieDims = True # yields garbage with basically all super-low or negative similarities
	tieDims = False
	m = len(seq) / 20
	print "Using subseq length %d" % m
	# k = [2, 10, 50]
	k = 20
	norm = 'each'
	maxCorr = .9

	if not isListOrTuple(k):
		k = [k]
	for kval in k:
		filename = saveDir + 'm=%d' % m
		if kval > 0:
			filename += '_k=%d' % kval
		filename += '_norm=' + norm
		filename += '_maxCorr=%g' % maxCorr
		if tieDims:
			filename += '_tiedDims'
		filename += '_' + dataId
		filename += '.png'

		Dtensor, Xnorm = pairwiseDists(seq, m, norm=norm, tieDims=tieDims, k=kval)
		print "Dtensor shape:", Dtensor.shape
		print "Xnorm shape:", Xnorm.shape
		filename = filename if save else None
		showPairwiseDists(seq, m, Dtensor, pruneCorrAbove=maxCorr, saveas=filename)
		# showPairwiseDists(seq, Dtensor, pruneCorrAbove=maxCorr, saveas=None)


if __name__ == '__main__':
	import doctest
	doctest.testmod()

	# s2 = [2,1,4,3,1,7,1]
	# optimalAlignK(s2, 2, [2, 3])
	# [array([2, 5]), array([0, 2, 5])]
	# optimalAlignK(s2, 3, [3])
	# [array([5]), array([2, 5]), array([0, 3, 6])]

	# s2 = [2,1,4,3,1,7]
	# optimalAlignK(s2, 2, [2, 3])
	# [array([2, 5]), array([0, 2, 5])]

	# s = [2,1,4,3]
	# optimalAlignK(s, 2, 2)
	# array([0, 2])

	# s = [2,1,4,3]
	# optimalAlignK(s, 2, [1, 2])
	# [array(2), array([0, 2])]

	# import matplotlib.pyplot as plt

	# from ..datasets import synthetic as synth
	# from ..datasets import read_msrc as msrc
	# from .arrays import downsampleMat

	# dataset = 'msrc'
	# # dataset = 'shapes'
	# # dataset = 'triangles'
	# # dataset = 'sines'

	# # for i in range(100):
	# # for i in range(5):
	# for i in range(1):
	# 	# tryDataset(dataset)
	# 	tryDataset(dataset, save=False)

	# # showPairwiseDists(seq, D, Dtensor, useTensor=True, saveas=filename)

	# ================================================================
	# dMax = .1 * m # TODO remove need for this parameter
	# # ^ to do so, prolly put all subseqs in the tree (since this will make the
	# # tree more balanced anyway) and examine distro of distances; note that
	# # we'll have to remove crap from the tree or something afterwards though
	# if tieDims:
	# 	dMax *= seq.shape[1]

	# repeatedOccurIdxsForDims = repeatedSubseqOccurIdxs(seq, m, dMax, tieDims=tieDims)

	# print repeatedOccurIdxsForDims
	# import sys
	# sys.exit()

	# for dim in range(nDims):
	# 	idx2positions = repeatedOccurIdxsForDims[dim]

	# for subseqs in repeatedSubseqsForDims:
	# 	plt.figure()
	# 	ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
	# 	ax1.plot(seq)
	# 	ax2 = plt.subplot2grid((2,2), (1,0), colspan=2)
	# 	ax2.plot(subseqs)
	# plt.show()

	# allOccur = np.zeros((1, seqLen))

	# # for dim in range(nDims):
	# for dim in range(len(repeatedOccurIdxsForDims)):
	# 	idx2positions = repeatedOccurIdxsForDims[dim]
	# 	nSeqs = len(idx2positions)
	# 	occur = np.zeros((nSeqs, seqLen))
	# 	seqNum = 0
	# 	for idx, positions in idx2positions.iteritems():
	# 		occur[seqNum, positions] = 1
	# 		seqNum += 1
	# 	allOccur = np.vstack((allOccur, occur))
	# 	padding = np.zeros((3, seqLen))
	# 	allOccur = np.vstack((allOccur, padding))
	# 	# plt.figure()
	# 	# plt.imshow(occur)

	# allOccur = zNormalizeRows(allOccur)
	# plt.imshow(allOccur)
	# plt.colorbar()
	# plt.show()
	# ================================================================


