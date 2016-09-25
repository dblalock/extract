#!/usr/env/python

from collections import namedtuple

import sequence

# ================================================================
# Data Structures
# ================================================================

_fields = [
	'fromSeq',
	'startIdx',
	'endIdx',
	'label',
	'data'
]
PatternInstance = namedtuple('PatternInstance', _fields)

def createPatternInstance(startIdx, endIdx, label=0, fromSeq=0, data=None):
	return PatternInstance(fromSeq, startIdx, endIdx, label, data)


# ================================================================
# Functions
# ================================================================

def subseqsMatch(reported, truth, ignorePositions=False, ignoreLabel=False,
	minOverlapFraction=.5, requireContainment=False):

	seqId, ts, te, c, _ = reported
	seqId2, ts2, te2, c2, _ = truth
	gap2 = te2 - ts2

	if seqId != seqId2: # false if seqs different
		return False

	if (not ignoreLabel) and (c != c2): # false if classes different
		return False

	if ignorePositions: # if ignoring positions, seqId and class were everything
		if requireContainment:
			raise ValueError("cannot simultaneously ignore positions and"
				"require reported seqs to be contained in true seqs!")
		return True

	# sanity check start and end times
	if te < ts:
		print("Reported start index %d > end index %d", ts, te)
		assert(False)
		return
	if te2 < ts2:
		print("Ground truth start index %d > end index %d", ts2, te2)
		assert(False)
		return

	if requireContainment: # reported seq must be completely contained by true seq
		return ts >= ts2 and te <= te2

	# we just got a single time step as a ground truth annotation,
	# so return true if the reported seq includes this time step
	if gap2 == 0:
		return ts <= ts2 <= te

	# false if no overlap
	if te < ts2 or te2 < ts:
		return False

	overlapFraction = instancesIOU(reported, truth)
	return overlapFraction >= minOverlapFraction

	# return true iff both start and end are within (1 - minOverlapFraction)
	# of true times
	# maxDiff = gap2 * (1. - minOverlapFraction)
	# return abs(ts - ts2) < maxDiff and abs(te - te2) < maxDiff
	# print "subseqsMatch(): comparing boundaries of ", reported, truth
	# isMatch = abs(ts - ts2) < maxDiff and abs(te - te2) < maxDiff
	# print "match? ", isMatch
	# return isMatch


def intersectionSize(ts, te, ts2, te2):
	assert(te >= ts)
	assert(te2 >= ts2)
	if ts == te or ts2 == te2:
		return 0

	if te < ts2 and te2 < ts:
		return 0

	# ensure that ts <= ts2
	if ts > ts2:
		ts, te, ts2, te2 = ts2, te2, ts, te

	if te <= te2:
		# ------
		#   ------
		return te - ts2
	else:
		# --------
		#   -----
		return te2 - ts2


def instancesIntersectionSize(inst1, inst2):
	return intersectionSize(inst1.startIdx, inst1.endIdx,
		inst2.startIdx, inst2.endIdx)


def summedInstanceSizes(inst1, inst2):
	return (inst1.endIdx - inst1.startIdx) + (inst2.endIdx - inst2.startIdx)


def instancesIOU(inst1, inst2):
	interSize = float(instancesIntersectionSize(inst1, inst2))
	summedSize = summedInstanceSizes(inst1, inst2)
	unionSize = summedSize - interSize
	return interSize / unionSize


def matchIgnoringPositions(reported, truth, **kwargs):
	return subseqsMatch(reported, truth, ignorePositions=True, **kwargs)


def matchingSubseqs(reportedSeqs, trueSeqs, matchFunc=None, **matchFuncKwargs):
	"""
	Given the (seqId, start, end, class) tuples reported by a classifier and the
	true (seqId, start, end, class) tuples, compute the index of which of the
	true tuples each reported tuple corresponds to (-1 if none of them). Returns
	these values as a list l such that l[i] = j, where j is the index of the
	trueSeq which the ith reportedSeq matches.
	"""
	# make sure we have a func to test for matches
	matchFunc = matchFunc or subseqsMatch

	matches = []
	matchesSet = set()
	for i, repSeq in enumerate(reportedSeqs):
		matchIdx = -1
		for j, trueSeq in enumerate(trueSeqs):
			if j in matchesSet:
				continue
			if matchFunc(repSeq, trueSeq, **matchFuncKwargs):
				matchIdx = j
				matchesSet.add(j)
				break
		matches.append(matchIdx)

	return matches


def subseqIntersectionSizes(reportedSeqs, trueSeqs, matchFunc=None, ignoreLabel=False):
	"""return the size of the intersection (in time steps) of each reportedSeq
	with its best-matching trueSeq, ignoring duplicates"""

	# make sure we have a func to test for matches
	matchFunc = matchFunc or subseqsMatch

	# intersectionSizes = np.zeros(len(reportedSeqs))
	intersectionSizes = []
	matchesSet = set()
	for i, repSeq in enumerate(reportedSeqs):
		matchIdx = -1
		bestSize = 0
		for j, truSeq in enumerate(trueSeqs):
			if j in matchesSet:
				continue
			# check if these are from the same seq, same class, etc, but
			# with no requirement for how much they must overlap
			if matchFunc(repSeq, truSeq, minOverlapFraction=0.0001,
				ignoreLabel=ignoreLabel):
				sz = instancesIntersectionSize(repSeq, truSeq)
				if sz > bestSize:
					bestSize = sz
					matchIdx = j

		intersectionSizes.append(bestSize)
		matchesSet.add(matchIdx)

	return intersectionSizes


def computeNumMatches(reportedSeqs, trueSeqs, *args, **kwargs):
	matches = matchingSubseqs(reportedSeqs, trueSeqs, *args, **kwargs)
	numMatches = len(filter(lambda matchIdx: matchIdx >= 0, matches))
	return numMatches


def totalInstancesSize(insts):
	return sum([inst.endIdx - inst.startIdx for inst in insts])


def computeIOU(reportedSeqs, trueSeqs, ignoreLabel=False):
	intersectionSizes = subseqIntersectionSizes(reportedSeqs, trueSeqs,
		ignoreLabel=ignoreLabel)
	intersectionSize = sum(intersectionSizes)

	summedSize = totalInstancesSize(reportedSeqs) + totalInstancesSize(trueSeqs)
	unionSize = summedSize - intersectionSize

	return intersectionSize, unionSize, float(intersectionSize) / unionSize


def old_matchingSubseqs(reportedSeqs, trueSeqs, matchFunc=None):
	"""
	Given the (seqId, start, end, class) tuples reported by a classifier and the
	true (seqId, start, end, class) tuples, compute which of the true tuples
	each reported tuple corresponds to (-1 if none of them).

	seqId is a unique ID for each input sequence, start and end are indices
	within this sequence, and matchFunc is the function used to determine
	whether a reported and ground truth tuple match. Tuples are split by
	seqId, so matchFunc need only assess start and end indices and class. By
	default, matchFunc defaults to subseqsMatch (also in this file).

	Matches are assigned greedily from beginning to end, sorted by start index.

	Returns a dict: seqId -> idxs of matching truth tuple (or -1) for each
	reported tuple
	"""

	# make sure we have a func to test for matches
	matchFunc = matchFunc or subseqsMatch

	# group reported and true seqs by sequence id (in position 0)
	seq2reported = sequence.splitElementsBy(lambda tup: tup[0], reportedSeqs)
	seq2truth = sequence.splitElementsBy(lambda tup: tup[0], trueSeqs)

	matchesDict = {}
	for seqId, reported in seq2reported.iteritems():
		truth = seq2truth.get(seqId)
		if not truth: # ground truth has no instances in this sequence
			continue
		matches = []

		# sort by start time
		reported = sorted(reported, key=lambda x: x[1])
		truth = sorted(truth, key=lambda x: x[1])

		for i, repSeq in enumerate(reported):
			matches.append(-1)
			for j, trueSeq in enumerate(truth):
				if matchFunc(repSeq, trueSeq):
					matches[i] = j
					del truth[j] # can't match the same thing twice
					break

		matchesDict[seqId] = matches

	return matchesDict


def old_numMatchingSeqs(matchesDict):
	numMatches = 0
	for k, idxs in matchesDict.iteritems():
		validIdxs = filter(lambda idx: idx >= 0, idxs)
		numMatches += len(validIdxs)
	return numMatches


# def classInstanceCountsInSeqs(reportedSeqs, trueSeqs, **sink):
# 	# group reported and true seqs by sequence id (in position 0)
# 	seq2reported = sequence.splitElementsBy(lambda tup: tup[0], reportedSeqs)
# 	seq2truth = sequence.splitElementsBy(lambda tup: tup[0], trueSeqs)

# 	seqId2ClassInstanceCounts = {}
# 	for seqId, reported in seq2reported.iteritems():
# 		truth = seq2truth.get(seqId)

# 		# create dict: class label -> sequences
# 		class2elements_reported = sequence.splitElementsBy(lambda seq: seq.label, reported)
# 		class2elements_truth = sequence.splitElementsBy(lambda seq: seq.label, truth)

# 		# dict: class label -> # instances
# 		classCounts_reported = sequence.applyToDict(lambda k, v: len(v), class2elements_reported)
# 		classCounts_truth = sequence.applyToDict(lambda k, v: len(v), class2elements_truth)

# 		seqId2ClassInstanceCounts[seqId] = (classCounts_reported, classCounts_truth)

# 	return seqId2ClassInstanceCounts

# def classStatsForSeqs(reportedClassCountsDict, truthClassCountsDict):

# 	reportedClasses = reportedClassCountsDict.keys()
# 	truthClasses = reportedClassCountsDict.keys()
# 	allClasses = sequence.uniqueElements(reportedClasses + truthClasses)


def precisionAndRecall(numReportedSeqs, numTrueSeqs, numMatchingSeqs):
	if (not numReportedSeqs) or (not numTrueSeqs):
		return 0., 0.

	prec = float(numMatchingSeqs) / numReportedSeqs
	rec = float(numMatchingSeqs) / numTrueSeqs
	return prec, rec


def f1Score(precision, recall):
	if (not precision) or (not recall):
		return 0.
	return 2. * precision * recall / (precision + recall)


def precisionRecallF1(numReported, numTrue, numMatches):
	prec, rec = precisionAndRecall(numReported, numTrue, numMatches)
	return prec, rec, f1Score(prec, rec)


# TODO refactor so less dup code with subseqMatchStats (and so returnMoreStats
# is documented)
def subseqIOUStats(reportedSeqs, trueSeqs, matchUpLabels=False, returnMoreStats=False):

	if matchUpLabels: # might find pattern, but not know what to label it
		lbl2reported = sequence.splitElementsBy(lambda inst: inst.label, reportedSeqs)
		lbl2truth = sequence.splitElementsBy(lambda inst: inst.label, trueSeqs)

		intersectionSize = 0.
		unionSize = 0.
		reportedSize = 0.
		truthSize = 0.
		for repLbl, repSeqs in lbl2reported.iteritems():
			bestIntersection = 0
			bestUnion = 0
			bestReportedSize = 0
			bestTruthSize = 0
			bestIOU = 0.
			for truthLbl, truthSeqs in lbl2truth.iteritems():
				interSz, unionSz, iou = computeIOU(repSeqs, truthSeqs,
					ignoreLabel=True)
				if iou >= bestIOU: # so that ties for 0 will replace stuff
					bestIntersection = interSz
					bestUnion = unionSz
					bestReportedSize = totalInstancesSize(repSeqs)
					bestTruthSize = totalInstancesSize(truthSeqs)
					bestIOU = iou
			intersectionSize += bestIntersection
			unionSize += bestUnion
			reportedSize += bestReportedSize
			truthSize += bestTruthSize
		iou = float(intersectionSize) / unionSize
	else:
		intersectionSize, unionSize, iou = computeIOU(reportedSeqs, trueSeqs,
			ignoreLabel=False)
		reportedSize = totalInstancesSize(reportedSeqs)
		truthSize = totalInstancesSize(trueSeqs)

	if returnMoreStats:
		return intersectionSize, unionSize, iou, reportedSize, truthSize

	return intersectionSize, unionSize, iou


def subseqMatchStats(reportedSeqs, trueSeqs, matchFunc=None,
	spoofNumReported=-1, spoofNumTrue=-1, ignorePositions=False,
	matchUpLabels=False, matchAllClasses=False, minOverlapFraction=.5,
	requireContainment=False, **sink):
	"""[PatternInstance] x [PatternInstance] -> numMatches, numReported, numTrue"""

	if (not matchFunc) and ignorePositions:
		matchFunc = matchIgnoringPositions

	if matchUpLabels: # might find pattern, but not know what to label it
		lbl2reported = sequence.splitElementsBy(lambda inst: inst.label, reportedSeqs)
		lbl2truth = sequence.splitElementsBy(lambda inst: inst.label, trueSeqs)

		# print "subseqMatchStats: matching up with {} true labels".format(len(lbl2truth))
		# print "subseqMatchStats: matching up {} reported with {} actual".format(
		# 	len(reportedSeqs), len(trueSeqs))

		# XXX if we report more than one label, there isn't necessarily a
		# bijective mapping between reported and ground truth labels; it's
		# possible to get more matches than true labels here, among other issues
		numMatches = 0 # or avg overlap fraction (IOU)
		numTrue = 0
		for repLbl, repSeqs in lbl2reported.iteritems():
			bestNumMatches = -1 # can't be 0 or numTrue stays unset if no matches
			bestNumTruth = 0
			for truthLbl, truthSeqs in lbl2truth.iteritems():
				if len(truthSeqs) < 2: # ignore patterns that only happen once
					continue
				numMatchesForLabel = computeNumMatches(repSeqs, truthSeqs, matchFunc,
					ignoreLabel=True, minOverlapFraction=minOverlapFraction,
					requireContainment=requireContainment)
				if numMatchesForLabel > bestNumMatches:
					bestNumMatches = numMatchesForLabel
					bestNumTruth = len(truthSeqs)
			numMatches += max(0, bestNumMatches)
			minNumTruth = min([len(truthSeqs) for _, truthSeqs in lbl2truth.iteritems()])
			minNumTruth = max(minNumTruth, 2)
			numTrue += max(minNumTruth, bestNumTruth)
	else:
		numMatches = computeNumMatches(reportedSeqs, trueSeqs, matchFunc,
			minOverlapFraction=minOverlapFraction,
			requireContainment=requireContainment)

	if matchAllClasses:
		numTrue = len(trueSeqs)

	numReported = spoofNumReported if spoofNumReported >= 0 else len(reportedSeqs)
	numTrue = spoofNumTrue if spoofNumTrue >= 0 else numTrue

	return numReported, numTrue, numMatches


def scoreSubseqs(reportedSeqs, trueSeqs, **kwargs):
	"""Returns (precision, recall, f1 score) based on matching via matchFunc"""

	numMatches, numReported, numTrue = subseqMatchStats(reportedSeqs, trueSeqs, **kwargs)

	prec, rec = precisionAndRecall(numReported, numTrue, numMatches)
	return prec, rec, f1Score(prec, rec)

# ================================================================
# Main
# ================================================================

if __name__ == '__main__':
	import doctest
	doctest.testmod()


