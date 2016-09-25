#!/usr/env/python

import os
import numpy as np

from ..datasets.datasets import DataLoader
from ..utils import learn
from ..utils.files import ensureDirExists
from ..utils.misc import nowAsString

from estimators import * # TODO don't do this

# ================================================================ Synthetic

def results_synthetic(saveDir):
	datasetParams = [{'datasetName': ('triangles', 'rects', 'sines', 'shapes')}]
	# datasetParams = [{'datasetName': ['triangles'], 'seed': [1234]}]
	# datasetParams = [{'datasetName': ['shapes'], 'seed': np.arange(20)}]
	# datasetParams = [{'datasetName': ['triangles']}]

	extractorParams = [
		# # just 2 instances
		# {'lengths': [25],
		# 'downsampleBy': [0, 2],
		# 'onlyReturnPair': [True],
		# 'saveMotifsDir': [saveDir]
		# },
		# # all instances, no added data
		# {'lengths': [25],
		# 'downsampleBy': [0, 2],
		# 'onlyReturnPair': [False],
		# 'threshAlgo': [None, 'minnen', 'maxsep'],
		# 'saveMotifsDir': [saveDir]
		# },
		# # all instances, added data
		# {'lengths': [25],
		# 'downsampleBy': [0, 2],
		# 'onlyReturnPair': [False],
		# 'threshAlgo': ['minnen', 'maxsep'],
		# 'addData': ['gauss', 'randwalk', 'freqMag'],
		# 'addDataFractionOfLength': [2, 5.],
		# 'saveMotifsDir': [saveDir]
		# }
		# all instances, mdl
		{'lengths': [np.arange(1./20, 1./3, .005)],
		# {'lengths': [.2],
		'downsampleBy': [0],
		'onlyReturnPair': [False],
		'threshAlgo': ['mdl'],
		'mdlBits': [4, 5, 6, 7, 8, 9, 10],
		# 'mdlBits': [8],
		'maxOverlapFraction': [0, .5],
		'saveMotifsDir': [saveDir],
		# 'showMotifs': [False],
		},
	]

	dataLoading = [DataLoader(), datasetParams]
	motifFinding = (MotifExtractor(), extractorParams)

	d = [("LoadData", [dataLoading]),
		("FindMotif", [motifFinding])
	]

	df = learn.tryParams(d, None, None, crossValidate=False)
	return df

# ================================================================ MSRC

MSRC_PARAMS = [{'datasetName': ['msrc'],
		# 'whichExamples':[[1,2]],
		# 'whichExamples':[[2]],
		# 'whichExamples':[[37, 52, 82]], # does super well on these
		'whichExamples':[range(2, 100, 5)]
}]

def results_msrc(saveDir):
	if saveDir:
		saveDir = os.path.join(saveDir, 'sota')

	# extractorParams = [{'lengths': [25],
	extractorParams = [
		# # just 2 instances
		# {'lengths': [np.arange(1./20, 1./8, .005)],
		# 'downsampleBy': [5],
		# 'onlyReturnPair': [True],
		# 'saveMotifsDir': [saveDir]
		# },
		# # all instances, no added data
		# {'lengths': [np.arange(1./20, 1./8, .005)],
		# 'downsampleBy': [5],
		# 'onlyReturnPair': [False],
		# 'threshAlgo': [None, 'minnen', 'maxsep'],
		# 'maxOverlapFraction': [0, .5],
		# 'ignorePositions': [True],
		# 'saveMotifsDir': [saveDir]
		# },
		# # all instances, added data
		# {'lengths': [np.arange(1./20, 1./8, .005)],
		# 'downsampleBy': [5],
		# 'onlyReturnPair': [False],
		# 'threshAlgo': ['minnen', 'maxsep'],
		# 'addData': ['gauss', 'randwalk', 'freqMag'],
		# 'addDataFractionOfLength': [2, 5.],
		# 'maxOverlapFraction': [0, .5],
		# 'ignorePositions': [True],
		# 'saveMotifsDir': [saveDir]
		# },
		# all instances, mdl
		{'lengths': [np.arange(1./20, 1./8, .005)],
		# {'lengths': [np.arange(1./20, 1./8, .05)],
		# {'lengths': [.1],
		# 'downsampleBy': [1],
		# 'downsampleBy': [5],
		'downsampleBy': [2],
		# 'downsampleBy': [10],
		'onlyReturnPair': [False],
		'threshAlgo': ['mdl'],
		# 'mdlBits': [4, 5, 6, 7, 8],
		'mdlBits': [6],
		'mdlAbandonCriterion': [None, 'bestPairNegative', 'allNegative'],
		# 'mdlAbandonCriterion': [None],
		'maxOverlapFraction': [.5],
		'ignorePositions': [True],
		'saveMotifsDir': [saveDir],
		# 'showMotifs': [True],
		},
	]

	dataLoading = [DataLoader(), MSRC_PARAMS]
	motifFinding = (MotifExtractor(), extractorParams)

	d = [("LoadData", [dataLoading]),
		("FindMotif", [motifFinding])
	]

	df = learn.tryParams(d, None, None, crossValidate=False)
	return df


def ff_results_msrc(saveDir):
	if saveDir:
		saveDir = os.path.join(saveDir, 'ff')

	dimSelectingParams = [{'whichDims': [[24,25,26]]}]
	ffParams = [
		{'Lmin': [1./20],
		'Lmax': [1./10],
		# 'downsampleBy': [2],
		'downsampleBy': [1],
		'includeLocalZnorm': [False],
		# 'includeLocalZnorm': [True],
		'includeLocalSlope': [False],
		# 'includeLocalSlope': [True],
		'includeSaxHashes': [True],
		'ignorePositions': [True],
		'saveMotifsDir': [saveDir],
		},
	]
	dataLoading = [DataLoader(), MSRC_PARAMS]
	dimSelecting = (DimsSelector(), dimSelectingParams)
	motifFinding = (FFExtractor(), ffParams)

	d = [("LoadData", [dataLoading]),
		("SelectDims", [dimSelecting]),
		("FindMotif", [motifFinding])
	]

	df = learn.tryParams(d, None, None, crossValidate=False)
	return df


# ================================================================ TIDIGITS

TIDIGITS_PARAMS = [{'datasetName': ['tidigits_grouped_mfccs'],
		# 'whichExamples':[[1,2]],
		# 'whichExamples':[[2]],
		'whichExamples':[range(2, 100, 5)],
		# 'whichExamples':[[32, 37]],
		# 'seed': [np.random.rand(2)]
		'seed': [123]
}]

def results_tidigits(saveDir):
	if saveDir:
		saveDir = os.path.join(saveDir, 'sota')

	extractorParams = [
		# all instances, mdl
		{'lengths': [np.arange(1./20, 1./8, .005)],
		# {'lengths': [np.arange(1./20, 1./8, .05)],
		# {'lengths': [.1],
		# 'downsampleBy': [1],
		'downsampleBy': [5],
		# 'downsampleBy': [2],
		# 'downsampleBy': [10],
		'onlyReturnPair': [False],
		'threshAlgo': ['mdl'],
		# 'mdlBits': [4, 5, 6, 7, 8],
		'mdlBits': [6],
		# 'mdlAbandonCriterion': [None, 'bestPairNegative', 'allNegative'],
		# 'mdlAbandonCriterion': ['bestPairNegative', 'allNegative'],
		'mdlAbandonCriterion': ['allNegative'],
		# 'mdlAbandonCriterion': [None],
		'maxOverlapFraction': [.5],
		'ignorePositions': [True],
		'saveMotifsDir': [saveDir],
		# 'showMotifs': [True],
		},
	]

	dataLoading = [DataLoader(), TIDIGITS_PARAMS]
	motifFinding = (MotifExtractor(), extractorParams)

	d = [("LoadData", [dataLoading]),
		("FindMotif", [motifFinding])
	]

	df = learn.tryParams(d, None, None, crossValidate=False)
	return df


def ff_results_tidigits(saveDir):
	if saveDir:
		saveDir = os.path.join(saveDir, 'ff')

	# dimSelectingParams = [{'whichDims': [[24,25,26]]}]
	ffParams = [
		{'Lmin': [1./20],
		'Lmax': [1./10],
		# 'downsampleBy': [2],
		# 'downsampleBy': [1],
		'downsampleBy': [5],
		# 'includeLocalZnorm': [False],
		'includeLocalZnorm': [True],
		# 'includeLocalSlope': [False],
		'includeLocalSlope': [True],
		# 'includeSaxHashes': [False],
		'includeSaxHashes': [True],
		'ignorePositions': [True],
		'saveMotifsDir': [saveDir],
		},
	]
	dataLoading = [DataLoader(), TIDIGITS_PARAMS]
	# dimSelecting = (DimsSelector(), dimSelectingParams)
	motifFinding = (FFExtractor(), ffParams)

	d = [("LoadData", [dataLoading]),
		# ("SelectDims", [dimSelecting]),
		("FindMotif", [motifFinding])
	]

	df = learn.tryParams(d, None, None, crossValidate=False)
	return df


# ================================================================ UCR

UCR_TS_PER_DATASET = 2
UCR_PARAMS = [{'datasetName': ['ucr_all'], 'whichExamples': [range(UCR_TS_PER_DATASET)]}]

def results_ucr(saveDir):
	if saveDir:
		saveDir = os.path.join(saveDir, 'sota')

	extractorParams = [
		# all instances, mdl
		{'lengths': [np.arange(1./20, 1./8, .005)],
		# {'lengths': [np.arange(1./20, 1./8, .05)],
		# {'lengths': [.1],
		# 'downsampleBy': [1],
		'downsampleBy': [5],
		# 'downsampleBy': [2],
		# 'downsampleBy': [10],
		'onlyReturnPair': [False],
		'threshAlgo': ['mdl'],
		# 'mdlBits': [4, 5, 6, 7, 8],
		'mdlBits': [6],
		# 'mdlAbandonCriterion': [None, 'bestPairNegative', 'allNegative'],
		# 'mdlAbandonCriterion': ['bestPairNegative', 'allNegative'],
		'mdlAbandonCriterion': ['allNegative'],
		# 'mdlAbandonCriterion': [None],
		'maxOverlapFraction': [.5],
		'ignorePositions': [False],
		'saveMotifsDir': [saveDir],
		# 'showMotifs': [True],
		},
	]

	dataLoading = [DataLoader(), UCR_PARAMS]
	motifFinding = (MotifExtractor(), extractorParams)

	d = [("LoadData", [dataLoading]),
		("FindMotif", [motifFinding])
	]

	df = learn.tryParams(d, None, None, crossValidate=False)
	return df


def ff_results_ucr(saveDir):
	ffParams = [
		{'Lmin': [1./20],
		'Lmax': [1./10],
		'downsampleBy': [2],
		# 'downsampleBy': [1],
		# 'downsampleBy': [5],
		# 'includeLocalZnorm': [False],
		'includeLocalZnorm': [True],
		# 'includeLocalSlope': [False],
		'includeLocalSlope': [True],
		# 'includeSaxHashes': [False],
		'includeSaxHashes': [True],
		'ignorePositions': [False],
		'saveMotifsDir': [saveDir],
		},
	]
	dataLoading = [DataLoader(), UCR_PARAMS]
	motifFinding = (FFExtractor(), ffParams)

	d = [("LoadData", [dataLoading]),
		("FindMotif", [motifFinding])
	]

	df = learn.tryParams(d, None, None, crossValidate=False)
	return df


# ================================================================ Dishwasher

DISHWASHER_PARAMS = [{'datasetName': ['dishwasher']}]
DISHWASHER_2_PARAMS = [{'datasetName': ['dishwasher_2']}]
DISHWASHER_3_PARAMS = [{'datasetName': ['dishwasher_3']}]
DISHWASHER_SHORT_PARAMS = [{'datasetName': ['dishwasher_short']}]

def results_dishwasher(saveDir, datasetName):
	if saveDir:
		saveDir = os.path.join(saveDir, 'sota')

	dishwasherParams = [{'datasetName': [datasetName]}]
	extractorParams = [
		# # all instances, no added data
		# {'lengths': [np.arange(1./20, 1./8, .005)],
		# 'downsampleBy': [5],
		# 'onlyReturnPair': [False],
		# 'threshAlgo': [None, 'minnen', 'maxsep'],
		# 'maxOverlapFraction': [0, .5],
		# 'ignorePositions': [True],
		# 'saveMotifsDir': [saveDir]
		# },
		# all instances, mdl
		# {'lengths': [np.arange(100) + 100],
		{'lengths': [150],
		# {'lengths': [np.arange(1./20, 1./8, .05)],
		# {'lengths': [.1],
		'downsampleBy': [1],
		# 'downsampleBy': [5],
		# 'downsampleBy': [2],
		# 'downsampleBy': [10],
		'onlyReturnPair': [False],
		'threshAlgo': ['mdl'],
		# 'mdlBits': [4, 5, 6, 7, 8],
		'mdlBits': [6],
		# 'mdlAbandonCriterion': [None, 'bestPairNegative', 'allNegative'],
		# 'mdlAbandonCriterion': ['bestPairNegative', 'allNegative'],
		'mdlAbandonCriterion': ['allNegative'],
		# 'mdlAbandonCriterion': [None],
		'maxOverlapFraction': [.5],
		'ignorePositions': [False],
		'saveMotifsDir': [saveDir],
		# 'showMotifs': [True],
		},
	]

	dataLoading = [DataLoader(), dishwasherParams]
	motifFinding = (MotifExtractor(), extractorParams)

	d = [("LoadData", [dataLoading]),
		("FindMotif", [motifFinding])
	]

	df = learn.tryParams(d, None, None, crossValidate=False)
	return df


def ff_results_dishwasher(saveDir, datasetName):
	if saveDir:
		saveDir = os.path.join(saveDir, 'ff', datasetName)

	dishwasherParams = [{'datasetName': [datasetName]}]
	ffParams = [
		{'Lmin': [50],
		'Lmax': [200],
		# 'Lmax': [400],
		'downsampleBy': [1],
		# 'downsampleBy': [4], # TODO was failing to match stuff at 1
		# 'includeLocalZnorm': [False],
		'includeLocalZnorm': [True],
		# 'includeLocalSlope': [False],
		'includeLocalSlope': [True],
		'includeSaxHashes': [False],
		# 'includeSaxHashes': [True],
		'ignorePositions': [False],
		# 'ignorePositions': [True],
		'saveMotifsDir': [saveDir],
		},
	]

	dataLoading = [DataLoader(), dishwasherParams]
	motifFinding = (FFExtractor(), ffParams)

	d = [("LoadData", [dataLoading]),
		("FindMotif", [motifFinding])
	]

	df = learn.tryParams(d, None, None, crossValidate=False)
	return df


# ================================================================ Generic


def find_pattern(algo, datasetParams, saveDir, fractionAdversarialDims=0.,
	fractionNoiseDims=0., downsampleBy=2, lengths=None, ignorePositions=True):

	if lengths is None:
		lengths = np.arange(1./20, 1./8, .005)

	maxOverlapFractions = [.5]

	# compensate for fact that adversarial dims will be appended beforehand
	fractionNoiseDims = fractionNoiseDims / (1. + fractionAdversarialDims)

	# ------------------------ motif discovery object + params

	if algo == 'mdl':
		finderParams = [
			{'lengths': [lengths],
			'downsampleBy': [downsampleBy],
			'onlyReturnPair': [False],
			'threshAlgo': ['mdl'],
			'mdlBits': [6],
			# 'mdlAbandonCriterion': ['bestPairNegative', 'allNegative'],
			'mdlAbandonCriterion': ['allNegative'],
			'maxOverlapFraction': maxOverlapFractions,
			'ignorePositions': [ignorePositions],
			'saveMotifsDir': [saveDir],
			},
		]
		motifFinding = (MotifExtractor(), finderParams)
	elif algo == 'minnen':
		# all instances, no added data
		finderParams = [
			{'lengths': [lengths],
			'downsampleBy': [downsampleBy],
			'onlyReturnPair': [False],
			'threshAlgo': ['minnen'],
			'maxOverlapFraction': maxOverlapFractions,
			'ignorePositions': [ignorePositions],
			'saveMotifsDir': [saveDir]
			},
		]
		motifFinding = (MotifExtractor(), finderParams)
	elif algo == 'ff':
		finderParams = [
			{'Lmin': [np.min(lengths)],
			'Lmax': [np.max(lengths)],
			'downsampleBy': [downsampleBy],
			# 'includeLocalZnorm': [False],
			'includeLocalZnorm': [True],
			# 'includeLocalSlope': [False],
			'includeLocalSlope': [True],
			'includeSaxHashes': [False],
			# 'includeSaxHashes': [True],
			'ignorePositions': [ignorePositions],
			'saveMotifsDir': [saveDir],
			},
		]
		motifFinding = (FFExtractor(), finderParams)

	# ------------------------ other objects + params

	dataLoading = [DataLoader(), datasetParams]

	adversarialDimParams = [{'fractionToAdd': [fractionAdversarialDims]}]
	noiseDimParams = [{'fractionToAdd': [fractionNoiseDims]}]

	adversarialDims = [AdversarialDimsAppender(), adversarialDimParams]
	noiseDims = [NoiseDimsAppender(), noiseDimParams]

	# ------------------------ main

	d = [("LoadData", [dataLoading]),
		("AdversarialDims", [adversarialDims]),
		("NoiseDims", [noiseDims]),
		("FindMotif", [motifFinding])
	]

	df = learn.tryParams(d, None, None, crossValidate=False)
	return df


# def find_pattern_mdl(datasetParams, saveDir, downsampleBy=2):
# 	if saveDir:
# 		saveDir = os.path.join(saveDir, 'sota')

# 	extractorParams = [
# 		{'lengths': [np.arange(1./20, 1./8, .005)],
# 		'downsampleBy': [downsampleBy],
# 		'onlyReturnPair': [False],
# 		'threshAlgo': ['mdl'],
# 		'mdlBits': [6],
# 		# 'mdlAbandonCriterion': ['bestPairNegative', 'allNegative'],
# 		'mdlAbandonCriterion': ['allNegative'],
# 		'maxOverlapFraction': [.5],
# 		'ignorePositions': [True],
# 		'saveMotifsDir': [saveDir],
# 		},
# 	]

# 	dataLoading = [DataLoader(), datasetParams]
# 	motifFinding = (MotifExtractor(), extractorParams)

# 	d = [("LoadData", [dataLoading]),
# 		("FindMotif", [motifFinding])
# 	]

# 	df = learn.tryParams(d, None, None, crossValidate=False)
# 	return df


# def find_pattern_ff(datasetParams, saveDir, downsampleBy=2):
# 	ffParams = [
# 		{'Lmin': [1./20],
# 		'Lmax': [1./10],
# 		'downsampleBy': [downsampleBy],
# 		# 'downsampleBy': [1],
# 		# 'downsampleBy': [5],
# 		# 'includeLocalZnorm': [False],
# 		'includeLocalZnorm': [True],
# 		# 'includeLocalSlope': [False],
# 		'includeLocalSlope': [True],
# 		# 'includeSaxHashes': [False],
# 		'includeSaxHashes': [True],
# 		'ignorePositions': [False],
# 		'saveMotifsDir': [saveDir],
# 		},
# 	]
# 	dataLoading = [DataLoader(), datasetParams]
# 	motifFinding = (FFExtractor(), ffParams)

# 	d = [("LoadData", [dataLoading]),
# 		("FindMotif", [motifFinding])
# 	]

# 	df = learn.tryParams(d, None, None, crossValidate=False)
# 	return df

# ================================================================ Main

def science():
	saveDir = 'figs/motif/experiments/'
	ensureDirExists(saveDir)
	# saveDir = None

	# saveDir = saveDir + 'synth/'
	# df = results_synthetic(saveDir)

	# saveDir = saveDir + 'msrc/'
	# df = results_msrc(saveDir)
	# df = ff_results_msrc(saveDir)

	# saveDir += 'tidigits'
	# df = results_tidigits(saveDir)
	# df = ff_results_tidigits(saveDir)

	saveDir += 'ucr'
	df = results_ucr(saveDir)
	# df = ff_results_ucr(saveDir)

	# saveDir += 'dishwasher'
	# df = results_dishwasher(saveDir, 'dishwasher_short')
	# df = results_dishwasher(saveDir, 'dishwasher_2')
	# df = ff_results_dishwasher(saveDir, 'dishwasher_2')
	# df = ff_results_dishwasher(saveDir, 'dishwasher_3')


	print df

	SAVE_DIR = './results/motif/'
	ensureDirExists(SAVE_DIR)
	df.to_csv(os.path.join(SAVE_DIR, nowAsString() + '.csv'))


if __name__ == '__main__':
	science()
