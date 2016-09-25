#!/usr/env/python

# This file was used to run official experiments

import os
import sys
import numpy as np
import pandas as pd

from ..datasets.datasets import DataLoader
from ..utils import learn
from ..utils.files import ensureDirExists
from ..utils.misc import nowAsString

from estimators import * # TODO don't do this

# ================================================================ Constants

# OFFICIAL_SAVE_DIR_FIG = 'figs/official/'
# OFFICIAL_SAVE_DIR_RESULTS = 'results/official/'

OFFICIAL_SAVE_DIR_FIG = 'figs/unofficial/'
OFFICIAL_SAVE_DIR_RESULTS = 'results/unofficial/'

# ------------------------ Default extractor / preprocesssing parameters

DEFAULT_DOWNSAMPLE_BY = 2
DEFAULT_LENGTHS = [1./20, 1./10]
DEFAULT_LENGTH_STEP = 5
DEFAULT_FRACTIONS_NOISE_DIMS = [1.] # run with orig data separately
DEFAULT_FRACTIONS_ADVERSARIAL_DIMS = [1.]

# ------------------------ Algorithms

ALGO_MDL = 'mdl'
ALGO_MINNEN = 'minnen'
ALGO_OUTCAST = 'outcast'
ALGO_FF = 'ff'

ALL_ALGORITHMS = [ALGO_MDL, ALGO_MINNEN, ALGO_FF] # ignore outcast

DEFAULT_MINNEN_PARAMS = [{
	'lengthStep': [DEFAULT_LENGTH_STEP],
	'threshAlgo': ['minnen'],
}]

DEFAULT_MDL_PARAMS = [{
	'lengthStep': [DEFAULT_LENGTH_STEP],
	'threshAlgo': ['mdl'],
	'mdlBits': [6],
	'mdlAbandonCriterion': ['allNegative'],
}]

DEFAULT_FF_PARAMS = [{
	'includeLocalSlope': [True],
	'detrend': [True],
	'ignoreFlat': [True]
}]

PARAMS_FOR_ALGO = {
	ALGO_MINNEN: DEFAULT_MINNEN_PARAMS,
	ALGO_MDL: DEFAULT_MDL_PARAMS,
	ALGO_FF: DEFAULT_FF_PARAMS
}

# ------------------------ Dataset names

DATASET_MSRC = 'msrc'
DATASET_TIDIGITS = 'tidigits'
DATASET_UCR = 'ucr'
DATASET_UCR_PAIRS = 'ucr_pairs'
DATASET_DISHWASHER = 'dishwasher'
DATASET_DISHWASHER_2 = 'dishwasher_2'
DATASET_DISHWASHER_3 = 'dishwasher_3'
DATASET_DISHWASHER_SHORT = 'dishwasher_short'
DATASET_DISHWASHER_GROUPS = 'dishwasher_groups'
DATASET_DISHWASHER_PAIRS = 'dishwasher_pairs'

DATASET_TRIANGLES = 'triangles'
DATASET_RECTS = 'rects'
DATASET_SINES = 'sines'
DATASET_SHAPES = 'shapes'
DATASET_SYNTHETIC = 'synthetic'
DATASET_RAND_WALK = 'randwalk'

# ------------------------ Dataset params

DEV_SEED = 12345
TEST_SEED = 123
SEED = DEV_SEED
# SEED = TEST_SEED

DEFAULT_INSTANCES_PER_TS = 5

MSRC_PARAMS = [{'datasetName': ['msrc'],
		'whichExamples': [None],
		'seed': [SEED]
}]
TIDIGITS_PARAMS = [{'datasetName': ['tidigits_grouped_mfccs'],
		'whichExamples': [None],
		'instancesPerTs': [DEFAULT_INSTANCES_PER_TS],
		'seed': [SEED]
}]
UCR_TS_PER_DATASET = 50
UCR_PARAMS = [{'datasetName': ['ucr_short'],
				'whichExamples': [range(UCR_TS_PER_DATASET)],
				'instancesPerTs': [DEFAULT_INSTANCES_PER_TS],
				'seed': [SEED]
}]
UCR_PAIRS_TS_PER_DATASET = 20
UCR_PAIRS_PARAMS = [{'datasetName': ['ucr_pairs'],
				'whichExamples': [range(UCR_PAIRS_TS_PER_DATASET)],
				'instancesPerTs': [2],
				'seed': [SEED]
}]
DISHWASHER_PARAMS = [{'datasetName': ['dishwasher']}]
DISHWASHER_2_PARAMS = [{'datasetName': ['dishwasher_2']}]
DISHWASHER_3_PARAMS = [{'datasetName': ['dishwasher_3']}]
DISHWASHER_SHORT_PARAMS = [{'datasetName': ['dishwasher_short']}]
DISHWASHER_GROUPS_PARAMS = [{'datasetName': ['dishwasher_groups'],
							'whichExamples': [None],
							'instancesPerTs': [DEFAULT_INSTANCES_PER_TS],
							'seed': [SEED]
}]
DISHWASHER_PAIRS_PARAMS = [{'datasetName': ['dishwasher_pairs'],
							'whichExamples': [None],
							'instancesPerTs': [2], # "aim for" 2
							'minNumInstances': [2], # enforce at least 2
							'maxNumInstances': [2], # enforce exactly 2
							'seed': [SEED]
}]

# synthetic data for prototyping
SYNTHETIC_TS_PER_TYPE = 50
TRIANGLES_PARAMS = [{'datasetName': ['triangles']}]
RECTS_PARAMS = [{'datasetName': ['rects']}]
SINES_PARAMS = [{'datasetName': ['sines']}]
SHAPES_PARAMS = [{'datasetName': ['shapes']}]
SYNTHETIC_PARAMS = [{'datasetName': [['triangles', 'rects', 'shapes']],
					'whichExamples': [range(SYNTHETIC_TS_PER_TYPE)],
}]
RAND_WALK_PARAMS = [{'datasetName': ['randwalk']}]

# ------------------------ Names -> Params

PARAMS_FOR_DATASET = {
	DATASET_MSRC: MSRC_PARAMS,
	DATASET_TIDIGITS: TIDIGITS_PARAMS,
	DATASET_UCR: UCR_PARAMS,
	DATASET_UCR_PAIRS: UCR_PAIRS_PARAMS,
	DATASET_DISHWASHER: DISHWASHER_PARAMS,
	DATASET_DISHWASHER_2: DISHWASHER_2_PARAMS,
	DATASET_DISHWASHER_3: DISHWASHER_3_PARAMS,
	DATASET_DISHWASHER_SHORT: DISHWASHER_SHORT_PARAMS,
	DATASET_DISHWASHER_GROUPS: DISHWASHER_GROUPS_PARAMS,
	DATASET_DISHWASHER_PAIRS: DISHWASHER_PAIRS_PARAMS,
	DATASET_TRIANGLES: TRIANGLES_PARAMS,
	DATASET_RECTS: RECTS_PARAMS,
	DATASET_SINES: SINES_PARAMS,
	DATASET_SHAPES: SHAPES_PARAMS,
	DATASET_SYNTHETIC: SYNTHETIC_PARAMS,
	DATASET_RAND_WALK: RAND_WALK_PARAMS
}


# ================================================================ Utils

def find_pattern(algo, datasetKey, saveDirFig, saveDirResults,
	fractionsAdversarialDims=None, fractionsNoiseDims=None,
	downsampleBy=2, lengths=DEFAULT_LENGTHS, lengthStep=DEFAULT_LENGTH_STEP,
	ignorePositions=True, requireContainment=False, mdlSearchEachTime=True,
	whichExamples=None, instancesPerTs=None, onlyReturnPair=False,
	forceNumDims=None, seedShiftStep=0., keepWhichDims=None):

	np.random.seed(SEED)

	if lengths is None:
		lengths = np.arange(1./20, 1./8, .005)

	maxOverlapFractions = [.25]

	saveDirResults = os.path.join(saveDirResults, algo)
	if saveDirFig:
		saveDirFig = os.path.join(saveDirFig, algo)
		if fractionsNoiseDims and any([frac > 0. for frac in fractionsNoiseDims]):
			saveDirFig = os.path.join(saveDirFig, 'noise')
		elif fractionsAdversarialDims and any([frac > 0. for frac in fractionsAdversarialDims]):
			saveDirFig = os.path.join(saveDirFig, 'adversarial')

	if not fractionsNoiseDims:
		fractionsNoiseDims = [0.]
	if not fractionsAdversarialDims:
		fractionsAdversarialDims = [0.]

	print "find_pattern(): using noise dim fractions: ", fractionsNoiseDims
	print "find_pattern(): using adversarial dim fractions: ", fractionsAdversarialDims

	# ------------------------ motif discovery object + params
	if algo == ALGO_MDL:
		finderParams = [
			{'lengths': [lengths],
			'lengthStep': [lengthStep],
			'downsampleBy': [downsampleBy],
			'onlyReturnPair': [onlyReturnPair],
			'threshAlgo': ['mdl'],
			'mdlBits': [6],
			# 'mdlAbandonCriterion': ['bestPairNegative', 'allNegative'],
			'mdlAbandonCriterion': ['allNegative'],
			'mdlSearchEachTime': [mdlSearchEachTime],
			'maxOverlapFraction': maxOverlapFractions,
			'ignorePositions': [ignorePositions],
			'requireContainment': [requireContainment],
			'forceNumDims': [forceNumDims],
			'seedShiftStep': [seedShiftStep],
			# 'saveMotifsDir': [saveDirFig],
			},
		]
		motifFinding = (MotifExtractor(), finderParams)
	elif algo == ALGO_MINNEN:
		# all instances, no added data
		finderParams = [
			{'lengths': [lengths],
			'lengthStep': [lengthStep],
			'downsampleBy': [downsampleBy],
			'onlyReturnPair': [onlyReturnPair],
			'threshAlgo': ['minnen'],
			'maxOverlapFraction': maxOverlapFractions,
			'ignorePositions': [ignorePositions],
			'requireContainment': [requireContainment],
			'forceNumDims': [forceNumDims],
			'seedShiftStep': [seedShiftStep],
			# 'saveMotifsDir': [saveDirFig]
			},
		]
		motifFinding = (MotifExtractor(), finderParams)
	elif algo == ALGO_OUTCAST:
		finderParams = [
			{'Lmin': [np.min(lengths)],
			'Lmax': [np.max(lengths)],
			'lengthStep': [lengthStep], # just for spacing small lengths
			'downsampleBy': [downsampleBy],
			'ignorePositions': [ignorePositions],
			'requireContainment': [requireContainment],
			# 'saveMotifsDir': [saveDirFig],
			},
		]
		motifFinding = (OutcastExtractor(), finderParams)
	elif algo == ALGO_FF:
		finderParams = [
			{'Lmin': [np.min(lengths)],
			'Lmax': [np.max(lengths)],
			'logX': [False],
			'logXblur': [False],
			'downsampleBy': [downsampleBy],
			# 'includeLocalZnorm': [False],
			# 'includeLocalZnorm': [True],
			# 'includeLocalSlope': [False],
			# 'includeLocalSlope': [True],
			# 'includeSaxHashes': [False],
			# 'includeSaxHashes': [True], # drowns out local slopes
			# 'includeMaxFilteredSlope': [False],
			# 'includeMaxFilteredSlope': [True],
			'includeNeighbors': [True],
			# 'includeVariance': [True],
			'saxCardinality': [3],
			'saxWordLen': [3],
			# 'detrend': [True],
			'detrend': [False],
			'ignoreFlat': [True],
			'ignorePositions': [ignorePositions],
			'requireContainment': [requireContainment],
			# 'saveMotifsDir': [saveDirFig],
			},
		]
		motifFinding = (FFExtractor(), finderParams)
	else:
		raise ValueError("Unrecognized algorithm {}!".format(algo))

	# ------------------------ other objects + params

	datasetParams = PARAMS_FOR_DATASET[datasetKey]
	if whichExamples:
		datasetParams[0]['whichExamples'] = [whichExamples]
	if instancesPerTs:
		datasetParams[0]['instancesPerTs'] = [instancesPerTs]

	dataLoading = [DataLoader(), datasetParams]

	dimAddingParams = [{'fractionAdversarialDims': fractionsAdversarialDims,
						'fractionNoiseDims': fractionsNoiseDims}]
	dimAdding = [DimsAppender(), dimAddingParams]

	# ------------------------ main

	d = [("DataLoading", [dataLoading]),
		("DimAdding", [dimAdding]),
		("FindMotif", [motifFinding])
	]

	# add option to only use some dimensions of the ts
	if keepWhichDims is not None and len(keepWhichDims):
		dimSelectingParams = [{'whichDims': [keepWhichDims]}]
		dimSelecting = (DimsSelector(), dimSelectingParams)
		stage = ("SelectDims", [dimSelecting])
		d.insert(1, stage) # insert right after loading data

	df = learn.tryParams(d, None, None, crossValidate=False)

	ensureDirExists(saveDirResults)
	fileName = "{}.csv".format(nowAsString())
	df.to_csv(os.path.join(saveDirResults, fileName))

	return df


class Experiment(object):
	"""Class to enforce passing in all of the parameters (and provide a
		clean interface to functions that need these params)"""

	def __init__(self, algo, dataset, downsampleBy, ignorePositions,
		requireContainment=False,
		lengths=DEFAULT_LENGTHS, lengthStep=DEFAULT_LENGTH_STEP,
		fractionsNoiseDims=DEFAULT_FRACTIONS_NOISE_DIMS,
		fractionsAdversarialDims=DEFAULT_FRACTIONS_ADVERSARIAL_DIMS,
		saveFig=True):

		self.algo = algo
		self.dataset = dataset
		self.downsampleBy = downsampleBy
		self.ignorePositions = ignorePositions
		self.requireContainment = requireContainment

		self.lengths = lengths
		self.lengthStep = lengthStep
		self.fractionsNoiseDims = fractionsNoiseDims
		self.fractionsAdversarialDims = fractionsAdversarialDims

		subdir = ''.join(dataset)
		print "Experiment: using subdir: ", subdir
		self.saveDirFig = ""
		if saveFig:
			self.saveDirFig = os.path.join(OFFICIAL_SAVE_DIR_FIG, subdir)
		self.saveDirRes = os.path.join(OFFICIAL_SAVE_DIR_RESULTS, subdir)

	def run(self, original=False, noise=False, adversarial=False, **kwargs):

		saveDirFig = self.saveDirFig
		saveDirRes = self.saveDirRes

		algo = self.algo
		dataset = self.dataset
		# dataset = kwargs.get('dataset', self.dataset)
		# kwargs.pop('dataset')
		# print "run(): using dataset {} (default is {})".format(dataset, self.dataset)
		# kwargs.setdefault('lengths', self.lengths) # hack to allow lengths as kwarg
		lengths = self.lengths
		lengthStep = self.lengthStep
		downsampleBy = self.downsampleBy
		fractionsNoiseDims = self.fractionsNoiseDims
		fractionsAdversarialDims = self.fractionsAdversarialDims
		ignorePositions = self.ignorePositions
		requireContainment = self.requireContainment

		if not (original or noise or adversarial):
			original = True # default to running on original data

		if original: # original dims
			find_pattern(algo, dataset, saveDirFig, saveDirRes,
				lengths=lengths, lengthStep=lengthStep, downsampleBy=downsampleBy,
				ignorePositions=ignorePositions,
				requireContainment=requireContainment, **kwargs)
		if noise: # noise dims
			find_pattern(algo, dataset, saveDirFig, saveDirRes,
				lengths=lengths, lengthStep=lengthStep, downsampleBy=downsampleBy,
				ignorePositions=ignorePositions,
				requireContainment=requireContainment,
				fractionsNoiseDims=fractionsNoiseDims, **kwargs)
		if adversarial: # adversarial dims
			find_pattern(algo, dataset, saveDirFig, saveDirRes,
				lengths=lengths, lengthStep=lengthStep, downsampleBy=downsampleBy,
				ignorePositions=ignorePositions,
				requireContainment=requireContainment,
				fractionsAdversarialDims=fractionsAdversarialDims, **kwargs)

# ============================================================== Official Stuff

# ------------------------------------------------ experiment creation / params

def create_msrc_experiment(algo):
	return Experiment(algo, DATASET_MSRC, downsampleBy=2, ignorePositions=True)
	# return Experiment(algo, DATASET_MSRC, downsampleBy=5, ignorePositions=True)


def create_tidigits_experiment(algo):
	return Experiment(algo, DATASET_TIDIGITS, downsampleBy=2,
		# ignorePositions=False, requireContainment=True, lengths=[1./16, 1./8])
		ignorePositions=False, requireContainment=False, lengths=[1./16, 1./8])


def create_dishwasher_experiment(algo):
	return Experiment(algo, DATASET_DISHWASHER_GROUPS, downsampleBy=1, ignorePositions=False)
	# return Experiment(algo, DATASET_DISHWASHER_GROUPS, downsampleBy=5, ignorePositions=False)


def create_ucr_experiment(algo):
	return Experiment(algo, DATASET_UCR, downsampleBy=2, ignorePositions=False)

# ------------------------------------------------ mdl

def run_msrc_mdl(**kwargs):
	create_msrc_experiment(ALGO_MDL).run(**kwargs)


def run_tidigits_mdl(**kwargs):
	create_tidigits_experiment(ALGO_MDL).run(**kwargs)


def run_dishwasher_mdl(**kwargs):
	create_dishwasher_experiment(ALGO_MDL).run(**kwargs)


def run_ucr_mdl(**kwargs):
	create_ucr_experiment(ALGO_MDL).run(**kwargs)

# ------------------------------------------------ minnen

def run_msrc_minnen(**kwargs):
	create_msrc_experiment(ALGO_MINNEN).run(**kwargs)


def run_tidigits_minnen(**kwargs):
	create_tidigits_experiment(ALGO_MINNEN).run(**kwargs)


def run_dishwasher_minnen(**kwargs):
	create_dishwasher_experiment(ALGO_MINNEN).run(**kwargs)


def run_ucr_minnen(**kwargs):
	create_ucr_experiment(ALGO_MINNEN).run(**kwargs)

# ------------------------------------------------ outcast

def run_msrc_outcast(**kwargs):
	create_msrc_experiment(ALGO_OUTCAST).run(**kwargs)


def run_tidigits_outcast(**kwargs):
	create_tidigits_experiment(ALGO_OUTCAST).run(**kwargs)


def run_dishwasher_outcast(**kwargs):
	create_dishwasher_experiment(ALGO_OUTCAST).run(**kwargs)


def run_ucr_outcast(**kwargs):
	create_ucr_experiment(ALGO_OUTCAST).run(**kwargs)

# ------------------------------------------------ ffs

def run_msrc_ff(**kwargs):
	create_msrc_experiment(ALGO_FF).run(**kwargs)


def run_tidigits_ff(**kwargs):
	create_tidigits_experiment(ALGO_FF).run(**kwargs)


def run_dishwasher_ff(**kwargs):
	create_dishwasher_experiment(ALGO_FF).run(**kwargs)


def run_ucr_ff(**kwargs):
	create_ucr_experiment(ALGO_FF).run(**kwargs)

# ------------------------------------------------ best pairs

def run_dishwasher_pairs_motif(**kwargs):
	e = Experiment(ALGO_MINNEN, DATASET_DISHWASHER_PAIRS, downsampleBy=1,
		ignorePositions=False, lengths=[150])
	e.run(onlyReturnPair=True, **kwargs)

def run_dishwasher_pairs_ff(**kwargs):
	e = Experiment(ALGO_FF, DATASET_DISHWASHER_PAIRS, downsampleBy=1,
		ignorePositions=False, lengths=[100, 200])
	e.run(**kwargs)

def run_ucr_pairs_motif(**kwargs):
	e = Experiment(ALGO_MINNEN, DATASET_UCR_PAIRS, downsampleBy=2,
		ignorePositions=False, lengths=[.17])
	e.run(onlyReturnPair=True, **kwargs)

def run_ucr_pairs_ff(**kwargs):
	e = Experiment(ALGO_FF, DATASET_UCR_PAIRS, downsampleBy=2,
		ignorePositions=False, lengths=[.1, .2])
	e.run(**kwargs)

def run_synthetic_pairs_motif(**kwargs):
	e = Experiment(ALGO_MINNEN, DATASET_SYNTHETIC, downsampleBy=1,
		ignorePositions=False, lengths=[50])
	e.run(onlyReturnPair=True, **kwargs)

def run_synthetic_pairs_ff(**kwargs):
	e = Experiment(ALGO_FF, DATASET_SYNTHETIC, downsampleBy=1,
		ignorePositions=False, lengths=[.1, .2])
	e.run(instancesPerTs=2, **kwargs)

# ------------------------------------------------ scalability

DEFAULT_SCALABILITY_M_LENGTHS = np.arange(50, 100)
DEFAULT_SCALABILITY_N = 5000
SCALABIILTY_N_FOR_M_LENGTHS = 5000
# DEFAULT_SCALABILITY_N = 1100 # TODO remove after test
# DEFAULT_SCALABILITY_N_LENGTHS = [500, 1000, 2000, 4000, 8000, 16000]
DEFAULT_SCALABILITY_N_LENGTHS = [500, 1000, 2000, 4000, 8000]
# DEFAULT_SCALABILITY_N_LENGTHS = [500, 1000]
# DEFAULT_SCALABILITY_N_LENGTHS = [500, 1000] # TODO remove after test
DEFAULT_SCALABILITY_M_LENGTH_MINS_MAXES = [[50, 100],
											[100, 150],
											[150, 200],
											[200, 250],
											[250, 300],
											[300, 350],
											[350, 400]]
# 											[350, 400],
# 											[400, 450],
# 											[450, 500]]
DEFAULT_SCALABILITY_M_SPAN_MINS_MAXES = [[150, 151],
										[140, 160],
										[130, 170],
										[120, 180],
										[110, 190],
										[100, 200]]
# 										[90, 210],
# 										[80, 220],
# 										[70, 230],
# 										[60, 240],
# 										[50, 250]]
SCALABILITY_RESULTS_DIR = os.path.join(OFFICIAL_SAVE_DIR_RESULTS, 'scalability')

SCALABILITY_TEST_N_LENGTH = 'n_length'
SCALABILITY_TEST_M_LENGTH = 'm_length'
SCALABILITY_TEST_M_SPAN = 'm_span'
SCALABILITY_ALL_TESTS = [SCALABILITY_TEST_N_LENGTH, SCALABILITY_TEST_M_LENGTH,
	SCALABILITY_TEST_M_SPAN]


def save_data_frame(df, saveDir):
	ensureDirExists(saveDir)
	fileName = "{}.csv".format(nowAsString())
	df.to_csv(os.path.join(saveDir, fileName))


def find_pattern_simple(datasetKey, algoKey, datasetParams=None, algoParams=None,
	saveDir=None):
	if not datasetParams:
		datasetParams = PARAMS_FOR_DATASET[datasetKey]
	if not algoParams:
		algoParams = PARAMS_FOR_ALGO[algoKey]

	print "datasetParams", datasetParams
	print "algoParams", algoParams

	if algoKey == ALGO_MINNEN or algoKey == ALGO_MDL:
		algoObj = MotifExtractor()
	elif algoKey == ALGO_FF:
		algoObj = FFExtractor()

	dataLoading = [DataLoader(), datasetParams]
	motifFinding = [algoObj, algoParams]

	d = [("DataLoading", [dataLoading]),
		("FindMotif", [motifFinding])
	]
	df = learn.tryParams(d, None, None, crossValidate=False)

	if saveDir:
		save_data_frame(df, saveDir)
	return df


def setLengthsInAlgoParams(algoKey, algoParams, lengths):
	if algoKey == ALGO_FF:
		minLen = np.min(lengths)
		maxLen = np.max(lengths)
		algoParams[0].update({'Lmin': [minLen], 'Lmax': [maxLen]})
	else:
		algoParams[0].update({'lengths': [lengths]})
		algoParams[0].update({'lengthStep': [1]})


def test_scalability(datasetKey, algoKey, whichTest=None, iterNum=1):
	if iterNum < 1:
		raise ValueError("Iteration number must be >= 1! Got {}".format(iterNum))

	datasetParams = PARAMS_FOR_DATASET[datasetKey]
	datasetParams[0]['seed'] = [iterNum] # different across iterations
	algoParams = PARAMS_FOR_ALGO[algoKey]

	nLengths = np.array(DEFAULT_SCALABILITY_N_LENGTHS)

	mLengths = []
	for pair in DEFAULT_SCALABILITY_M_LENGTH_MINS_MAXES:
		mLengths.append(np.arange(pair[0], pair[1]))

	mSpans = []
	for pair in DEFAULT_SCALABILITY_M_SPAN_MINS_MAXES:
		mSpans.append(np.arange(pair[0], pair[1]))

	saveDir = os.path.join(SCALABILITY_RESULTS_DIR, datasetKey, algoKey, whichTest)

	dfs = []
	subdir = os.path.join(saveDir, 'incremental-{}'.format(iterNum))
	if whichTest == SCALABILITY_TEST_N_LENGTH:
		setLengthsInAlgoParams(algoKey, algoParams, DEFAULT_SCALABILITY_M_LENGTHS)
		for n in nLengths:
			datasetParams[0]['cropDataLength'] = [n]
			df = find_pattern_simple(datasetKey, algoKey, datasetParams, algoParams,
				saveDir=subdir)
			dfs.append(df)
	elif whichTest == SCALABILITY_TEST_M_LENGTH:
		datasetParams[0]['cropDataLength'] = [SCALABIILTY_N_FOR_M_LENGTHS]
		for lengths in mLengths:
			setLengthsInAlgoParams(algoKey, algoParams, lengths)
			df = find_pattern_simple(datasetKey, algoKey, datasetParams, algoParams,
				saveDir=subdir)
			dfs.append(df)
	elif whichTest == SCALABILITY_TEST_M_SPAN:
		datasetParams[0]['cropDataLength'] = [DEFAULT_SCALABILITY_N]
		for lengths in mSpans:
			setLengthsInAlgoParams(algoKey, algoParams, lengths)
			df = find_pattern_simple(datasetKey, algoKey, datasetParams, algoParams,
				saveDir=subdir)
			dfs.append(df)
	else:
		raise ValueError("can't run unknown scalability test {}!".format(which))

	df = pd.concat(dfs, ignore_index=True)
	df['iter'] = pd.Series(np.zeros(df.shape[0]) + iterNum)
	save_data_frame(df, saveDir)

	# recurse to iterate multiple times
	if iterNum > 1:
		df2 = test_scalability(datasetKey, algoKey, whichTest, iterNum - 1)
		df = pd.concat([df, df2], ignore_index=True)

	return df


def run_scalability_experiments(datasets=None, algorithms=None, tests=None, numIters=1):
	if not datasets:
		datasets = [DATASET_RAND_WALK, DATASET_DISHWASHER]
	if not algorithms:
		algorithms = [ALGO_MINNEN, ALGO_MDL, ALGO_FF]
	if not tests:
		tests = SCALABILITY_ALL_TESTS

	# wrap single strings in a list
	datasets = synth.ensureIsCollection(datasets)
	algorithms = synth.ensureIsCollection(algorithms)
	tests = synth.ensureIsCollection(tests)

	for dataset in datasets:
		for algo in algorithms:
			for test in tests:
				print "SCALABILITY: testing {} {} {}".format(
					dataset, algo, test)
				test_scalability(dataset, algo, test, numIters)


# ================================================================ Main

def main():
	# run_scalability_experiments(algorithms=ALGO_FF)
	# run_scalability_experiments(algorithms=ALGO_MINNEN)

	# run_tidigits_outcast(whichExamples=range(5))
	# run_ucr_outcast(whichExamples=range(1))

	# ------------------------ best pair experiments
	# run_synthetic_pairs_motif()
	# run_synthetic_pairs_ff(whichExamples=range(20))
	# run_ucr_ff(whichExamples=range(1))
	# run_ucr_ff(whichExamples=range(3))
	# run_ucr_pairs_motif()
	# run_ucr_pairs_ff(whichExamples=range(1))
	# run_dishwasher_pairs_motif(whichExamples=range(2))
	# run_dishwasher_pairs_ff(whichExamples=range(5))

	# run_tidigits_ff(original=True, whichExamples=range(1))
	# run_tidigits_ff(original=True, whichExamples=range(5))
	run_tidigits_ff(original=True, whichExamples=range(10))
	# run_tidigits_ff(original=True, whichExamples=[range(50)])
	# run_msrc_ff(original=True, whichExamples=range(1))
	# run_msrc_ff(original=True, whichExamples=range(10))
	# run_msrc_ff(original=True, whichExamples=range(0,100,10))
	# run_msrc_ff(original=True, whichExamples=[range(50)])
	# run_dishwasher_ff(original=True, whichExamples=range(1))
	# run_dishwasher_ff(original=True, whichExamples=range(5))
	# run_dishwasher_ff(original=True, whichExamples=range(1), keepWhichDims=[5])
	# run_ucr_ff(original=True, whichExamples=range(2))

	# run_msrc_outcast(original=True, whichExamples=[range(1)])
	# run_tidigits_outcast(original=True, whichExamples=[range(1)])
	# run_msrc_mdl(original=True, whichExamples=[range(1)])

	# aha; recording 16 crashes it for some reason...
	# run_msrc_mdl(original=True, whichExamples=[range(15, 17)]) # like 200s
	# run_msrc_mdl(original=True, whichExamples=range(50), seedShiftStep=.1)
	# run_msrc_mdl(original=True, whichExamples=range(50), forceNumDims=1)
	# run_msrc_mdl(original=True, whichExamples=range(50), forceNumDims=-1)
	# run_msrc_mdl(original=True, whichExamples=[range(1)], mdlSearchEachTime=True)
	# run_msrc_mdl(original=True, whichExamples=[range(1)], mdlSearchEachTime=False)
	# run_msrc_mdl(original=True, whichExamples=[range(10)])
	# run_msrc_mdl(original=True)
	# run_msrc_mdl(noise=True)
	# run_msrc_mdl(adversarial=True)

	# run_tidigits_mdl(original=True, whichExamples=range(10))
	# run_tidigits_mdl(original=True, whichExamples=range(2), forceNumDims=1) # like 10s
	# run_tidigits_mdl(original=True, whichExamples=range(50), forceNumDims=1)
	# run_tidigits_mdl(original=True, whichExamples=range(50), forceNumDims=-1)
	# run_tidigits_mdl(original=True, whichExamples=range(50), seedShiftStep=.1)
	# run_tidigits_mdl(original=True, whichExamples=[range(5)], mdlSearchEachTime=True) # like 10s
	# run_tidigits_mdl(original=True, whichExamples=[range(5)], mdlSearchEachTime=False) # like 10s
	# run_tidigits_mdl(original=True)
	# run_tidigits_mdl(noise=True)
	# run_tidigits_mdl(adversarial=True)

	# run_dishwasher_mdl(original=True, whichExamples=[range(2)])
	# run_dishwasher_mdl(original=True, whichExamples=[range(1, 2)], mdlSearchEachTime=True) # like 120s
	# run_dishwasher_mdl(original=True, whichExamples=[range(1, 2)], mdlSearchEachTime=False) # like 120s
	# run_dishwasher_mdl(original=True)
	# run_dishwasher_mdl(noise=True)
	# run_dishwasher_mdl(adversarial=True)
	# official_dishwasher_minnen(original=True)

	# run_msrc_minnen(original=True, whichExamples=range(50), forceNumDims=1)
	# run_msrc_minnen(original=True, whichExamples=range(50), forceNumDims=-1)
	# run_tidigits_minnen(original=True, whichExamples=range(50), forceNumDims=1)
	# run_tidigits_minnen(original=True, whichExamples=range(50), forceNumDims=-1)
	# run_dishwasher_minnen(original=True, whichExamples=range(30), forceNumDims=1)
	# run_dishwasher_minnen(original=True, whichExamples=range(30), forceNumDims=-1)

	# run_ucr_mdl(original=True, whichExamples=[range(2)], mdlSearchEachTime=True) # like 200s
	# run_ucr_mdl(original=True, whichExamples=[range(2)], mdlSearchEachTime=False) # like 200s
	# run_ucr_mdl(original=True, whichExamples=range(3))
	# run_ucr_mdl(original=True)
	# run_ucr_mdl(noise=True)
	# run_ucr_mdl(adversarial=True)

	# run_tidigits_minnen(original=True, whichExamples=range(10))
	# run_tidigits_minnen(original=True, whichExamples=[range(1)]) # like 12s
	# run_dishwasher_minnen(original=True, whichExamples=[range(1,2)])
	# run_ucr_minnen(original=True, whichExamples=range(3))


def commandLineMain(dataset, algorithm, original=False, noise=False,
	adversarial=False, scalabilityTests=None, numIters=1):

	if scalabilityTests:
		# note that this ignores the dataset
		datasets = [dataset] if dataset else None
		return run_scalability_experiments(datasets=datasets, algorithms=algorithm,
			tests=scalabilityTests, numIters=numIters)

	if dataset == DATASET_MSRC:
		if adversarial:
			raise ValueError("adversarial makes no sense with MSRC!")
		if algorithm == ALGO_MDL:
			run_msrc_mdl(original=original, noise=noise, adversarial=adversarial)
		elif algorithm == ALGO_MINNEN:
			run_msrc_minnen(original=original, noise=noise, adversarial=adversarial)
		elif algorithm == ALGO_FF:
			run_msrc_ff(original=original, noise=noise, adversarial=adversarial)

	elif dataset == DATASET_TIDIGITS:
		if algorithm == ALGO_MDL:
			run_tidigits_mdl(original=original, noise=noise, adversarial=adversarial)
		elif algorithm == ALGO_MINNEN:
			run_tidigits_minnen(original=original, noise=noise, adversarial=adversarial)
		elif algorithm == ALGO_FF:
			run_tidigits_ff(original=original, noise=noise, adversarial=adversarial)

	elif dataset == DATASET_DISHWASHER_GROUPS or dataset == DATASET_DISHWASHER:
		if algorithm == ALGO_MDL:
			run_dishwasher_mdl(original=original, noise=noise, adversarial=adversarial)
		elif algorithm == ALGO_MINNEN:
			run_dishwasher_minnen(original=original, noise=noise, adversarial=adversarial)
		elif algorithm == ALGO_FF:
			run_dishwasher_ff(original=original, noise=noise, adversarial=adversarial)

	elif dataset == DATASET_UCR:
		if algorithm == ALGO_MDL:
			run_ucr_mdl(original=original, noise=noise, adversarial=adversarial)
		elif algorithm == ALGO_MINNEN:
			run_ucr_minnen(original=original, noise=noise, adversarial=adversarial)
		elif algorithm == ALGO_FF:
			run_ucr_ff(original=original, noise=noise, adversarial=adversarial)

	else:
		print("ERROR: dataset {} not recognized".format(dataset))
		sys.exit(1)


def parseAndRunCommands():
	dataset = sys.argv[1]
	argStr = ''.join(sys.argv[2:])

	if dataset not in PARAMS_FOR_DATASET and dataset != 'scalability':
		raise ValueError("unrecognized dataset {}".format(dataset))

	# algorithm to use
	if 'mdl' in argStr:
		algo = ALGO_MDL
	elif 'minnen' in argStr:
		algo = ALGO_MINNEN
	elif 'ff' in argStr:
		algo = ALGO_FF
	else:
		raise ValueError("no algorithm specified! options = 'mdl', 'minnen', 'ff'")

	# run scalability tests instead of normal experiments; TODO make this
	# logic less of a hack
	scalability = 'scalability' in argStr or 'scale' in argStr
	if dataset == 'scalability': # if was first arg
		scalability = True
		dataset = None

	if scalability:
		tests = []
		for test in SCALABILITY_ALL_TESTS:
			if test in argStr:
				tests.append(test)

		numIters = 1
		try:
			num = int(sys.argv[-1]) # check if last arg was a number
			if num > 1:
				numIters = num
		except:
			pass

		print "Main(): Running {} iterations on dataset: {}".format(numIters, dataset)
		return commandLineMain(dataset, algo, scalabilityTests=tests, numIters=numIters)

	# dims to add to data
	original = 'orig' in argStr
	noise = 'noise' in argStr
	adversarial = 'adver' in argStr
	if 'all' in argStr:
		original = True
		noise = True
		adversarial = True

	if not (original or noise or adversarial):
		raise ValueError("no original/noise/adversarial setting specified")

	return commandLineMain(dataset, algo, original, noise, adversarial)


if __name__ == '__main__':
	if len(sys.argv) > 1: # arguments passed in
		parseAndRunCommands()
	else: # no arguments passed in
		main()

	sys.exit()

	# dataset = DATASET_SINES
	# dataset = DATASET_TRIANGLES
	# dataset = DATASET_RECTS
	# dataset = DATASET_SHAPES
	# dataset = DATASET_SYNTHETIC
	# dataset = DATASET_MSRC
	# dataset = DATASET_TIDIGITS

	howMany = 1
	# howMany = 10

	# e = Experiment(ALGO_FF, dataset, downsampleBy=1,
	e = Experiment(ALGO_OUTCAST, dataset, downsampleBy=1,
		ignorePositions=False, lengths=[.1, .2],
		# ignorePositions=True, lengths=[.0625, .125],
		# ignorePositions=True, lengths=[.05, .1],
		)
		# saveFig=False)
	e.run(whichExamples=range(howMany))
	# print("Ran experiment.")
	plt.show()




