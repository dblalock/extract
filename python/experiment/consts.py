#!/usr/bin/env python

from ..datasets import datasets as ds

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

DATASET_MSRC = ds.MSRC
DATASET_TIDIGITS = ds.TIDIGITS
DATASET_UCR = ds.UCR
DATASET_DISHWASHER_GROUPS = ds.DISHWASHER
DATASET_UCR_PAIRS = 'ucr_pairs'
DATASET_DISHWASHER = 'dishwasher'
DATASET_DISHWASHER_2 = 'dishwasher_2'
DATASET_DISHWASHER_3 = 'dishwasher_3'
DATASET_DISHWASHER_SHORT = 'dishwasher_short'
DATASET_DISHWASHER_PAIRS = 'dishwasher_pairs'

DATASET_TRIANGLES = ds.TRIANGLES
DATASET_RECTS = ds.RECTS
DATASET_SINES = ds.SINES
DATASET_SHAPES = ds.SHAPES
DATASET_RAND_WALK = ds.RANDWALK
DATASET_SYNTHETIC = 'synthetic'

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
