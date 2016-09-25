#!/usr/bin/env/ python

import os
import sys
import numpy as np
import pandas as pd

from ..datasets.datasets import DataLoader
from ..utils.files import ensureDirExists, nowAsString
from ..utils import learn
from ..utils import pyience as pyn

from main2 import PARAMS_FOR_DATASET, PARAMS_FOR_ALGO
from main2 import ALGO_FF, ALGO_MDL, ALGO_MINNEN
from estimators import MotifExtractor, FFExtractor, DimsAppender

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


