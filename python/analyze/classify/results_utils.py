
# reusable funcs for the problem of analyzing results of the form I have, but
# that are not totally data-agnostic

import os
import numpy as np
from pandas import DataFrame as df
from pandas.io.parsers import read_csv

# from utils import listFilesInDir
from munge import *

DATASET_COL_NAME = 'Dataset'
ERR_RATE_COL_NAME = 'ErrRate'
CLASSIFIER_COL_NAME = 'Classifier'
ACCURACY_COL_NAME = 'accuracy' # lower case to match sklearn scorefunc

# note that this has to be a list, not tuple, if we want to index
# a dataframe using this directly
RESULT_COL_NAMES = [DATASET_COL_NAME, ACCURACY_COL_NAME, CLASSIFIER_COL_NAME]

parentDir = os.path.dirname(os.path.abspath(__file__))
CSVS_DIR = os.path.join(parentDir, 'sota_results')

UCR_DATASETS = [
    "ChlorineConcentration",
    "wafer",
    "MedicalImages",
    "FaceAll",
    "OSULeaf",
    "Adiac",
    "SwedishLeaf",
    "yoga",
    "Fish",
    "Lightning7",       # note that these are "Lighting", not
    "Lightning2",       # "Lightning", on the filesystem by default
    "Trace",
    "synthetic_control",
    "FacesUCR",
    "CinC_ECG_torso",
    "MALLAT",
    "Symbols",
    "Coffee",
    "ECG200",
    "FaceFour",
    "OliveOil",
    "Gun_Point",
    "Beef",
    "DiatomSizeReduction",
    "CBF",
    "ECGFiveDays",
    "TwoLeadECG",
    "SonyAIBORobotSurfaceII",
    "MoteStrain",
    "ItalyPowerDemand",
    "SonyAIBORobotSurface",
    "Two_Patterns",
    "StarLightCurves",
    "Haptics",
    "InlineSkate",
    "50words",
    "Cricket_Y",
    "Cricket_X",
    "Cricket_Z",
    "WordsSynonyms",
    "uWaveGestureLibrary_Z",
    "uWaveGestureLibrary_Y",
    "uWaveGestureLibrary_X"
]


def makeDatasetNameConsistent(name):
    return name.strip()     \
        .replace(' ', '_')  \
        .replace('-', '')   \
        .replace('ighting', 'ightning')


def cleanFrames(frames):
    for frame in frames:
        removeEmptyRowsAndCols(frame)

        for colName in frame._get_numeric_data():
            # ignore dataset stats
            if '#' in colName:
                col = np.array(col)
                continue

            col = frame[colName]

            # convert percentages
            if any(col > 1.0):
                col /= 100.
                frame[colName] = col

            # convert accuracies to error rates
            if colName[-3:].lower() == 'acc':
                frame[colName[:-3]] = (1. - col)
                deleteCols(frame, colName)

        # make column / dataset names pretty + consistent
        stripColNames(frame, ' \t_')
        for i, colName in enumerate(frame):
            name = colName.lower()
            if 'dataset' in name or 'name' in name:
                renameColAtIdx(frame, i, DATASET_COL_NAME)
                frame[DATASET_COL_NAME] = \
                    frame[DATASET_COL_NAME].map(makeDatasetNameConsistent)
    return frames


def readResultsAsFrames():
    # files = listFilesInDir(CSVS_DIR, endswith='.csv', absPaths=True)
    files = os.listdir(CSVS_DIR)
    absPaths = map(lambda f: os.path.join(CSVS_DIR, f), files)
    return [read_csv(f) for f in absPaths if f.endswith('.csv')]


def joinFramesByDataset(frames, onlyFromList=None):
    if onlyFromList:
        dummyFrame = df.from_dict({DATASET_COL_NAME: onlyFromList})
        frames.insert(0, dummyFrame)
        how = 'left'
    else:
        how = 'outer'

    joined = frames[0]
    for frame in frames[1:]:
        assert DATASET_COL_NAME in frame
        joined = joined.merge(frame, on=DATASET_COL_NAME, how=how, sort=True)
    return joined


def buildCombinedResults(includeEnsembles=False):
    frames = readResultsAsFrames()
    frames = cleanFrames(frames)
    # combined = joinFramesByDataset(frames)
    combined = joinFramesByDataset(frames, onlyFromList=UCR_DATASETS)

    # these screw up everything cuz they were tested on like no datasets,
    # so attempts to evaluate other classifiers get screwed up if they
    # only look at datasets that everything has in common; also, the
    # ensembles sort of sucked, so it doesn't matter much
    if not includeEnsembles:
        removeEnsembleCols(combined)

    return combined


def cleanCombinedResults(combined):
    df = combined.copy()
    removeEnsembleCols(df)
    setDatasetAsIndex(df)
    return df


def setDatasetAsIndex(df):
    df.set_index(DATASET_COL_NAME, drop=True, inplace=True)


def removeEnsembleCols(combined):
    removeColsContaining(combined, '-')   # Ensemble-<whateverClassifier>


def removeStatsCols(combined):
    # rip out the length + cardinality stats
    return combined.select(lambda s: '#' not in s, axis=1)


def extractStatsCols(combined):
    df = combined.copy()
    for colName in df.columns.values:
        if "#" not in colName:
            df.drop(colName, axis=1, inplace=True)
    return df
