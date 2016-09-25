
import numpy as np
import pandas as pd
from pandas.tools.merge import concat
from scipy.stats import ranksums

from munge import toPrettyCsv
# from utils import *
from results_utils import *

COMBINED_CSV_FILE = 'combined.csv'
ERRS_CSV_FILE = 'errRates.csv'
RANK_CSV_FILE = 'combined_rank.csv'
ZVALS_CSV_FILE = 'combined_zvals.csv'
PVALS_CSV_FILE = 'combined_pvals.csv'

def extractErrRates(combined, onlySharedDatasets=True):
    # df = cleanCombinedResults(combined)
    df = removeStatsCols(combined)     # remove length / cardinality stats
    if DATASET_COL_NAME in df.columns:
        setDatasetAsIndex(df)
    if onlySharedDatasets:
        df.dropna(axis=0, how='any', inplace=True)
    return df

def computeRanks(errRates, lowIsBetter=True, onlyFullRows=True):
    if onlyFullRows:
        errRates = errRates.dropna(axis=0, how='any', inplace=False)
    return errRates.rank(axis=1, numeric_only=True, ascending=lowIsBetter)
    # return errRates.copy().rank(axis=1, numeric_only=True)


def avgRanks(errRates, lowIsBetter=True, onlyFullRows=True):
    ranks = computeRanks(errRates, lowIsBetter=lowIsBetter,
        onlyFullRows=onlyFullRows)
    # return ranks.mean(axis=0, skipna=True)
    return ranks.mean(axis=0)


def computeRankSumZvalsPvals(errRates, lowIsBetter=True):
    ranks = computeRanks(errRates, onlyFullRows=False)

    # compute the ranked sums test p-value between different classifiers
    numClassifiers = errRates.shape[1]
    dims = (numClassifiers, numClassifiers)
    zvals = np.empty(dims)
    pvals = np.empty(dims)
    for i in range(numClassifiers):
        zvals[i, i] = 0
        pvals[i, i] = 1
        for j in range(i+1, numClassifiers):
            x = errRates.iloc[:, i]
            y = errRates.iloc[:, j]

            # compare using all datasets they have in common
            rowsWithoutNans = np.invert(np.isnan(x) + np.isnan(y))
            x = x[rowsWithoutNans]
            y = y[rowsWithoutNans]

            zvals[i, j], pvals[i, j] = ranksums(y, x) # cols are indep var
            zvals[j, i], pvals[j, i] = -zvals[i, j], pvals[i, j]

    classifierNames = ranks.columns.values
    zvals = pd.DataFrame(data=zvals, index=classifierNames,
        columns=classifierNames)
    pvals = pd.DataFrame(data=pvals, index=classifierNames,
        columns=classifierNames)
    return zvals, pvals

def extractBestResultsRows(df,
        objectiveColName=ACCURACY_COL_NAME,
        classifierColName=CLASSIFIER_COL_NAME,
        datasetColName=DATASET_COL_NAME):
    """
    df: a dataframe with columns for {accuracy (or another objective
        function), Dataset, Classifier}, and possibly others

    return: a dataframe of only the rows containing the highest accuracy
    for each classifier
    """
    df = df.copy()

    # ensure that all idxs are unique...everything breaks otherwise
    df.index = np.arange(df.shape[0])

    datasetCol = df[datasetColName]
    classifiersCol = df[classifierColName]
    uniqDatasets = datasetCol.unique()
    uniqClassifiers = classifiersCol.unique()

    # print uniqDatasets
    # print uniqClassifiers
    # return

    # allRows = pd.DataFrame()
    keepIdxs = []
    for dset in uniqDatasets:
        dsetDf = df[df[datasetColName] == dset]
        # print dsetDf
        for classifier in uniqClassifiers:
            # classifierDf = df[(datasetCol == dset) * (classifiersCol == classifier)]
            # classifierDf = df[df[classifierColName] == classifier]
            classifierDf = dsetDf[dsetDf[classifierColName] == classifier]

            # print classifier
            # print classifierDf
            assert(classifierDf.empty or classifierDf.shape[0] == 1)
            # continue

            if classifierDf.empty:
                continue

            idx = classifierDf[ACCURACY_COL_NAME].idxmax(skipna=True)
            if np.isnan(idx):
                continue

            try:    # deal with ties by taking first element
                idx = idx[0]
            except:
                pass

            keepIdxs.append(idx)

            # if not np.isnan(idx):
            # row = classifierDf.iloc(idx)
            # print row
            # allRows = concat([allRows, row], join='outer')
            # keepIdxs.append(idx)
        # break

    # print keepIdxs
    return df.iloc[sorted(keepIdxs)]

def extractBestResults(df,
        objectiveColName=ACCURACY_COL_NAME,
        classifierColName=CLASSIFIER_COL_NAME,
        datasetColName=DATASET_COL_NAME):
    """
    df: a dataframe with columns for {accuracy (or another objective
        function), Dataset, Classifier}, and possibly others

    return: a dataframe indexed by Dataset, with one column for each
    classifier; entries in the classifier column are equal to the highest
    result of that classifier in objectiveColName
    """
    best = extractBestResultsRows(df, objectiveColName,
        classifierColName, datasetColName)

    uniqDatasets = best[datasetColName].unique()
    uniqClassifiers = best[classifierColName].unique()

    colNames = [datasetColName]
    df = pd.DataFrame(np.asarray(uniqDatasets), columns=colNames)

    for classifier in uniqClassifiers:
        clsDf = best[best[classifierColName] == classifier]

        # dont care about other parameters and dont need a column
        # of just the classifier name
        clsDf = clsDf[[datasetColName, objectiveColName]]

        # rename the objective column to the name of this classifier
        # so each classifier will have its own column
        clsDf.columns = [datasetColName, classifier]

        df = df.merge(clsDf, on=datasetColName, how='outer')

    return df

def main():
    combined = buildCombinedResults()
    toPrettyCsv(combined, COMBINED_CSV_FILE)
    combined = cleanCombinedResults(combined)

    errRates = extractErrRates(combined)
    toPrettyCsv(errRates, ERRS_CSV_FILE)

    ranks = computeRanks(errRates)
    toPrettyCsv(ranks, RANK_CSV_FILE)
    print ranks.sum(axis=0)

    zvals, pvals = computeRankSumZvalsPvals(errRates)
    toPrettyCsv(zvals, ZVALS_CSV_FILE)
    toPrettyCsv(pvals, PVALS_CSV_FILE)

if __name__ == '__main__':
    main()
