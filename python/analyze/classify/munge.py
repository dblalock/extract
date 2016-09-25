import numpy as np

# general-purpose data munging funcs

def deleteCols(df, cols):
    df.drop(cols, inplace=True, axis=1)


def renameColAtIdx(df, idx, newName):
    df.columns.values[idx] = newName


def stripColNames(df, chars=None):
    df.rename(columns=lambda x: x.strip(chars), inplace=True)


def sortColsByName(df):
    df.sort_index(axis=1, inplace=True)


def removeEmptyRowsAndCols(df):
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)


def removeColsContaining(df, str):
    for colName in df.columns.values:
        if str in colName:
            df.drop(colName, axis=1, inplace=True)


def makeColumnFirst(df, colName):
    colNames = list(df.columns.values)
    colNames.remove(colName)
    colNames.insert(0, colName)
    return df[colNames]


def toPrettyCsv(df, path):
    # df.to_csv(path, index=False, float_format='%.3f')
    df.to_csv(path, float_format='%.3f')


def groupXbyY(X, y):
    """Given an iterable X and a set of labels Y, returns a list whose elements
    are all the values of X that share a given label. Only tested with 2D
    arrays as X and 1D arrays as Y"""
    idxs = [np.where(y == val)[0] for val in np.unique(y).tolist()]
    return [X[idx] for idx in idxs]
