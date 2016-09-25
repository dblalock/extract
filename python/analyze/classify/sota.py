
# make a giant pandas dataframe I can combine with my own crap

import pandas as pd

from results_utils import *
from munge import deleteCols

def datasetStats():
	# note that these cols probably have a lot of nans
	df = buildCombinedResults()
	return extractStatsCols(df)


def rawSotaResults():
	# start with a df with one col of dataset names and n-1 cols of
	# classifier err rates
	errs = buildCombinedResults()

	# pull out set of datasets
	datasets = errs[DATASET_COL_NAME]

	# get a df with just the (sorted) classifier cols
	errs = removeStatsCols(errs)
	deleteCols(errs, DATASET_COL_NAME)
	errs.reindex_axis(sorted(errs.columns), axis=1)

	allRows = []
	for colName in errs.columns:
		# print colName
		col = errs[colName]
		for i, errRate in enumerate(col):
			allRows.append({
				DATASET_COL_NAME: datasets.iloc[i],
				ACCURACY_COL_NAME: 1.0 - errRate,
				CLASSIFIER_COL_NAME: colName,
			})

	union = pd.DataFrame.from_records(allRows)
	union = union[RESULT_COL_NAMES]
	return union


if __name__ == '__main__':
	print rawSotaResults()
