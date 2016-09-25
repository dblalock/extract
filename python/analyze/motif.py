
import os
import glob
import numpy as np
import pandas as pd

join = os.path.join # abbreviation

RESULTS_DIR = join('results', 'official')
SCALABILITY_RESULTS_DIR = join(RESULTS_DIR, 'scalability')

ACCURACY_SUBDIRS = ['dishwasher_groups', 'msrc', 'tidigits', 'ucr']
PAIRS_SUBDIRS = ['dishwasher_pairs', 'ucr_pairs', 'synthetic']
SCALABILITY_SUBDIRS = ['dishwasher','randwalk']

ACCURACY_DIRS = [join(RESULTS_DIR, subdir) for subdir in ACCURACY_SUBDIRS]
PAIRS_DIRS = [join(RESULTS_DIR, subdir) for subdir in PAIRS_SUBDIRS]
SCALABILITY_DIRS = [join(SCALABILITY_RESULTS_DIR, subdir) for subdir in SCALABILITY_SUBDIRS]


def readAccuracyResults():
	"""XXX this just returns most recent results"""
	dirs = ACCURACY_DIRS
	# allCsvs = []
	dfs = []
	for d in dirs:
		# print d + '/*/*.csv'
		paths = glob.glob(d + '/*/*.csv')
		times = [os.path.getmtime(path) for path in paths]
		timesDescending = sorted(times)[::-1]
		# dfs = [pd.Dataframe.from_csv(path) for path in paths]

		idx = 0
		while True:
			print "reading df from path {}".format(paths[idx])

			df = pd.DataFrame.from_csv(paths[idx])
			name = df['datasetName'][0]
			needsWhichExamples = ('ucr' in name) or ('synthetic' in name)
			whichExamples = df['whichExamples'][0]
			if whichExamples and not needsWhichExamples:
				idx += 1
				continue

			break

			# if (df['whichExamples'] and
			# dfs.append(df)

		print df



if __name__ == '__main__':
	readAccuracyResults()
