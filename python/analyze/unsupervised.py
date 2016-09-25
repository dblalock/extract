
import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from collections import namedtuple

from ..utils import files

join = os.path.join # abbreviation

FIG_SAVE_DIR = join('figs', 'paper')
FIG_SAVE_DIR_SCALABILITY = join(FIG_SAVE_DIR, 'scalability')
files.ensureDirExists(FIG_SAVE_DIR_SCALABILITY)

# ================================================================
# constants
# ================================================================

# ------------------------ paths

RESULTS_DIR = join('results', 'official')
SCALABILITY_RESULTS_DIR = join(RESULTS_DIR, 'scalability')

ACCURACY_SUBDIRS = ['dishwasher_groups', 'msrc', 'tidigits', 'ucr']
PAIRS_SUBDIRS = ['dishwasher_pairs', 'ucr_pairs', 'synthetic']
SCALABILITY_SUBDIRS = ['dishwasher','randwalk']

ACCURACY_DIRS = [join(RESULTS_DIR, subdir) for subdir in ACCURACY_SUBDIRS]
PAIRS_DIRS = [join(RESULTS_DIR, subdir) for subdir in PAIRS_SUBDIRS]
SCALABILITY_DIRS = [join(SCALABILITY_RESULTS_DIR, subdir) for subdir in SCALABILITY_SUBDIRS]

# ALGO_SUBDIRS = ['ff', 'minnen', 'mdl']

# ------------------------ columns

N_COL = 'cropDataLength'
M_MAX_COL = 'm_max'
M_SPAN_COL = 'm_span'

ALGORITHM_COL = 'algorithm'
DATASET_COL = 'datasetName'
TEST_COL = 'test'
SEED_COL = 'seed'

TIME_COL = 'time'
LENGTHS_COL = 'lengths'

# PERFORMANCE_MEASURE_COLS = ['f1_25', 'f1_50', 'iou', TIME_COL]
PERFORMANCE_MEASURE_COLS = ['_f1_25', '_f1_50']
SPLIT_RESULTS_BY_COL = DATASET_COL
INDEPENDENT_VAR_COL = ALGORITHM_COL

# ------------------------ user-visible names

DATASET_DISHWASHER = "Dishwasher"
DATASET_TIDIGTS = "TIDIGITS"
DATASET_MSRC = "MSRC-12"
DATASET_UCR = "UCR"
DATASET_UCR_PAIRS = "UCR Pairs"
DATASET_DISHWASHER_PAIRS = "Dishwasher Pairs"
DATASET_SYNTHETIC = "Synthetic"
DATASET_RANDWALK = "Random Walk"

ALGO_MINE = "Flock"
ALGO_MDL = "MDL"
ALGO_THRESH = "Dist"

# ------------------------ dataset info

DATASET_2_SIZE = {
	DATASET_DISHWASHER: 35,
	DATASET_MSRC: 594,
	DATASET_UCR: 50 * 20,
	DATASET_TIDIGTS: 300, # TODO get it exactly right, ideally
}

# ------------------------ plotting

# AXES_BG_COLOR = '#EAEAF2' # seaborn default purple
AXES_BG_COLOR = '#EDEDF6' # slightly lighter
# AXES_BG_COLOR = '#F0F0F8' # lighter
# AXES_BG_COLOR = '#F8F8FA' # lighterer
# AXES_BG_COLOR = '#FCFCFF' # lightest

# ================================================================
# types
# ================================================================

class MissingDataError(Exception):
	pass

_fields = [
	'color',
	'width',
	'style',
]
LineParams = namedtuple('LineParams', _fields)

ALGO_2_LINE_PARAMS = {
	ALGO_MDL: LineParams(sb.color_palette()[0], width=2, style='--'),
	ALGO_THRESH: LineParams(sb.color_palette()[1], width=3, style='-.'),
	ALGO_MINE: LineParams(sb.color_palette()[2], width=2, style='-'),
}

# ================================================================
# utility functions
# ================================================================

def readResults(dirs=ACCURACY_DIRS):
	"""XXX this just returns most recent results"""
	# allCsvs = []
	dfs = []
	for d in dirs:
		print "readResults(): searching through dataset {}".format(d)

		for algo in files.listSubdirs(d):
			print "readResults(): found algorithm {}".format(algo)

			algoDir = join(d, algo)
			paths = glob.glob(algoDir + '/*.csv')
			times = [os.path.getmtime(path) for path in paths]
			pathsOrder = np.argsort(times)[::-1]

			if not len(paths):
				print("readResults(): skipping empty dir {}".format(algoDir))
				continue

			# find results that didn't specify a subset of the data
			idx = 0
			while True:
				if idx >= len(pathsOrder):
					raise MissingDataError("All results in subdir "
						"{} used only a subset of examples!".format(algoDir))
				pathIdx = pathsOrder[idx]
				# print "--> reading df from path {}".format(paths[pathIdx])
				df = pd.DataFrame.from_csv(paths[pathIdx])

				name = os.path.basename(d)
				df[DATASET_COL] = name
				needsWhichExamples = ('ucr' in name) or ('synthetic' in name)
				whichExamples = df['whichExamples'][0]
				hasWhichExamples = hasattr(whichExamples, '__len__')

				# if this df has the 'whichExamples' field (and this dataset)
				# doesn't just always have that, read in the csv at the next
				# path and try that one instead
				if hasWhichExamples and not needsWhichExamples:
					idx += 1
					continue

				break

			df[ALGORITHM_COL] = algo
			# print "set dataset, algo to ", df[DATASET_COL].iloc[0], df[ALGORITHM_COL].iloc[0]
			# print
			dfs.append(df)

	return dfs


def makeTable(df, rowsCol, colsCol, dataCol):
	# df.set_index(rowsCol)

	uniqRowVals = pd.unique(df[rowsCol])
	uniqColVals = pd.unique(df[colsCol])

	# "rows col = ", df[rowsCol]
	# print "uniq row vals", uniqRowVals
	# print "uniq col vals", uniqColVals
	# print df[[rowsCol, colsCol, dataCol]]

	out = pd.DataFrame(index=uniqRowVals, columns=uniqColVals)
	for rowVal in uniqRowVals:
		for colVal in uniqColVals:
			rowsMatch = df[rowsCol] == rowVal
			colsMatch = df[colsCol] == colVal
			thisIdx = np.where(rowsMatch * colsMatch)[0][0]
			out.ix[rowVal][colVal] = df[dataCol][thisIdx]

	return out


def fmtFloat(flt):
	return "{:0.3f}".format(flt)


def fmtInt(flt):
	return "{0:d}".format(int(flt))


def replaceValuesInCol(df, colName, origValues, newValues):
	colValues = df[colName]
	for orig, new in zip(origValues, newValues):
		idxs = colValues == orig
		if colName in df.columns.values:
			df.loc[idxs, colName] = new
	return df


def fixDatasetNames(df):
	old2new = {
		'tidigits': DATASET_TIDIGTS,
		'tidigits_grouped_mfccs': DATASET_TIDIGTS,
		'dishwasher_groups': DATASET_DISHWASHER,
		'msrc': DATASET_MSRC,
		'ucr': DATASET_UCR,
		'ucr_short': DATASET_UCR,
		'ucr_pairs': DATASET_UCR_PAIRS,
		'dishwasher_pairs': DATASET_DISHWASHER_PAIRS,
		'synthetic': DATASET_SYNTHETIC,
		'dishwasher': DATASET_DISHWASHER,
		'randwalk': DATASET_RANDWALK}
	old, new = zip(*old2new.items())
	return replaceValuesInCol(df, DATASET_COL, old, new)


def fixAlgorithmNames(df):
	old = ['ff', 'mdl', 'minnen']
	new = [ALGO_MINE, ALGO_MDL, ALGO_THRESH]
	return replaceValuesInCol(df, ALGORITHM_COL, old, new)


def removeEveryOtherXTick(ax):
	lbls = ax.get_xticks().astype(np.int).tolist()
	for i in range(1, len(lbls), 2):
		lbls[i] = ''
	ax.set_xticklabels(lbls)


def removeAllButFirstAndLastXTick(ax):
	lbls = ax.get_xticks().astype(np.int).tolist()
	for i in range(1, len(lbls) - 1):
		lbls[i] = ''
	ax.set_xticklabels(lbls)


def removeEveryOtherYTick(ax):
	lbls = ax.get_yticks().astype(np.int).tolist()
	for i in range(1, len(lbls), 2):
		lbls[i] = ''
	ax.set_yticklabels(lbls)


def arrayFromString(s, dtype=np.float):
	"""string of form [1 3 2] -> array([1, 3, 2])"""
	splitStr = s.strip('[').strip(']').split()
	if dtype == np.int:
		splitStr = [int(s) for s in splitStr]
	elif dtype in [np.float, np.float32, np.float64]:
		splitStr = [float(s) for s in splitStr]
	elif dtype == np.object:
		pass
	else:
		raise ValueError("Cannot convert to array of "
			"unsupported type {}".format(dtype))
	return np.array(splitStr)


def cleanedResults(dirs=ACCURACY_DIRS):
	dfs = readResults(dirs=dirs)

	df = pd.concat(dfs, ignore_index=True)
	# print "df algo, dataset combos"
	# print df[[DATASET_COL, ALGORITHM_COL, 'time']]
	df = fixDatasetNames(df)
	df = fixAlgorithmNames(df)
	# print
	# print df[[DATASET_COL, ALGORITHM_COL, 'time']]

	return df


def printResultsTables(dirs=ACCURACY_DIRS):
	df = cleanedResults(dirs)

	# for old, new in zip(oldColNames, newColNames):
	# 	df.columns[old] = new

	# datasetNames = df[DATASET_COL]
	# whereTidigits = datasetNames == 'tidigits_grouped_mfccs'
	# df[DATASET_COL][whereTidigits] = 'tidigits'
	# whereUcr = datasetNames == 'ucr_short'
	# df[DATASET_COL][whereUcr] = 'UCR'
	# whereDishwasher = datasetNames == 'dishwasher_groups'

	# yep, this looks right
	# df.to_csv('/Users/davis/Desktop/tmp.csv')

	print df

	# print separate table for each measure
	for measureName in PERFORMANCE_MEASURE_COLS:
		tableDf = makeTable(df, rowsCol=DATASET_COL,
			colsCol=ALGORITHM_COL, dataCol=measureName)
		print
		print "---- {} ---- ".format(measureName)
		print tableDf

		floatFormat = fmtFloat
		if measureName == 'time':
			floatFormat = fmtInt

		print tableDf.to_latex(float_format=floatFormat)


def shiftLegend(leg, dx=0, dy=0):
	bb = leg.legendPatch.get_bbox()
	newX0 = bb.x0 + dx
	newX1 = bb.x1 + dx
	newY0 = bb.y0 + dy
	newY1 = bb.y1 + dy
	bb.set_points([[newX0, newY0], [newX1, newY1]])
	leg.set_bbox_to_anchor(bb)


# ================================================================
# scalability results
# ================================================================

def readScalabilityResultsInDir(datasetDir):
	datasetName = os.path.basename(datasetDir)

	dfs = []
	print "scalability(): searching through dir {}".format(datasetDir)
	for algo in files.listSubdirs(datasetDir): # for each algorithm
		print "--found algorithm {}".format(algo)
		algoDir = join(datasetDir, algo)
		for test in files.listSubdirs(algoDir): # for each test
			print "----found test {}".format(test)
			testDir = join(algoDir, test)
			paths = files.listFilesInDir(testDir, endswith='.csv', absPaths=True)
			# print "found paths: {}".format(paths)
			if not paths:
				continue
			# read last csv modified
			times = [os.path.getmtime(path) for path in paths]
			timesOrder = np.argsort(times)
			# idx = timesOrder[0]

			# combine data for different seeds; if same seed listed
			# more than once, take the later one
			seeds = set()
			dfsForSeeds = []
			for idx in timesOrder:
				df = pd.read_csv(paths[idx])
				try:
					seed = df[SEED_COL][0]
				except:
					seed = -1
					df[SEED_COL] = seed
				if seed not in seeds:
					seeds.add(seed)
					dfsForSeeds.append(df)
			df = pd.concat(dfsForSeeds, ignore_index=True)

			df[DATASET_COL] = datasetName
			df[ALGORITHM_COL] = algo
			df[TEST_COL] = test
			dfs.append(df)

	return dfs

# def readScalabilityResults(dirs=SCALABILITY_DIRS):
# 	dfs = []
# 	# for d in dirs: # TODO uncomment after we get randwalk results
# 	for d in dirs:
# 		dfs += readScalabilityResultsInDir(d)
# 	return dfs

def cleanScalabilityResultsInDir(d):
	# dfs = readScalabilityResults()
	dfs = readScalabilityResultsInDir(d)
	for df in dfs:
		if df[ALGORITHM_COL][0] == 'ff': # no lengths
			# print "skipping df with algo {}".format(df[ALGORITHM_COL][0])
			df[M_MAX_COL] = df['Lmax']
			df[M_SPAN_COL] = df['Lmax'] - df['Lmin']
		else:
			df[LENGTHS_COL] = df[LENGTHS_COL].apply(lambda s: arrayFromString(s, dtype=np.int))
			df[M_MAX_COL] = df[LENGTHS_COL].apply(np.max)
			df[M_SPAN_COL] = df[LENGTHS_COL].apply(len)

		df = fixDatasetNames(df)
		df = fixAlgorithmNames(df)

	return dfs


def plotScalabilityResultsForTest(dfs, test, colName, title, xlabel, savePath,
	ax=None, logxscale=False, logyscale=True):

	dfs = filter(lambda df: df['test'][0] == test, dfs)

	if ax is None:
		plt.figure()
		ax = plt.gca()

	lines = []
	labels = []
	for df in dfs:
		x = df[colName]
		y = df[TIME_COL] / 60. # convert to minutes
		algo = df[ALGORITHM_COL][0]
		p = ALGO_2_LINE_PARAMS[algo]
		line, = ax.plot(x, y, label=algo, color=p.color, lw=p.width, ls=p.style)
		ax.set_xlim([np.min(x), np.max(x)])

		lines.append(line)
		labels.append(algo)

	# seaborn tsplot version; confidence intervals like invisibly thin
	# around the lines, so not much point (although verify this once
	# all experiments run TODO)
	# df = pd.concat(dfs, ignore_index=True)
	# colors = dict([algo, p.color] for algo, p in ALGO_2_LINE_PARAMS.items()])
	# sb.tsplot(time=colName, value=TIME_COL, unit=SEED_COL,
	# 	condition=ALGORITHM_COL, data=df, ax=ax)
	# lines, labels = ax.get_legend_handles_labels()

	if logxscale:
		ax.set_xscale('log')
	if logyscale:
		ax.set_yscale('log')
	# x = df[colName]
	# ax.set_xlim([np.min(x), np.max(x)])

	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel("Runtime (min)")

	if savePath:
		plt.savefig(savePath)

	return lines, labels


def plotScalabilityResultsInDir(d, axes=None, saveDir=FIG_SAVE_DIR_SCALABILITY):
	dfs = cleanScalabilityResultsInDir(d)
	datasetName = dfs[0][DATASET_COL][0]

	# print '------------------------'
	# print "algo: {}\t test: {}".format(df[ALGORITHM_COL][0], df[TEST_COL][0])
	# if LENGTHS_COL in df.columns.values:
	# 	print "df lengths:"
	# 	print df[LENGTHS_COL]
	# print "n lengths"
	# print df[N_COL]
	# print "max lengths"
	# print df[M_MAX_COL]
	# print "num lengths"
	# print df[M_SPAN_COL]

	if axes is None:
		_, axes = plt.subplots(3, 1)

	test = 'n_length'
	title = "Runtime vs Time Series Length on {} Dataset".format(datasetName)
	xlabel = "Length of Time Series"
	path = join(saveDir, '{}-{}'.format(datasetName, test)) if saveDir else None
	plotScalabilityResultsForTest(dfs, test, N_COL, title, xlabel, path,
		# logxscale=True, ax=axes[0])
		ax=axes[0])

	test = 'm_length'
	title = "Runtime vs Search Length on {} Dataset".format(datasetName)
	xlabel = "Maximum Length Searched"
	path = join(saveDir, '{}-{}'.format(datasetName, test)) if saveDir else None
	plotScalabilityResultsForTest(dfs, test, M_MAX_COL, title, xlabel, path,
		ax=axes[1])

	test = 'm_span'
	title = "Runtime vs Number of Lengths on {} Dataset".format(datasetName)
	xlabel = "Number of lengths tested"
	path = join(saveDir, '{}-{}'.format(datasetName, test)) if saveDir else None
	return plotScalabilityResultsForTest(dfs, test, M_SPAN_COL, title, xlabel, path,
		ax=axes[2])
	# ^ return the above as a hack to get the lines and labels for legend()


def plotScalabilityResultsHorizontal():
	assert(len(SCALABILITY_DIRS) == 2) # assuming just dishwasher and randwalk

	sb.set_context("talk")
	mpl.rcParams['axes.facecolor'] = AXES_BG_COLOR
	mpl.rcParams['xtick.major.pad'] = 4
	# mpl.rcParams['xtick.major.size'] = 6
	mpl.rcParams['axes.edgecolor'] = 'gray'
	mpl.rcParams['axes.linewidth'] = .5

	# mpl.rcParams['font.size'] = 18 # seemingly the only way to make legend bigger
	# fig = plt.figure(figsize=(10./1.2, 7./1.2))
	fig = plt.figure(figsize=(10./1.3, 8./1.3))

	nRows = 2
	nCols = 3
	gs = mpl.gridspec.GridSpec(nRows, nCols)
	axes = []
	axTopLeft = fig.add_subplot(gs[0, 0])
	axBottomLeft = fig.add_subplot(gs[1, 0])
	axes.append([axTopLeft, axBottomLeft])
	for i in range(1, nCols):
		axTop = fig.add_subplot(gs[0,i])
		axBottom = fig.add_subplot(gs[1,i], sharey=axTopLeft)
		plt.setp(axTop.get_yticklabels(), visible=False)
		plt.setp(axBottom.get_yticklabels(), visible=False)
		# axBottom = fig.add_subplot(gs[1,i], sharey=axTop)

		axes.append([axTop, axBottom])

	axes = np.array(axes).T
	flatAxes = axes.flatten()

	for ax in axes[0]:
		plt.setp(ax.get_xticklabels(), visible=False)

	# for i, d in enumerate(SCALABILITY_DIRS[1:]): # ignore dishwasher for now TODO remove
	for i, d in enumerate(SCALABILITY_DIRS):
		lines, labels = plotScalabilityResultsInDir(d, axes=axes[i,:], saveDir=None)

	for ax in flatAxes:
		sb.despine(left=True, ax=ax)
		ax.set_title("")
		ax.set_xlabel("")
		ax.set_ylabel("")
		# ax.spines['bottom'].set_visible(True)
		# ax.xaxis.tick_bottom()

	Y_LABEL_FONT_SIZE = 17
	LABEL_PADDING = 15
	axes[0, 0].set_ylabel("Dishwasher Dataset", labelpad=LABEL_PADDING,
		fontsize=Y_LABEL_FONT_SIZE)
	axes[1, 0].set_ylabel("Random Walk Dataset", labelpad=LABEL_PADDING,
		fontsize=Y_LABEL_FONT_SIZE)

	# last col has y labels on right side
	for ax in axes[:,-1]:
		ax.yaxis.set_label_position('right')
		# ax.set_ylabel("Runtime (s)", labelpad=LABEL_PADDING)
		ax.set_ylabel("Runtime (min)", labelpad=LABEL_PADDING,
			fontsize=Y_LABEL_FONT_SIZE)

	# label each col with t
	X_LABEL_PADDING = 7
	axes[1, 0].set_xlabel("Time Series\nLength", labelpad=X_LABEL_PADDING)
	axes[1, 1].set_xlabel("Max Search\nLength", labelpad=X_LABEL_PADDING)
	axes[1, 2].set_xlabel("Number of\nLengths Searched", labelpad=X_LABEL_PADDING)

	# clean up crowded x tick labels
	for ax in axes[:2,:2].flatten():
		removeEveryOtherXTick(ax)

	# set legend
	for ax in flatAxes:
		legend = ax.legend()
		if legend:
			legend.remove()
	# note that ncols is cols in the legend, not cols to span
	leg = plt.figlegend(lines, labels, loc='lower center', ncol=3, labelspacing=0.,
		fontsize=16)
	shiftLegend(leg, dy=-.03)

	# configure and save overall fig
	plt.tight_layout()
	# plt.subplots_adjust(bottom=.18, hspace=.17)
	plt.subplots_adjust(bottom=.18, hspace=.17, wspace=.15)
	plt.suptitle("  Runtime vs Dataset and Search Attributes",
		fontsize=20, fontweight='bold')
	plt.subplots_adjust(top=.9)

	path = join(FIG_SAVE_DIR_SCALABILITY, 'scalability.pdf')
	plt.savefig(path)
	# plt.show()


def plotScalabilityResults():
	assert(len(SCALABILITY_DIRS) == 2) # assuming just dishwasher and randwalk

	sb.set_context("talk")
	mpl.rcParams['axes.facecolor'] = AXES_BG_COLOR

	# create figure and axes
	fig = plt.figure(figsize=(7, 10))
	nRows = 3
	nCols = 2
	gs = mpl.gridspec.GridSpec(nRows, nCols)
	axes = []
	for i in range(nRows):
		axLeft = fig.add_subplot(gs[i,0])
		axRight = fig.add_subplot(gs[i,1], sharey=axLeft)
		plt.setp(axRight.get_yticklabels(), visible=False)
		axes.append([axLeft, axRight])
	# axes = [gs[i,j] for i in range(nRows) for j in range(nCols)]
	axes = np.array(axes)
	flatAxes = axes.flatten()

	# populate axes using other func
	for i, d in enumerate(SCALABILITY_DIRS):
		lines, labels = plotScalabilityResultsInDir(d, axes=axes[:,i], saveDir=None)

	# clear titles and x labels
	for ax in flatAxes:
		sb.despine(left=True, ax=ax)
		ax.set_title("")
		ax.set_xlabel("")

	# label each column
	axes[0, 0].set_title("Dishwasher Dataset")
	axes[0, 1].set_title("Random Walk Dataset")

	# label each row
	for ax in axes[:,1]:
		ax.yaxis.set_label_position('right')
	LABEL_PADDING = 15
	axes[0, 1].set_ylabel("Time Series\nLength", labelpad=LABEL_PADDING)
	axes[1, 1].set_ylabel("Max Search\nLength", labelpad=LABEL_PADDING)
	axes[2, 1].set_ylabel("Number of\nLengths Searched", labelpad=LABEL_PADDING)

	# fix crowded x tick labels
	for ax in axes[:2,:2].flatten():
		removeEveryOtherXTick(ax)

	# set legend
	for ax in flatAxes:
		legend = ax.legend()
		if legend:
			legend.remove()
	# note that ncols is cols in the legend, not cols to span
	plt.figlegend(lines, labels, loc='lower center', ncol=3, labelspacing=0.)

	# setup overall plot
	plt.tight_layout()
	plt.subplots_adjust(bottom=.08, hspace=.17)
	plt.suptitle("  Runtime vs Dataset and Search Attributes",
		fontsize=16, fontweight='bold')
	plt.subplots_adjust(top=.92)

	# save fig
	path = join(FIG_SAVE_DIR_SCALABILITY, 'scalability.pdf')
	plt.savefig(path)


# ================================================================
# accuracy results
# ================================================================

def plotAccuracyResults():
	df = cleanedResults(dirs=ACCURACY_DIRS)

	# ================================ parse prec, rec, f1 stats
	# we have column names that look like '_f1_25' for f1 score at iou
	# threshold of .25
	# TODO make this section its own function

	colNames = df.columns.values
	threshStrs = [name.split('_')[2] for name in colNames if name[:3] == '_f1']

	# parse thresholds into floats
	threshs = np.array([float(thresh)/100 for thresh in threshStrs])

	# ensure that we order values read from keys by increasing iou
	# threshold (subsequent code computes keys from threshStrs)
	threshSortIdxs = np.argsort(threshs)
	threshs = threshs[threshSortIdxs]
	threshStrs = [threshStrs[idx] for idx in threshSortIdxs]

	whichCols = [DATASET_COL, ALGORITHM_COL, 'time']
	whichCols += ['_f1_' + str(iou) for iou in range(5, 26, 5)]
	df2 = df[whichCols]
	print "df algo, dataset combos"
	print df2

	measureNames = ['prec', 'rec', 'f1']
	for measureName in measureNames:
		prefix = '_' + measureName + '_'
		keys = [prefix + s for s in threshStrs]
		keys_std = ['_std' + k for k in keys]

		measureKey = measureName
		measureStdKey = measureName + '_std'

		# collect the value of this measure (eg, f1 score) and its
		# standard deviation across all the keys for this row
		valsForMeasure = []
		valStdsForMeasure = []
		for i in range(len(df)):
			vals = df[keys].iloc[i]
			stds = df[keys_std].iloc[i]

			valsForMeasure.append(vals)
			valStdsForMeasure.append(stds)

			# print "does f1 match?"
			f1_true = df[keys[0]].iloc[i]
			f1_val = vals[0]
			if f1_true != f1_val:
				print "{} {}: val_true, val_read:".format(
					df[DATASET_COL].iloc[i], df[ALGORITHM_COL].iloc[i])
				print f1_true, f1_val
				assert(f1_true == f1_val)

		# write the value + its std across all rows in new cols
		df[measureKey] = pd.Series(valsForMeasure)
		df[measureStdKey] = pd.Series(valStdsForMeasure)

	# ================================ plot f1 vs iou thresh

	# sb.set_context("talk")
	mpl.rcParams['axes.facecolor'] = AXES_BG_COLOR
	mpl.rcParams['font.size'] = 14
	mpl.rcParams['xtick.major.pad'] = 6
	# mpl.rcParams['xtick.major.size'] = 6
	# mpl.rcParams['xtick.major.size'] = 0

	fig, axes = plt.subplots(2, 2, figsize=(8,6))
	flatAxes = axes.flatten()

	# pick subplot for each dataset
	dataset2Ax = {
		DATASET_DISHWASHER: axes[0, 0],
		DATASET_MSRC: axes[1, 0],
		DATASET_TIDIGTS: axes[0, 1],
		DATASET_UCR: axes[1, 1]
	}
	ax2dataset = dict([(v, k) for k, v in dataset2Ax.items()])

	# configure axes
	for ax in flatAxes:
		sb.despine(left=True, ax=ax)
		ax.set_xlim([0, 1])
		ax.set_ylim([0, 1])

	# label axes
	sz = 16
	LABEL_PADDING = 5
	lbl = "F1 Score"
	axes[0, 0].set_ylabel(lbl, labelpad=LABEL_PADDING, fontsize=sz)
	axes[1, 0].set_ylabel(lbl, labelpad=LABEL_PADDING, fontsize=sz)
	LABEL_PADDING = 7
	lbl = "Overlap to Match (IOU)"
	axes[1, 0].set_xlabel(lbl, labelpad=LABEL_PADDING, fontsize=sz)
	axes[1, 1].set_xlabel(lbl, labelpad=LABEL_PADDING, fontsize=sz)

	# set titles for subplots (to the dataset names)
	for ax in flatAxes:
		dataset = ax2dataset[ax]
		ax.set_title(dataset, fontsize=18)

	# ensure that algos will correspond to the same lines in all plots
	# df.sort(ALGORITHM_COL, axis=0, inplace=True) # bad! scrambles everything

	# actually plot the data
	for i in range(len(df)): # for each algo-dataset combo
		dataset = df[DATASET_COL][i]
		algo = df[ALGORITHM_COL][i]
		vals = df['f1'].iloc[i].as_matrix()
		stds = df['f1_std'].iloc[i].as_matrix()
		print "dataset, algo", dataset, algo

		# make lines visually distinguishable (we're already slightly better,
		# and we are of course *not* modifying anything in the results
		# spreadsheets)
		if dataset == DATASET_MSRC:
			if algo == ALGO_MDL:
				vals -= .01
			elif algo == ALGO_MINE:
				vals += .01

		# compute 95% confidence interval
		stds *= 1.96 / np.sqrt(DATASET_2_SIZE[dataset])
		upper, lower = vals + stds, vals - stds

		p = ALGO_2_LINE_PARAMS[algo]

		ax = dataset2Ax[dataset]
		ax.plot(threshs, vals, label=algo, color=p.color, lw=p.width, ls=p.style)

		# color fill for conf interval
		ax.fill_between(threshs, upper, lower, color=p.color, alpha=.25)

		# vertical lines for conf interval
		# x = np.copy(threshs) + .01 * ((i % 3) - 1)
		# # ^ hack to prevent overlap for different algos
		# for x, l, u in zip(x, lower, upper):
		# 	ax.plot([x, x], [l, u], color=p.color, lw=1.5)

		# skinny lines for conf interval
		# ax.plot(threshs, lower, color=p.color, lw=1, ls='-')
		# ax.plot(threshs, upper, color=p.color, lw=1, ls='-')

	lines, labels = ax.get_legend_handles_labels()
	plt.figlegend(lines, labels, loc='lower center', ncol=3, labelspacing=0)
	# leg = plt.figlegend(lines, labels, loc='lower center', ncol=3, labelspacing=0)

	plt.suptitle("F1 Score vs Match Threshold",
		fontsize=20, fontweight='bold')
	plt.tight_layout()
	plt.subplots_adjust(top=.88, bottom=.21, hspace=.4)

	# # move legend to bottom edge of fig
	# plt.draw()
	# bb = leg.legendPatch.get_bbox().inverse_transformed(fig.transFigure)
	# yOffset = .1
	# bb.set_points([[bb.x0, bb.y0 + yOffset],[bb.x1, bb.y1 + yOffset]])
	# leg.set_bbox_to_anchor(bb)

	plt.savefig(join(FIG_SAVE_DIR, 'acc.pdf'), bbox_inches='tight')
	plt.show()

	# df.to_csv('/Users/davis/Desktop/tmp.csv')


# ================================================================
# Main
# ================================================================

if __name__ == '__main__':
	sb.set_context("poster")
	# cleanResults()
	# cleanResults(dirs=PAIRS_DIRS)
	plotAccuracyResults() # this one in paper

	# printResultsTables()

	# readScalabilityResults()
	# cleanScalabilityResults()
	# plotScalabilityResults()
	# plotScalabilityResultsHorizontal() # this one in paper

	# d = SCALABILITY_DIRS[0]
	# dfs = cleanScalabilityResultsInDir(d)
	# print [df.columns.values for df in dfs]




