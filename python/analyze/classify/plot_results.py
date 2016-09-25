
import matplotlib.pyplot as plt

from colormap_utils import remappedColorMap
from results_utils import *
from compare_algos import extractErrRates, computeRankSumZvalsPvals

def plotDf(df, xlabel="Classifier", ylabel="Dataset", title=None,
	symmetricAboutMean=False, cmap='RdBu'):

	data = df.values
	if symmetricAboutMean:
		mean = data.mean()
		bound = max(abs(data.max() - mean), abs(data.min() - mean))
		lower = mean - bound
		upper = mean + bound
	else:
		lower = data.min()
		upper = data.max()

	plt.rcParams["font.size"] = 17
	# plt.figure(figsize=(10, 10))
	fig, ax = plt.subplots(figsize=(10, 10))
	p = ax.pcolormesh(data, cmap=cmap, vmin=lower, vmax=upper)
	plt.colorbar(p, ax=ax)

	plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
	plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns,
		fontsize=13, rotation=80)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)

	plt.autoscale(True, tight=True)
	plt.tight_layout()
	plt.show()


def plotErrs(df, scoreName="Error Rate", onlySharedDatasets=True):
	df = extractErrRates(df, onlySharedDatasets=onlySharedDatasets)
	print("------------------------")
	print("Err Rates:")
	print("------------------------")
	print(1 - df.mean(axis=1))
	print(1 - df.mean(axis=0))

	title = "%s for Different Classifiers" % scoreName
	plotDf(df, title=title)


def plotZvals(df, lowIsBetter=True):
	errs = extractErrRates(df)
	zvals, pvals = computeRankSumZvalsPvals(errs, lowIsBetter=lowIsBetter)
	plotDf(zvals, title="Rank-Sum Test Z Values", symmetricAboutMean=True)


def plotPvals(df, lowIsBetter=True):
	errs = extractErrRates(df)
	zvals, pvals = computeRankSumZvalsPvals(errs, lowIsBetter=lowIsBetter)

	# make a colormap that's red for p < 5%
	minVal = pvals.min().min()
	maxVal = pvals.max().max()
	cutoff = (np.log10(1.96) - minVal) / (maxVal - minVal)
	cmap = remappedColorMap(plt.cm.RdBu, midpoint=cutoff)

	plotDf(np.log10(pvals), title="Log10 Rank-Sum Test P Values", cmap=cmap)


def main():
	combined = buildCombinedResults()
	setDatasetAsIndex(combined)

	sharedErrs = combined.copy()
	removeEnsembleCols(sharedErrs)

	plotErrs(sharedErrs)
	plotZvals(combined)
	plotPvals(combined)


if __name__ == '__main__':
	main()

