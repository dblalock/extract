#!/bin/env/python

# import numpy as np
import pandas as pd
from pandas.tools.merge import concat
from sklearn.svm import SVC, LinearSVC
# from sklearn.qda import QDA
# from sklearn.lda import LDA

from datasets import datasets as ds
from utils.misc import nowAsString
from utils.learn import tryParams
from analyze.sota import rawSotaResults
from analyze.results_utils import CLASSIFIER_COL_NAME, DATASET_COL_NAME
from analyze.compare_algos import extractBestResults, avgRanks
from analyze.plot_results import plotErrs, plotZvals, plotPvals
# from analyze.viz_clusters import showClusters
from viz.clusters import *
# from viz.clusters import makeKMeans, makeMeanShift, makeSparseClusterer

def classify(name, Xtrain, Ytrain, Xtest, Ytest):
	rbfParams = [{'C': [.01, .1, 1, 10, 100]}]
	linearParams = [{'C': [.01, .1, 1, 10, 100]}]
	rbf = (SVC(), rbfParams)
	linearSVM = (LinearSVC(), linearParams)
	# classifiers = [rbf, linearSVM, LDA()] # lda is slow / hanging on medImgs
	classifiers = [rbf, linearSVM]
	# classifiers = [linearSVM]

	d = [(CLASSIFIER_COL_NAME, classifiers)]

	df = tryParams(d, Xtrain, Ytrain, Xtest, Ytest, cacheBlocks=False)
	df[DATASET_COL_NAME] = pd.Series([name] * df.shape[0])
	return df


# def cluster(X, windowLen):

def classifyDatasets():
	results = rawSotaResults()

	# for d in datasets.allUCRDatasets():
	# for d in datasets.tinyUCRDatasets():
	for d in datasets.smallUCRDatasets():
		print "------------------------"
		print d.name + '...'
		result = classify(d.name, d.Xtrain, d.Ytrain, d.Xtest, d.Ytest)
		results = concat([results, result], join='outer')

	best = extractBestResults(results)
	# results.to_csv('../results/results.csv')
	best.to_csv('../results/best.csv')
	results.to_csv('../results/classify_%s.csv' % (nowAsString(),))
	results.to_csv('../results/classify.csv')

	plotErrs(best, scoreName="Accuracy")
	plotZvals(best, lowIsBetter=False)
	plotPvals(best, lowIsBetter=False)

	ranks = avgRanks(best, lowIsBetter=False)
	print("mean rank: %g" % ranks.mean())
	print(ranks)


def clusterDatasets():
	ks = [4, 8, 16, 32]
	# ks = [-1]
	lengths = [8, 16, 32, 64]

	# algos = [makeKMeans]
	algos = [makeMaxLinkage]
	# algos = [makeSparseClusterer]
	# algos = [makeKMeans, makeMeanShift]
	# algos = [makeMeanShift]
	# algos = [makeKMeans, makeDBScan, makeAvgLinkage,
	# 		makeMeanShift, makeWard]

	# data = ds.allUCRDatasets()
	data = ds.smallUCRDatasets()
	data = filter(lambda d: d.name == "Trace", data)
	# for d in ds.tinyUCRDatasets()[:2]:
	# for d in ds.smallUCRDatasets():
	# for d in ds.allUCRDatasets():
	for d in data:
		for algo in algos:
			showDatasetClusters(d.name, d.Xtrain, d.Ytrain, d.Xtest, d.Ytest,
				ks, lengths, clusterFactory=algo, normSubSeqs=True)
		# break

if __name__ == '__main__':
	# classifyDatasets()
	clusterDatasets()

	# print ds.__dict__[]
