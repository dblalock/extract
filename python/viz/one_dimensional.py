#!/bin/env/python

import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import r_
from math import ceil

from viz_utils import saveCurrentPlot, colorForLabel, nameFromDir
from ..utils import sequence as seq
# from ..utils.arrays import meanNormalizeCols, stdNormalizeCols, zNormalizeCols
from ..transforms import *
from ..datasets import read_ucr as ucr

UCR_DATASETS_DIR = "~/Desktop/datasets/ucr_data"


# ================================================================ Utils

def getAllUCRDatasetDirs():
	datasetsPath = os.path.expanduser(UCR_DATASETS_DIR)
	files = os.listdir(datasetsPath)
	for i in range(len(files)):
		files[i] = os.path.join(datasetsPath, files[i])
	dirs = filter(os.path.isdir, files)
	return dirs


def ensureDirExists(dir):
	dirPath = os.path.expanduser(dir)
	if not os.path.exists(dirPath):
		os.mkdir(dirPath)


def znormalize(x):
	std = np.std(x)
	if std == 0:
		return x 	# don't die for constant arrays
	return (x - np.mean(x)) / std


# ================================================================ Plotting

def plotExamples(X, Y, maxPerClass, normalize, transforms):
	numPlottedForClasses = np.arange(max(Y) + 1)
	plt.figure()
	for row in range(X.shape[0]):
		# only plot a fixed number of examples of each
		# class so that the plots come out legible
		lbl = Y[row]
		if (numPlottedForClasses[lbl] > maxPerClass):
			continue
		numPlottedForClasses[lbl] += 1

		# possibly z-normalize
		data = X[row, :]
		if normalize:
			data = znormalize(data)

		if transforms:
			for transform in transforms:
				data = transform(data)

		plt.plot(data, color=colorForLabel(lbl))


def imgExamples(X, Y, maxPerClass=np.inf, transforms=None):
	MAX_EXAMPLES = 2000
	EXAMPLES_PER_COL = 400
	DEFAULT_INTER_CLASS_PADDING = 1
	FIG_HEIGHT = 10
	FIG_WIDTH = 12

	# if input too large, resample X
	if len(X) > MAX_EXAMPLES:
		idxs = np.linspace(0,len(X) - 1, MAX_EXAMPLES, dtype=np.int)
		X = X[idxs, :]

	# apply transforms, if present
	Xnew = []
	if transforms and len(transforms):
		for row in range(len(X)):
			Xt = X[row, :]
			for transform in transforms:
				Xt = transform(Xt)
			Xnew.append(Xt)
		X = np.array(Xnew)

	# x stats
	minVal = np.min(X)
	maxVal = np.max(X)
	exampleLen = len(X[0])

	# y stats
	classes = seq.uniqueElements(Y)
	classes = sorted(classes)
	numClasses = len(classes)

	# figure out how many columns to have
	fractionalCols = len(X) / float(EXAMPLES_PER_COL)
	numCols = max(1, int(ceil(fractionalCols)))

	# create padding to visually separate clases--thicker if it
	# falls on column boundaries
	paddingWidth = DEFAULT_INTER_CLASS_PADDING
	paddingVal = minVal
	if (numCols >= numClasses) and (numCols % numClasses == 0):
		paddingWidth *= 5
	padding = np.zeros((paddingWidth, exampleLen)) + paddingVal

	# split rows of X by class label
	examplesForClasses = seq.splitElementsBy(lambda i, el: Y[i], X)

	# concatenate all the data for each class, separated by a
	# few empty rows so we can distinguish the classes
	examplesWithPadding = np.zeros((1, exampleLen))
	for lbl in sorted(examplesForClasses.keys()):
		examples = np.asarray(examplesForClasses[lbl])
		examplesWithPadding = np.r_[examplesWithPadding,
			examples, padding]
	examplesWithPadding = examplesWithPadding[:(-paddingWidth), :]

	# split data into columns so we can see it better
	if numCols > 1:
		totalRows = len(examplesWithPadding)
		fig, axes = plt.subplots(1, numCols, figsize=(FIG_WIDTH, FIG_HEIGHT))
		for i, ax, in enumerate(axes):
			startIdx = int(totalRows / numCols * i)
			endIdx = int(totalRows / numCols * (i + 1))
			subset = examplesWithPadding[startIdx:endIdx, :]

			# hack for same scale everywhere
			subset[0, 0] = minVal
			subset[-1, -1] = maxVal

			ax.imshow(subset, aspect='auto')
			ax.set_yticklabels(())
	else:
		plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
		plt.imshow(examplesWithPadding, aspect='auto')

	plt.tight_layout()
	# plt.show()

def plotDataset(X, Y, name, img=False, normalizeCols=False,
	maxPerClass=np.inf, transforms=None):
	print("plotting dataset: " + name + "...")

	if normalizeCols:
		X = meanNormalizeCols(X)

	if img:
		imgExamples(X, Y, maxPerClass, transforms)
		plt.suptitle(name)
	else:
		plotExamples(X, Y, maxPerClass, normalize, transforms)
		plt.title(name)

	suffix = ""
	subdir = ""
	if img:
		suffix += "_img"
		subdir += "_img"
	if normalizeCols:
		suffix += "_colnormalized"
		subdir += "_colnormalized"
	if maxPerClass < np.inf:
		suffix += "_" + str(maxPerClass)
		subdir += "_" + str(maxPerClass)
	if transforms:
		for transform in transforms:
			name = '_' + str(transform.__name__)
			suffix += name
			subdir += name
	saveCurrentPlot(name, suffix=suffix, subdir=subdir)


def plotDatasetInDir(datasetDir, img=False, normalizeCols=False,
	maxPerClass=np.inf, transforms=None):
	# superimpose all the data in one plot, color-coded by class
	X, Y = readAllData(datasetDir)
	plotDataset(X, Y, nameFromDir(datasetDir), img, normalizeCols,
		maxPerClass, transforms)


# ================================================================ Main

def main():

	# d = '/Users/davis/Desktop/datasets/ucr_data/ItalyPowerDemand'
	# plotDatasetInDir(d, img=True)
	# # plotDatasetInDir(d, img=True, transforms=[downsample8, znormalize, sax8])
	# return

	# for datasetDir in getAllUCRDatasetDirs():
	# for i, datasetDir in enumerate(getAllUCRDatasetDirs()):
	for i, dataset in enumerate(ucr.getAllUCRDatasets()):
		# plotDatasetInDir(datasetDir,normalize=True)
		# plotDatasetInDir(datasetDir,normalize=False, maxPerClass=20)
		# plotDatasetInDir(datasetDir,normalize=True, maxPerClass=20)
		# plotDatasetInDir(datasetDir,normalize=True,
		# 	normalizeCols=True, maxPerClass=20)
		# plotDatasetInDir(datasetDir, normalize=True, transforms=[fftMag])
		# plotDatasetInDir(datasetDir, transforms=[fftMag], maxPerClass=20)
		# plotDatasetInDir(datasetDir, transforms=[fftPhase])
		# plotDatasetInDir(datasetDir, img=True)
		# plotDatasetInDir(datasetDir, img=True, transforms=[znormalize])
		# plotDatasetInDir(datasetDir, img=True, transforms=[znormalize, np.cumsum])
		# plotDatasetInDir(datasetDir, normalizeCols=True, img=True,
		# 	transforms=[znormalize])
		# plotDatasetInDir(datasetDir, img=True, transforms=[znormalize, firstDeriv])
		# plotDatasetInDir(datasetDir, img=True, transforms=[firstDeriv, znormalize])
		# if i > 5: return
		# plotDatasetInDir(datasetDir, img=True, transforms=[downsample2, znormalize])
		# plotDatasetInDir(datasetDir, img=True, transforms=[downsample4, znormalize])
		# plotDatasetInDir(datasetDir, img=True, transforms=[downsample8, znormalize])
		# plotDatasetInDir(datasetDir, img=True, transforms=[downsample4, znormalize, sax8])
		# plotDatasetInDir(datasetDir, img=True, transforms=[downsample8, znormalize, sax8])

		X = np.vstack((dataset.Xtrain, dataset.Xtest))
		Y = np.hstack((dataset.Ytrain, dataset.Ytest))
		plotDataset(X, Y, dataset.name, img=True, transforms=[resampleToLength64, znormalize, sax8])

if __name__ == '__main__':
	main()
