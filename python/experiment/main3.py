#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from ..algo import ff11 as ff
from ..datasets import datasets
from ..utils import arrays as ar
from ..viz.ff import plotFFOutput

# from consts import * # because so many of these...
import consts
from estimators import DimsAppender, FFExtractor, MotifExtractor

def main():
	sb.set_style("whitegrid", {'axes.grid': False})

	np.random.seed(12345)

	whichExamples = np.arange(6) # tidigits example 5 is good
	tsList = datasets.loadDataset(datasets.TIDIGITS, whichExamples=whichExamples)

	# uncomment any of these to use a different dataset
	# tsList = datasets.loadDataset(datasets.DISHWASHER, whichExamples=whichExamples)
	# tsList = datasets.loadDataset(datasets.MSRC, whichExamples=whichExamples)
	# tsList = datasets.loadDataset(datasets.UCR, whichExamples=[0])

	Lmin, Lmax = 1./20, 1./10 # fractions of time series length

	# appender = DimsAppender(numAdversarialDims=1, numNoiseDims=10)
	appender = DimsAppender(numAdversarialDims=2, numNoiseDims=1)
	tsList = appender.transform(tsList)

	ts = tsList[-1]
	ts.data = ar.zNormalizeCols(ts.data)
	startIdxs, endIdxs, model, featureMat, featureMatBlur = ff.learnFFfromSeq(
		ts.data, Lmin, Lmax)

	plotFFOutput(ts, startIdxs, endIdxs, featureMat, model)

	plt.show()

if __name__ == '__main__':
	main()
