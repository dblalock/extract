#!/usr/env/python

import numpy as np
from scipy import signal, stats

def znormalize(x):
	std = np.std(x)
	if std == 0:
		return x 	# don't die for constant arrays
	return (x - np.mean(x)) / std

def fftMag(v):
	cmplx = np.fft.fft(v)
	return np.abs(cmplx)

def fftPhase(v):
	return np.angle(np.fft.fft(v))

def firstDeriv(v):
	return v[1:] - v[:-1]

def downsampleBy(v, factor):
	order = 8 	# default value for decimate() anyway
	minSamples = 5
	if len(v) < (minSamples * factor):
		factor = int(len(v) / minSamples)
	v = signal.decimate(v, factor, n=order)
	if len(v) >= (order + minSamples):
		return v[(order - 1):]
	return v

def downsample2(v):
	return downsampleBy(v, 2)

def downsample4(v):
	return downsampleBy(v, 4)

def downsample8(v):
	return downsampleBy(v, 8)

def resampleToLengthN(v, n):	# can't guarantee length = exactly n
	if len(v.shape) == 1:
		factor = int(len(v) / float(n))
		if factor <= 1:
			return v
		# return downsampleBy(v, factor)
		return signal.resample(v, n)
	elif len(v.shape) == 2:	# resample each row
		rows = map(lambda row: resampleToLengthN(row.flatten(), n), v)
		return np.vstack(rows)
	else:
		print("Error: can't resample tensor to length N!")

def resampleToLength64(v):
	return resampleToLengthN(v, 64)

def saxN(v, n):
	quantiles = stats.norm.ppf(np.arange(1. / n, 1, 1. / n))
	v = znormalize(v)
	z = np.tile(v,((n - 1), 1))  # only n-1 levels, cuz extremes = inf
	greater = z.T > quantiles
	return np.sum(greater, axis=1)

def sax8(v):
	return saxN(v, 8)

