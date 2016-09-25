#!/usr/env/python

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

from ..datasets import synthetic as synth
from ..utils.arrays import zNormalizeRows

def tryTriangle():
	l = 3
	n = 100
	m = 60
	startIdx = 25
	nIters = 30

	# X = synth.randconst((l,n))
	X = synth.randwalk((l,n))
	for i in range(len(X)):
		inst = synth.bell(m)
		# inst = synth.cylinder(m)
		# inst = synth.sines(m)
		synth.embedSubseq(X[i], inst, startIdx)

	X = zNormalizeRows(X)

	# plt.plot(X.T) # yep, looks right
	# plt.show()

	# filt = sig.gaussian(3, std=1)
	# filt = np.ones(5) / 5
	filt = sig.hamming(3)

	# w0 = 1. / n
	u = X[0]
	uSim = .5
	# u = np.mean(X, axis=0)

	# w = np.random.rand(n) * w0
	# w = np.zeros(n) + w0
	w = np.ones(n)
	w = w / np.linalg.norm(w)
	lurn = .1

	ax1 = plt.subplot2grid((3,1), (0,0))
	ax2 = plt.subplot2grid((3,1), (1,0))
	ax3 = plt.subplot2grid((3,1), (2,0))
	# ax1 = plt.subplot2grid((2,3), (0,0), colspan=2)
	# ax2 = plt.subplot2grid((2,3), (1,0), colspan=2)
	ax1.autoscale(tight=True)
	ax2.autoscale(tight=True)
	ax3.autoscale(tight=True)

	ax1.set_title("Learned weights")
	ax2.set_title("Input signals")
	ax3.set_title("Motif")
	ax2.plot(X.T)

	# what can we get just looking at avg differences for each feature?
	# allDiffs = X - u
	# allDists = allDiffs * allDiffs
	# allSims = 1 - allDists
	# # allSims = np.max(np.zeros(allSims.shape, allSims)) # clamp at 0
	# allSims = np.maximum(0, allSims) # clamp at 0
	# uSim = np.mean(allSims)
	# print np.max(allSims)
	# uSims = np.mean(allSims, axis=0)
	# meanSims = np.mean(allSims, axis=0)
	# meanSims = np.convolve(meanSims, filt)
	# ax1.plot(meanSims, linewidth=5)
	# plt.show()
	# return

	for it in range(nIters):
		order = np.random.permutation(l)
		# order = range(l)

		for idx in order:
			# x = X[idx]
			# Ex = np.sum(x)
			# xx = x*x
			# Ex2 = np.sum(xx)
			# Eu = np.sum(u)
			# uu = u*u
			# Eu2 = np.sum(uu)
			# xu = x*u
			# Exu = np.sum(xu)

			# Sxx = Ex2 - Ex*Ex / n
			# Suu = Eu2 - Eu*Eu / n
			# Sxu = Exu - Ex*Eu / n

			# # L2^2 distance between u and x, not normed by length
			# d = Suu - Sxu*Sxu / Sxx

			# # stats with each element missing; eg vEx = vector of what sum of
			# # x vals would be with each x val missing; ie, vEx[i] = Ex - x[i]
			# vEx = Ex - x
			# vEx2 = Ex2 - xx
			# vEu = Eu - u
			# vEu2 = Eu2 - uu
			# vExu = Exu - xu

			# vSxx = vEx2 - vEx*vEx / n
			# vSuu = vEu2 - vEu*vEu / n
			# vSxu = vExu - vEx*vEu / n

			# ds = vSuu - vSxu*vSxu / vSxx
			# # ds = 1 - ds / n
			# # ds = w * ds
			# # ax1.plot(ds)
			# ax1.plot(d - ds)

			# TODO
			# -try it using similarity that depends on change in dist with
			# each point removed
			# -try it with prefixes and suffixes removed instead of individual points

			# cum = np.cumsum

			# x = X[idx]
			# xx = x*x
			# uu = u*u
			# xu = x*u
			# # prefix ("forward") stats
			# fEx = cum(x)
			# fEx2 = cum(xx)
			# fEu = cum(u)
			# fEu2 = cum(uu)
			# fExu = cum(xu)
			# # suffix ("backward") stats
			# bEx = cum(x[::-1])
			# bEx2 = cum(xx[::-1])
			# bEu = cum(u[::-1])
			# bEu2 = cum(uu[::-1])
			# bExu = cum(xu[::-1])

			# idxs = np.arange(n) + 1

			# fSxx = fEx2 - fEx*fEx / idxs
			# fSuu = fEu2 - fEu*fEu / idxs
			# fSxu = fExu - fEx*fEu / idxs

			# bSxx = bEx2 - bEx*bEx / idxs
			# bSuu = bEu2 - bEu*bEu / idxs
			# bSxu = bExu - bEx*bEu / idxs

			# fD = fSuu - fSxu*fSxu / fSxx
			# bD = bSuu - bSxu*bSxu / bSxx
			# bD = bD[::-1]

			# fDeltaD = np.r_[0, np.diff(fD)]
			# bDeltaD = np.r_[-np.diff(bD), 0]

			# fSims = np.maximum(0, 1 - fDeltaD)
			# bSims = np.maximum(0, 1 - bDeltaD)

			# # this has a nan in the 2nd to last element, so it and uSim get wrecked
			# sims = (fSims + bSims) / 2

			# uSim = np.mean(sims)
			# print sims
			# print uSim

			# ax1.plot(fD)
			# ax1.plot(bD)
			# ax1.plot(np.abs(np.diff(fD)))
			# ax1.plot(np.abs(np.diff(bD)))
			# ax1.plot(fDeltaD)
			# ax1.plot(bDeltaD)
			# ax1.plot(fSims)
			# ax1.plot(bSims)
			# ax1.plot(sims)
			# ax1.plot(sims - uSim)
			# ax1.plot(fD + bD)

			# alternate strategy: find dist to mean of each point

			# Levy's rule -- works well with raw dists when mean not off too much
			x = X[idx].copy()
			x += np.random.randn(n) * .01
			# includeIdxs = np.where(w > 0)
			# # diffs = x[includeIdxs] - u[includeIdxs]
			# xInclude = x[includeIdxs]
			# x[includeIdxs] = (xInclude - np.mean(xInclude)) / np.std(xInclude)
			# uInclude = u[includeIdxs]
			# u[includeIdxs] = (uInclude - np.mean(uInclude)) / np.std(uInclude)
			diffs = x - u
			sims = 1 - diffs*diffs
			sims = np.maximum(0, sims)

			# print sims - allSims[idx]
			# sims[w <= 0] = 0
			# assert(sims[w <= 0] == 0)

			# diffs = x - u
			# alpha = .1
			# uSim = (1.-alpha) * uSim + alpha * np.mean(sims)
			# print uSim
			# sims = allSims[idx]
			y = np.dot(w, sims)
			# lurn = .2 / (it + 1) # oscillates and dies occasionally

			# TODO learning rates should be a function of y and/or z

			lurn = .1 / (it + 1)
			w += lurn * (sims - uSim - w)*y

			uLurn = 1. / (it + 1)
			u = (1-uLurn) * u + uLurn * X[idx]

			simLurn = uLurn / 10
			uSim = (1-simLurn) * uSim + simLurn * np.mean(sims)

			# LPF
			# w = np.convolve(w, filt, mode='same')

			# L1 regularize # TODO pgm inference subj to adjacency constraints
			# w -= .0001 * np.sum(np.abs(w))

			# drop low weights
			THRESH = .001
			w[w < THRESH] = 0

			if np.sum(w) < .01:
				print "W died!"
				print "Y:", y
				print "Learning rate:", lurn
				print "uSim:", uSim
				return

			# w = w / np.linalg.norm(w)
			# print np.linalg.norm(w) # always like .9 without norming

			# ax1.plot(w)

		ax1.plot(w, linewidth=(it/5+1), label="%d" % it)

	motif = u
	motif[w <= .01] = 0
	ax3.plot(motif)

	# handles, labels = ax1.get_legend_handles_labels()
	# ax2.legend(handles, labels, loc='lower center', ncol=len(labels)/2)
	# plt.legend(loc='upper left', shadow=False, fontsize='x-large')
	# plt.legend(handles, labels, loc='bottom left', bbox_to_anchor=(1, .5), ncol=2)
	plt.tight_layout()
	plt.show()

def learnWeights(X, w=None):
	nIters = 30
	for it in range(nIters):


if __name__ == '__main__':
	tryTriangle()
