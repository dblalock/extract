#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ..utils.files import ensureDirExists

SAVE_DIR = "~/Desktop/ts/figs"


def colorForLabel(lbl):
	COLORS = ['b','g','r','c','m','y','k']
	if hasattr(lbl, '__len__'):
		lbl = lbl[0]
	try:
		idx = int(lbl) % len(COLORS) # number
	except:
		idx = ord(lbl) % len(COLORS) # char
	return COLORS[idx]


def nameFromDir(datasetDir):
	return os.path.basename(datasetDir)


def saveCurrentPlot(plotName, suffix='',subdir=''):
	saveDir = os.path.join(os.path.expanduser(SAVE_DIR),subdir)
	ensureDirExists(saveDir)
	fileName = os.path.join(saveDir,plotName) + suffix
	print('saving plot as file %s' % fileName)
	plt.savefig(fileName)


# TODO remove dup code
def plotVertLine(x, ymin=None, ymax=None, ax=None, **kwargs):
	if ax and (not ymin or not ymax):
		ymin, ymax = ax.get_ylim()
	if not ax:
		ax = plt

	kwargs.setdefault('color', 'k')
	kwargs.setdefault('linestyle', '--')
	if 'linewidth' not in kwargs:
		kwargs.setdefault('lw', 2)

	ax.plot([x, x], [ymin, ymax], **kwargs)


def plotRect(ax, xmin, xmax, ymin=None, ymax=None, alpha=.2,
	showBoundaries=True, color='grey', fill=True, hatch=None, **kwargs):
	if ax and (ymin is None or ymax is None):
		ymin, ymax = ax.get_ylim()
	if fill:
		patch = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
				facecolor=color, alpha=alpha, hatch=hatch)
		ax.add_patch(patch)
	if showBoundaries:
		plotVertLine(xmin, ymin, ymax, ax=ax, color=color, **kwargs)
		plotVertLine(xmax, ymin, ymax, ax=ax, color=color, **kwargs)


def plotRanges(ax, startEndIdxPairs, **kwargs):
	for start, end in startEndIdxPairs:
		plotRect(ax, start, end, **kwargs)


def imshowBetter(X, ax=None):
	if not ax:
		ax = plt
	ax.imshow(X, interpolation='nearest', aspect='auto')


def setXtickPadding(ax, padding):
	for tick in ax.get_xaxis().majorTicks:
		tick.set_pad(padding)


def setYtickPadding(ax, padding):
	for tick in ax.get_yaxis().majorTicks:
		tick.set_pad(padding)

# def removeAllButFirstAndLastXTick(ax):
# 	ticks = ax.get_xticks()
# 	print "xticks", ticks
# 	# print "xticks array", np.array(ticks)
# 	# lbls = list(ax.get_xticklabels())
# 	# lbls = [item.get_text() for item in ax.get_xticklabels()]
# 	lbls = list(ticks)
# 	print "orig lbls", lbls
# 	for i in range(1, len(lbls) - 1):
# 		lbls[i] = ''
# 	ax.set_xticklabels(lbls)


def removeEveryOtherXTick(ax):
	lbls = ax.get_xticks().astype(np.int).tolist()
	for i in range(1, len(lbls), 2):
		lbls[i] = ''
	ax.set_xticklabels(lbls)


def removeEveryOtherYTick(ax):
	lbls = ax.get_yticks().astype(np.int).tolist()
	for i in range(1, len(lbls), 2):
		lbls[i] = ''
	ax.set_yticklabels(lbls)

