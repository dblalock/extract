#!/usr/env/python

import numpy as np
import pandas as pd

def main():
	df = pd.read_csv('dishwasher-labels.txt', delim_whitespace=True, header=None)
	df.columns = ['start', 'end', 'lbl']

	durations = df['end'] - df['start']

	allDurationsPositive = np.all(durations > 0)
	if not allDurationsPositive:
		print("Oh no! Durations negative in lines: ")
		print(np.where(durations <= 0)[0] + 1)
		assert(allDurationsPositive)

	allDurationsReasonable = np.all(durations < 250)
	if not allDurationsReasonable:
		print("Oh no! Durations are huge in lines: ")
		print(np.where(durations >= 250)[0] + 1)
		assert(allDurationsReasonable)

	endTimes = np.array(df['end'], dtype=np.int)
	increases = endTimes[1:] - endTimes[:-1]
	allIncreasing = np.all(increases >= 0)
	if not allIncreasing:
		print("Oh no! End indices decreased in lines: ")
		print(np.where(increases < 0)[0] + 2)
		assert(allIncreasing)

	print("unique labels:")
	print(np.unique(df['lbl']))

if __name__ == '__main__':
	main()
