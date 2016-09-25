This document describes the datasets used, as well as our preprocessing, in more detail than what is given in the paper.

# TIDIGITS

The [TIDIGITS dataset](https://catalog.ldc.upenn.edu/LDC93S10) is a large collection of human utterances of decimal digits. We use a subset of the data consisting of all recordings containing only one type of digit (e.g., only ``9''s). We randomly concatenated sets of these recordings to form 1604 time series in which multiple speakers utter the same word a total of 5-8 times. As is standard practice, we represented the resulting time series using Mel-Frequency Cepstral Coefficients (MFCCs), rather than as the raw speech signal. We use the first 13 MFCCs (as is common) extracted via a sliding window of length 10ms and a stride of 5ms. We then downsampled this densely-sampled data by a factor of 2.

Because each original recording brackets the word utterance with silence, we used as instance boundaries the first and last time steps in which the signal power exceeded 10% of its maximum value. This is imperfect, but tends to be within 10-20% of the correct boundaries of each word.

# Dishwasher

The [Dishwasher dataset](http://ampds.org) consists of energy consumption and related electricity metrics at a power meter connected to a residential dishwasher. It contains twelve variables and two years worth of data sampled once per minute, for a total of 12.6 million data points.

We manually labeled pattern instances across all 1 million+ samples. See the dishwasher README (also in this directory) for details of how this was done and what the annotations mean.

Because this data is 100x longer than what the comparison algorithms can process in a day, we followed much the same procedure as for the TIDIGITS dataset. Namely, we extracted sets of five to eight pattern instances and concatenated them to form shorter time series. We also extracted a random amount of data around each instance, with length in [25, 200] on either side (the patterns are of length ~150).

# MSRC-12

The [MSRC-12 dataset](http://research.microsoft.com/en-us/um/cambridge/projects/msrc12/) consists of (x,y,z) human joint positions captured by a Microsoft Kinect while subjects repeatedly performed particular motions. Each of the 594 time series in the dataset is 80 dimensional and contains 8-12 pattern instances. To help existing algorithms (which are quadratic) run in a reasonable time frame, we downsampled each time series by a factor of 2.

Each instance is labeled with a single marked time step, rather than with its boundaries. Since the boundaries were not recorded, we use the number of marks in each time series as ground truth. That is, if there are k marks, we treat the first k regions returned as correct. This is a less stringent criterion than on other datasets, but favors existing algorithms insofar as they often fail to identify exact region boundaries. We attempted to infer ground truth boundaries based on the positions of the marks, but found that this was not possible to do accurately.

# UCR

Following [Yingchareonthawornchai et al.](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6729632), we constructed synthetic datasets by planting examples from the [UCR Time Series Archive](http://www.cs.ucr.edu/~eamonn/time_series_data/) in random walks. We took examples from the 20 smallest datasets (before the 2015 update), as measured by the length of their objects. For each dataset, we created 50 time series, each containing five objects of one class. This yields 1000 time series and 5000 instances. The random walk segments between the starts and ends of instances are 1.25x the lengths of the instances.

