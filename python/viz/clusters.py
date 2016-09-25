import time

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from joblib import Memory
memory = Memory('./output', verbose=0)

from ..utils import sliding_window as window
from ..analyze.munge import groupXbyY
from ..transforms import resampleToLengthN
from ..utils.arrays import zNormalizeRows

from ..sparseFiltering import SparseClusterer

from viz_utils import saveCurrentPlot

@memory.cache
def makeSparseClusterer(X, k=-1):
    return SparseClusterer(k)

@memory.cache
def makeMeanShift(X, k=-1):
    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
    return cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

@memory.cache
def makeWard(X, k=2):
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    return cluster.AgglomerativeClustering(n_clusters=k,
                        linkage='ward', connectivity=connectivity)

@memory.cache
def makeKMeans(X=None, k=2):
    return cluster.MiniBatchKMeans(n_clusters=k)

@memory.cache
def makeSpectral(X=None, k=2):
    return cluster.SpectralClustering(n_clusters=k,
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors")

@memory.cache
def makeDBScan(X=None, k=-1):
    return cluster.DBSCAN(eps=.2)

@memory.cache
def makeAffinityProp(X=None, k=-1):
    return cluster.AffinityPropagation(damping=.9, preference=-200)

@memory.cache
def makeAvgLinkage(X=None, k=2):
    connectivity = kneighbors_graph(X, n_neighbors=10)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    return cluster.AgglomerativeClustering(linkage="average",
                                affinity="cityblock", n_clusters=k,
                                connectivity=connectivity)

@memory.cache
def makeMaxLinkage(X=None, k=2):
    connectivity = kneighbors_graph(X, n_neighbors=10)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    return cluster.AgglomerativeClustering(linkage="complete",
                                affinity="cityblock", n_clusters=k,
                                connectivity=connectivity)

def makeClusterers(X, k=2):
    return [('MiniBatchKMeans', makeKMeans(X, k)),
            ('AffinityPropagation', makeAffinityProp()),
            ('MeanShift', makeMeanShift(X)),
            ('SpectralClustering', makeSpectral(X, k)),
            ('Ward', makeWard(X, k)),
            ('AgglomerativeAvg', makeAvgLinkage(X, k)),
            ('AgglomerativeMax', makeMaxLinkage(X, k)),
            ('AgglomerativeWard', makeWardLinkage(X, k)),
            ('DBSCAN', makeDBScan())]

def makeSimpleDatasets(n_samples=1500): # from sklearn example
    np.random.seed(0)
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    return [noisy_circles, noisy_moons, blobs, no_structure]

def colorsForLabels(y):
    colors = np.array([x for x in 'bgrcmyk'])
    y = np.asanyarray(y)
    pointColorIdxs = np.mod(y, len(colors))
    return colors[pointColorIdxs].tolist()

def sklearn_example():
    plt.figure(figsize=(17, 9.5))
    plt.subplots_adjust(left=.001, right=.999, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    plotNum = 1
    for i_dataset, dataset in enumerate(makeSimpleDatasets()):
        X, y = dataset
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        for name, algorithm in makeClusterers(X):
            # predict cluster memberships
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            # plot
            plt.subplot(4, 7, plotNum)
            if i_dataset == 0:
                plt.title(name, size=18)

            plt.scatter(X[:, 0], X[:, 1], color=colorsForLabels(y_pred), s=10)

            if hasattr(algorithm, 'cluster_centers_'):
                centers = algorithm.cluster_centers_
                center_colors = colorsForLabels(range(len(centers)))
                plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plotNum += 1

    plt.show()


# @memory.cache
# def cachedFitClusterer(clusterer, X):
#     clusterer.fit(X)
#     return clusterer

def showClusters(name, X, Y, ks, lengths, clusterFactory=makeKMeans, stride=1, normSubSeqs=True):
    print("showing clusters for dataset %s" % name)
    plt.figure(figsize=(17, 9.5))
    plt.rcParams["font.size"] = 18

    plotNum = 0
    plotRows = len(ks)
    plotCols = len(lengths)

    n = max(max(lengths), 64) # if we want k > 64, ts must be at least this long
    n = min(n, X.shape[1])  # can't be longer than ts

    print("Resampling and normalizing...")
    X2 = resampleToLengthN(X, n)
    X2 = zNormalizeRows(X2)

    print("Fitting and plotting...")
    for i, k in enumerate(ks):
        for j, l in enumerate(lengths):
            plotNum += 1
            if l > n:   # window longer than whole sequence
                continue

            subseqs = window.sliding_windows_of_rows(X2, l, stride)
            if normSubSeqs:
                subseqs = zNormalizeRows(subseqs)
            clusterer = clusterFactory(subseqs, k)
            print subseqs.shape
            print subseqs
            clusterer.fit(subseqs)

            # cluster centers are the actual means
            if hasattr(clusterer, 'cluster_centers_'):
                centers = clusterer.cluster_centers_
            else:
                lbls = clusterer.labels_
                # grouped = munge.groupXbyY(subseqs, lbls)
                grouped = groupXbyY(subseqs, lbls)
                centers = map(lambda rows: rows.mean(axis=0), grouped)

            plt.subplot(plotRows, plotCols, plotNum)
            for center in centers:
                plt.plot(np.arange(len(center)), center)

            plt.xticks(())
            plt.yticks(())
            plt.tight_layout()

            if i == 0:
                plt.title("Window length = %d" % l)
            if j == 0:
                plt.ylabel("K = %d" % k)

    algoName = clusterer.__class__.__name__
    if normSubSeqs:
        algoName += '-Normalized'
    plt.suptitle("Clusters in %s Dataset using %s" % (name, algoName))
    plt.tight_layout()
    plt.subplots_adjust(left=.03, right=.97, bottom=.01, top=.9, wspace=.02,
                        hspace=.01)

    # all algos for each dataset in one dir, as well as subdir for each algo
    saveCurrentPlot("%s_%s.png" % (name, algoName), subdir='cluster')
    saveCurrentPlot("%s_%s.png" % (name, algoName),
                    subdir=os.path.join('cluster', algoName))
    plt.close()


def showDatasetClusters(name, Xtrain, Ytrain, Xtest, Ytest, ks, lengths,
                        clusterFactory=makeKMeans, stride=1, normSubSeqs=True):
    X = np.vstack((Xtrain, Xtest))
    Y = np.hstack((Ytrain, Ytest))
    showClusters(name, X, Y, ks, lengths, clusterFactory, stride)


# def showAgglomerativeClusters(name, Xtrain, Ytrain, Xtest, Ytest, lengths,
#                               stride=1, normSubSeqs=True):
#     X = np.vstack((Xtrain, Xtest))
#     Y = np.hstack((Ytrain, Ytest))

#     print("showing clusters for dataset %s" % name)
#     plt.figure(figsize=(17, 9.5))
#     plt.rcParams["font.size"] = 18

#     print("Resampling and normalizing...")
#     X2 = resampleToLengthN(X, n)
#     X2 = zNormalizeRows(X2)

#     n = max(max(lengths), 64) # if we want k > 64, ts must be at least this long
#     n = min(n, X.shape[1])  # can't be longer than ts

#     for j, l in enumerate(lengths):
#         if l > n:   # window longer than whole sequence
#                 continue


if __name__ == '__main__':
    sklearn_example()
