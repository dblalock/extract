#!/bin/env/python

import itertools
import copy
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.cross_validation import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin, clone

from pandas import DataFrame

from joblib import Memory
memory = Memory('./output', verbose=0)


# ================================================================
# Constants
# ================================================================

SCORE_NAME = ' Score'

# ================================================================
# Funcs
# ================================================================

def isScalar(x):
	return not hasattr(x, "__len__")


def cleanKeys(pipeParamsDict):
	"""go from, e.g., 'classify__C' to 'C'"""
	dictionary = {}
	for key, val in pipeParamsDict.iteritems():
		newKey = key.split('__', 1)[1]
		dictionary[newKey] = val
	return dictionary


def name_estimators(estimators):
	"""Generate names for estimators. (Stolen from sklearn pipeline.py)"""

	if isScalar(estimators):
		return type(estimators).__name__

	names = [type(estimator).__name__ for estimator in estimators]
	namecount = defaultdict(int)
	for est, name in zip(estimators, names):
		namecount[name] += 1

	for k, v in list(namecount.iteritems()):
		if v == 1:
			del namecount[k]

	for i in reversed(range(len(estimators))):
		name = names[i]
		if name in namecount:
			names[i] += "-%d" % namecount[name]
			namecount[name] -= 1

	return list(zip(names, estimators))


def getEstimatorNames(pipeline):
	# names, estimators = zip(*pipeline.steps)
	# estimatorClassNames = _name_estimators(estimators)
	# return zip(names, estimatorClassNames)
	_, estimators = zip(*pipeline.steps)
	namesAndDumps = name_estimators(estimators)
	names, _ = zip(*namesAndDumps)
	return names


def stageNames2EstimatorNames(pipeline):
	estNames = getEstimatorNames(pipeline)
	stageNames, _ = zip(*pipeline.steps)
	return dict(zip(stageNames, estNames))


def add_caching_to_funcs(obj, funcNames):
	mem = Memory('../.add_caching_to_funcs', verbose=11)
	if obj is None or funcNames is None:
		return
	if isScalar(funcNames):
		funcNames = [funcNames]
	for name in funcNames:
		func = getattr(obj, name, None)
		if func is not None:
			setattr(obj, name, mem.cache(func))


def add_caching(estimator):
	return add_caching_to_funcs(estimator,
		['transform', 'fit', 'fit_transform', 'predict'])


cached_apply_mem = Memory('../.cached_apply', verbose=0)
@cached_apply_mem.cache
def cached_apply(func, X):
	return func(X)


class FuncWrapper(BaseEstimator, TransformerMixin):

	@staticmethod
	def wrap(func):
		return FuncWrapper(func)

	# BaseEstimator figures out what our params are based on the signature
	# of init, so we have to list them all here (though in this case it's
	# just the funciton we're wrapping)
	def __init__(self, func, cache=True):
		self.func = func
		self.cache = cache

	def __call__(self):	# so this works as a func that returns a transform
		return self

	def fit(self, X, y=None, **params):
		return self

	def transform(self, X):
		if self.cache:
			return cached_apply(self.func, X)
		return self.func(X)


def _prependStringToKeys(string, dictionary):
	for key in dictionary.keys():
		val = dictionary.pop(key)
		newKey = ''.join((string, key))
		dictionary[newKey] = val


def makePipelines(stages, cacheBlocks=False):
	# should make this really good and then pull request to sklearn

	stages = copy.deepcopy(stages) # needed to avoid rare err when called twice
	stageNames, _ = zip(*stages) # first element of each pair

	funcsForStages = []
	for i in range(len(stages)):
		funcsForStages.append([])

	# print "makePipelines(): stages = ", stages

	for i, (stageName, funcsWithParams) in enumerate(stages):
		# print "makePipelines(): i, stageName = ", i, stageName
		# print "makePipelines(): funcsWithParams = ", funcsWithParams

		# if we got, eg, ("classify", SVC), make it ("classify", [SVC])
		if (isScalar(funcsWithParams)):
			# print "makePipelines(): funcsWithParams was a scalar"
			funcsWithParams = [funcsWithParams]

		# print "makePipelines(): funcsWithParams = ", funcsWithParams

		for funcAndParams in funcsWithParams:	# eg (SVC, ['C':[1, 5]])

			# print "makePipelines(): funcAndParams = ", funcAndParams

			# if it's just, say SVC, or None, then use default parameters
			if isScalar(funcAndParams):
				func = funcAndParams
				# if func is not None:
				funcsForStages[i].append((func, {}))

			# otherwise, it's a function with something we assume can be made
			# into a ParamsGrid (and it's the user's fault if it can't).
			# To leverage sklearn Pipeline's ability to work with GridSearchCV,
			# we need to have all the keys in the parameters be of the form
			# stageName__paramName; e.g., for stage name "dimReduction" and
			# parameter name "n_dimensions", we want "dimReduction__n_dimensions"
			else:
				assert len(funcAndParams) == 2, \
					"Bad (function, params) given; neither scalar nor length 2: %s" \
					% (str(funcAndParams),)

				func = funcAndParams[0]
				params = funcAndParams[1]
				prefix = ''.join((stageName, '__'))

				# if it's just one level of dict, change keys directly
				if isinstance(params, dict):
					_prependStringToKeys(prefix, params)
					funcsForStages[i].append((func, params))

				# otherwise, it must be a collection of dicts
				else:
					for d in params:
						assert isinstance(d, dict), \
							"Parameter element %s not a dict!" % str(d)
						_prependStringToKeys(prefix, d)
						funcsForStages[i].append((func, d))

	# now we take the cartesian product of all funcs at all stages
	# and store the result from each, along with the parameters;
	allPipelines = []
	allParams = []
	allCombos = itertools.product(*funcsForStages)
	for combo in allCombos:
		objs = []
		comboParams = {}
		for c in combo:
			func = c[0]
			params = c[1]
			if func is not None:
				comboParams.update(params)
			objs.append(func)

		stagesNotNone = [(stageName, obj) for stageName, obj in zip(stageNames, objs)
			if obj is not None]
		try:
			# this will only work with my modified Pipeline class, which hasn't been
			# pulled into sklearn yet
			allPipelines.append(Pipeline(stagesNotNone, cache=cacheBlocks))
		except TypeError:
			allPipelines.append(Pipeline(stagesNotNone))
		allParams.append(ParameterGrid(comboParams))

	# TODO option to return unique (pipe, params) for each possible params
	# configuration by iterating through each paramsGrid and adding a pair
	return allPipelines, allParams

@memory.cache
def gridSearchPipeline(pipeline, paramsGrid, Xtrain, Ytrain, **cvParams):
	print("Grid Searching pipeline:")
	print(pipeline)

	# use 5-fold stratified cross-validation by default to maintain
	# consistent class balance across training and testing
	if 'cv' not in cvParams:
		# print "Ytrain: ", Ytrain
		# numClasses = len(np.unique(Ytrain))
		# examplesPerClass = len(Ytrain) / numClasses
		# nFolds = max(5, examplesPerClass / 5)
		# if nFolds < 5:
		# if True:
			# r, c = Ytrain.shape
			# print "tiny Ytrain size: (%d, %d)" % Ytrain.shape # (r, c)
			# for row in Ytrain: print row
		# cvParams['cv'] = StratifiedKFold(Ytrain, n_folds=nFolds)
		cvParams['cv'] = StratifiedKFold(Ytrain, n_folds=5)

	cv = GridSearchCV(pipeline, paramsGrid, **cvParams)
	cv.fit(Xtrain, Ytrain)
	return cv

@memory.cache
def stratifiedSplitTrainTest(X, Y, n_folds=4):
	split = StratifiedKFold(Y, n_folds=n_folds, random_state=12345)
	train_index, test_index = next(iter(split))
	X, Xtest = X[train_index], X[test_index]
	Y, Ytest = Y[train_index], Y[test_index]
	return X, Xtest, Y, Ytest

def scoresForParams(d, X, Y, Xtest=None, Ytest=None, crossValidate=True,
	returnValidationScores=False, cacheBlocks=False, needsTestSet=True,
	**cvParams):

	# printVar("Xtrain", X.shape)
	# printVar("Ytrain", Y.shape)
	# printVar("Xtest", Xtest.shape)
	# printVar("Ytest", Ytest.shape)
	# return

	pipelines, pipeParams = makePipelines(d, cacheBlocks=cacheBlocks)

	# ------------------------ No cross-validation; just map over param values
	if not crossValidate: # eg, for unsupervised pattern discovery
		allScores = []
		allParams = []
		for pipe, params in zip(pipelines, pipeParams):
			# print "pipe, params = ", pipe, params
			# params = params if len(params) else [{}]
			# print "scoresForParams() paramsGrid", params
			for parameters in params: # params is a ParamsGrid
				p = clone(pipe)
				# print "scoresForParams() params", parameters
				p.set_params(**parameters)
				p.fit(X, Y)
				allScores.append(p.score(X, Y))
				allParams.append(parameters)
		return allScores, allParams

	# ------------------------ Cross-validation

	# split into training and test sets; we use a stratified split
	# to maintain class balance; more folds -> more training;
	# we may not need a test set if we're using
	if needsTestSet and ((Xtest is None) or (Ytest is None)):
		X, Xtest, Y, Ytest = stratifiedSplitTrainTest(X, Y)

	if returnValidationScores:
		allScores = []
		allParams = []
	else:
		allBestScores = []
		allBestParams = []

	for pipe, params in zip(pipelines, pipeParams):

		# TODO option to just try each combination of params on
		# the test data and return everything--often want to know
		# not just what was best, but what the distro of performance
		# as a function of parameter values is

		if 'scoring' not in cvParams: # only overwrite if not passed; preserve None
			cvParams['scoring'] = 'accuracy'
		cvParams['n_jobs'] = cvParams.get('n_jobs') or 3
		cvParams['verbose'] = cvParams.get('verbose') or 1

		cv = gridSearchPipeline(pipe, params, X, Y, **cvParams)

		whichStageParams = stageNames2EstimatorNames(pipe)

		if returnValidationScores:
			allScoresTuples = cv.grid_scores_ # each tup is params, score, fold-scores
			for tup in allScoresTuples:
				allScores.append(tup.mean_validation_score)
				params = copy.deepcopy(tup.parameters)
				params.update(whichStageParams)
				allParams.append(params)
		else:
			bestParams = cleanKeys(cv.best_params_)
			bestParams.update(whichStageParams)

			score = cv.score(Xtest, Ytest)
			allBestParams.append(bestParams)
			allBestScores.append(score)

	if returnValidationScores:
		return allScores, allParams
	return allBestScores, allBestParams

def createScoreAndParamsTable(scores, params, scoreName=' Score', **sink):
	# add a leading space to ensure score is first alphabetically
	# scoreName = ' ' + scoreName

	# print "createScoreAndParamsTable()", scores, params

	# add score to the data that will constitute the table; we first
	# make a copy of the parameters though so as not to modify the
	# collection passed in; since this is a collection of dicts that
	# we don't want to alter, we need a deep copy
	params = copy.deepcopy(params)

	for score, p in zip(scores, params):
		if isinstance(score, dict):
			for k, v in score.iteritems():
				p[k] = v
		else:
			p[scoreName] = score

	outcomes = DataFrame.from_records(params) # pandas magically eats dictionaries

	# remove stage name from parameter name
	betterNames = map(lambda name: name.split('__')[-1], outcomes.columns.values)
	outcomes.columns = betterNames

	outcomes.sort(inplace=True)								# sort by col name
	if scoreName in betterNames:
		outcomes.sort(scoreName, ascending=False, inplace=True)	# sort by score
	return outcomes


# def tryParams(d, X, Y, Xtest=None, Ytest=None, n_folds=4, scoring='accuracy', cacheBlocks=FalseXtest=None, Ytest=None, n_folds=4, scoring='accuracy', cacheBlocks=FalseXtest=None, Ytest=None, n_folds=4, scoring='accuracy', cacheBlocks=False):
def tryParams(d, X, Y, **kwargs):
	scores, params = scoresForParams(d, X, Y, **kwargs)
	return createScoreAndParamsTable(scores, params, **kwargs)


if __name__ == '__main__':
	pass
