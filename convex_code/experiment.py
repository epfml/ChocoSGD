import argparse
import multiprocessing as mp
import os
import pickle
from sklearn.datasets import load_svmlight_file

import numpy as np

from logistic import LogisticDecentralizedSGD
from parameters import Parameters
from utils import pickle_it

A, y = None, None


def run_logistic(param):
    m = LogisticDecentralizedSGD(param)
    res = m.fit(A, y)
    print('{} - score: {}'.format(param, m.score(A, y)))
    return res


def run_experiment(directory, dataset_path, params, nproc=None):
    global A, y
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle_it(params, 'params', directory)

    print('load dataset')
    with open(dataset_path, 'rb') as f:
      A, y = pickle.load(f)
      print("A.shape= ", A.shape)

    print('start experiment')
    with mp.Pool(nproc) as pool:
        results = pool.map(run_logistic, params)
    print(results)

    pickle_it(results, 'results', directory)
    print('results saved in "{}"'.format(directory))
