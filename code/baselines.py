import argparse
import os
import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file

from utils import pickle_it

def loss(clf, X, y, reg):
    baseline_loss = np.sum(np.logaddexp(0, -y * (X @ clf.coef_.transpose()).squeeze())) / X.shape[0]
    baseline_loss += reg / 2 * np.sum(np.square(clf.coef_))
    return baseline_loss


# """ EPSILON """

print('epsilon')
dataset_path = os.path.expanduser('../data/binary/epsilon_normalized.bz2')
A, y = load_svmlight_file(dataset_path)

reg = 1 / A.shape[0]
clf = SGDClassifier(tol=1e-5, loss='log', penalty='l2', alpha=reg)
clf.fit(A, y)
l = loss(clf, A, y, reg)
print("loss: {}".format(l))
print("train accuracy: {}".format(clf.score(A, y)))
optimums = {}
optimums['epsilon'] = l

directory = 'dump/optimum-epsilon/'
if not os.path.exists(directory):
        os.makedirs(directory)
pickle_it(optimums, 'baselines', directory)
