import pickle
import os
from sklearn.datasets import load_svmlight_file
import time
import scipy.sparse as sps
from sklearn.preprocessing import normalize

print("epsilon")
dataset_path = os.path.expanduser('../data/epsilon_normalized.bz2')

A, y = load_svmlight_file(dataset_path)
A = A.toarray()
with open('../data/epsilon.pickle', 'wb') as pickle_file:
    pickle.dump((A, y), pickle_file, protocol=4)



print("RCV1 test")
dataset_path = os.path.expanduser('../data/rcv1_test.binary.bz2')

A, y = load_svmlight_file(dataset_path)
A = A.toarray()
with open('../data/rcv1_test.pickle', 'wb') as f:
    pickle.dump((A, y), f, protocol=4)
