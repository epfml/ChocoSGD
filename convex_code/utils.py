import glob
import os
import pickle


def to_percent(x):
    return round(100 * x, 2)


def pickle_it(var, name, directory):
    with open(os.path.join(directory, "{}.pickle".format(name)), 'wb') as f:
        pickle.dump(var, f)


def unpickle_dir(d):
    data = {}
    assert os.path.exists(d), "{} does not exists".format(d)
    for file in glob.glob(os.path.join(d, '*.pickle')):
        name = os.path.basename(file)[:-len('.pickle')]
        with open(file, 'rb') as f:
            var = pickle.load(f)
        data[name] = var
    return data
