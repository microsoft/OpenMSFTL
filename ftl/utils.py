import os
import pickle
import glob


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
