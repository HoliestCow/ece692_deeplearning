
import numpy as np
# import matplotlib.pyplot as plt
import os.path
import glob

from joblib import Parallel, delayed
import h5py
from itertools import islice
import time


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def make_sequence(x, y, sequence_length, leftandright, filehandle, name):
    tostore_spectra = np.zeros((0, sequence_length, 1024))
    tostore_labels = []
    # random_file = trainlist[i]
    # # if i % 100 == 0:
    # #     print(i)
    # x = np.array(train[random_file]['features'])
    # y = np.array(train[random_file]['label'])

    hit_index = np.where(y > 0.5)[0]
    if len(hit_index) == 0:
        return
    start_index = hit_index[0]
    end_index = hit_index[-1]
    x = x[start_index - leftandright:end_index + leftandright, :]
    y = y[start_index - leftandright:end_index + leftandright]

    # I now I have the spectra I want to use. Now I have to batch it out into fixed segments.
    index = np.arange(x.shape[0])
    index_generator = window(index, n=sequence_length)
    # tostore_spectra = np.zeros((0, sequence_length, 1024))
    # temp_tostore_labels = []

    for index_list in index_generator:
        tostore_spectra = np.concatenate((tostore_spectra, x[index_list, :].reshape((1, sequence_length, 1024))))
        # tostore_bgspectra = np.concatenate((tostore_bgspectra, background_spectra[index_list[-1], :].reshape(1, 1024)))
        tostore_labels += [y[list(index_list)[-1]]]
    tostore_labels = np.array(tostore_labels)
    # print(tostore_spectra.shape, tostore_labels.shape)
    # print(item)
    print(name)
    grp = filehandle.create_group('sample_{}'.format(name))
    grp.create_dataset('features', data=tostore_spectra, compression='gzip')
    # grp.create_dataset('target_spectra', data=tostore_bgspectra, compression='gzip')
    grp.create_dataset('labels', data=tostore_labels, compression='gzip')
    return


def main():
    # only need to do this once.
    ncores = 4
    nsamples = 10000
    sequence_length = 15
    leftandright = 20

    # Just make sequences for background files.
    f = h5py.File('cnnfeatures_dataset.h5', 'r')
    train = f['training']
    trainlist = list(train.keys())
    test = f['testing']
    testlist = list(test.keys())
    validate = f['validation']
    validatelist = list(validate.keys())

    g = h5py.File('cnnfeatures_sequential_dataset.h5', 'w')
    new_train = g.create_group('train')
    new_test = g.create_group('test')
    new_validate = g.create_group('validate')

    for item in trainlist:
        make_sequence(np.array(train[item]['features']),
                      np.array(train[item]['label']),
                      sequence_length,
                      leftandright,
                      new_train,
                      item)
    for item in testlist:
        make_sequence(np.array(train[item]['features']),
                      np.array(train[item]['label']),
                      sequence_length,
                      leftandright,
                      new_test,
                      item)

    for i in range(len(validatelist)):
        tostore_spectra = np.zeros((0, sequence_length, 1024))
        random_file = validatelist[i]
        if i % 100 == 0:
            print(i)
        x = np.array(validate[random_file]['spectra'])
        # y = np.array(validate[random_file]['label'])
        # I now I have the spectra I want to use. Now I have to batch it out into fixed segments.
        index = np.arange(x.shape[0])
        index_generator = window(index, n=sequence_length)
        # tostore_spectra = np.zeros((0, sequence_length, 1024))
        # temp_tostore_labels = []

        for index_list in index_generator:
            tostore_spectra = np.concatenate((tostore_spectra, x[index_list, :].reshape((1, sequence_length, 1024))))
            # tostore_bgspectra = np.concatenate((tostore_bgspectra, background_spectra[index_list[-1], :].reshape(1, 1024)))
            # tostore_labels += [y[list(index_list)[-1]]]
        # grp = new_validate.create_group('sample_{}'.format(i))
        grp = new_validate.create_group(validatelist[i])
        print(validatelist[i])
        grp.create_dataset('features', data=tostore_spectra, compression='gzip')
        # grp.create_dataset('target_spectra', data=tostore_bgspectra, compression='gzip')
        # grp.create_dataset('labels', data=tostore_labels, compression='gzip')

    f.close()

    return

main()
