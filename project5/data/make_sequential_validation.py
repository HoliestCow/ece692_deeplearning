
import numpy as np
import h5py
import os.path
import glob
from itertools import islice


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


def main():
    # this is for the validation set, where i have to analyze
    #     each file individual
    sequence_length = 15
    f = h5py.File('sequential_dataset_validation.h5', 'w')
    test_filelist = glob.glob('./test_integrations/2*.npy')

    for i in range(len(test_filelist)):
        if i % 10 == 0:
            print('validation sample {}'.format(i))
        filename = test_filelist[i]
        head, tail = os.path.split(filename)
        dataname = tail[:-4]
        x = np.load(os.path.join('./test_integrations', dataname + '.npy'))
        spectra = x[:, 1:]
        index = np.arange(spectra.shape[0])
        index_generator = window(index, n=sequence_length)
        tostore_spectra = np.zeros((0, sequence_length, 1024))
        for index_list in index_generator:
            tostore_spectra = np.concatenate((tostore_spectra, spectra[index_list, :].reshape((1, sequence_length, 1024))), axis=0)
        f.create_dataset(dataname, data=tostore_spectra, compression='gzip')
    f.close()
    return

main()
