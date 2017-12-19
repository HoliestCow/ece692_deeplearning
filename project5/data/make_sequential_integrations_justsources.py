
import numpy as np
import matplotlib.pyplot as plt
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


def label_datasets():

    targetfile = '/home/holiestcow/Documents/zephyr/datasets/muse/trainingData/answers.csv'
    head, tail = os.path.split(targetfile)

    # filename = []
    source_labels = {}

    id2string = {0: 'Background',
                 1: 'HEU',
                 2: 'WGPu',
                 3: 'I131',
                 4: 'Co60',
                 5: 'Tc99',
                 6: 'HEUandTc99'}


    f = open(targetfile, 'r')
    a = f.readlines()
    for i in range(len(a)):
        line = a[i].strip()
        if line[0] == 'R':
            continue
        parsed = line.split(',')
        filename = parsed[0]
        source = parsed[1]
        source_time = parsed[2]
        source_labels[filename] = {'source': id2string[int(source)],
                                   'time': float(source_time)}
    f.close()
    return source_labels

def make_spectral_plots(targetfile, outdir, labels):
    random_file = targetfile
    x = np.load('./integrations/' + random_file + '.npy')
    source_index = int(labels[random_file]['time'])
    source_type = labels[random_file]['source']

    left = source_index - 20
    right = source_index + 20

    if left < 0:
        left = 0
    if right > x.shape[0]:
        right = x.shape[0]

    for j in np.arange(left, right):
        fig = plt.figure()
        plt.plot(x[j, 1:])
        plt.title('type{}_current{}_target{}'.format(source_type, j, source_index))
        plt.axis([0, 1024, 0, 50])
        fig.savefig('./{}/{:06d}_{:04d}.png'.format(outdir, int(targetfile), j))
        plt.close()
    return



def main():
    # only need to do this once.
    ncores = 4

    # test_filelist = glob.glob('/home/holiestcow/Documents/zephyr/datasets/muse/testData/2*.csv')

    id2string = {0: 'Background',
                 1: 'HEU',
                 2: 'WGPu',
                 3: 'I131',
                 4: 'Co60',
                 5: 'Tc99',
                 6: 'HEUandTc99'}

    string2id = {'Background': 0,
                 'HEU': 1,
                 'WGPu': 2,
                 'I131': 3,
                 'Co60': 4,
                 'Tc99': 5,
                 'HEUandTc99': 6}

    # filelist = glob.glob('/home/holiestcow/Documents/zephyr/datasets/muse/trainingData/1*.csv')
    # Parallel(n_jobs=ncores)(delayed(parse_datafiles)(item, binnumber) for item in filelist)
    labels = label_datasets()

    allfilelist = []
    backgroundfilelist = []
    sourcefilelist = []
    for anyfile in labels.keys():
        if labels[anyfile]['source'] == 'Background':
            backgroundfilelist += [anyfile]
        else:
            sourcefilelist += [anyfile]
        allfilelist += [anyfile]
    sourcefilelist.sort()
    backgroundfilelist.sort()

    # bg_training_threshold = int(len(backgroundfilelist) / 2)
    s_training_threshold = int(len(sourcefilelist) / 2)
    # backgroundfilelist_train = backgroundfilelist[:bg_training_threshold]
    # backgroundfilelist_test = backgroundfilelist[bg_training_threshold:]
    sourcefilelist_train = sourcefilelist[:s_training_threshold]
    sourcefilelist_test = sourcefilelist[s_training_threshold:]

    validatelist = glob.glob('./test_integrations/2*.npy')
    validatelist.sort()

    # print(labels)

    # Just make sequences for background files.
    # source_spectra_train = np.zeros((len(sourcefilelist_train), 5, 1024))
    # make_spectral_plots(sourcefilelist_train[0], 'train_plots', labels)
    Parallel(n_jobs=ncores)(delayed(make_spectral_plots)(item, 'train_plots', labels) for item in sourcefilelist_train)
    Parallel(n_jobs=ncores)(delayed(make_spectral_plots)(item, 'test_plots', labels) for item in sourcefilelist_test)

    # for i in range(len(validatelist)):
    #     tostore_spectra = np.zeros((0, sequence_length, 1024))
    #     random_file = validatelist[i]
    #     if i % 100 == 0:
    #         print('{} validation samples done in {} s '.format(i, time.time() - a))
    #     x = np.array(np.load(random_file))
    #     x = x[:, 1:]
    #     head, tail = os.path.split(random_file)
    #     runname = tail[:-4]
    #     tostore_spectra = x
    #     validate.create_dataset(runname, data=tostore_spectra, compression='gzip')
    return

main()
