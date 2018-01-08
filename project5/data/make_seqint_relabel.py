
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

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def findMiddle(input_list):
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
        # return (input_list[int(middle)], input_list[int(middle-1)])
        return input_list[int(middle)]

def store_sequence(targetfile, filehandle, labels):


    string2id = {'Background': 0,
                 'HEU': 1,
                 'WGPu': 2,
                 'I131': 3,
                 'Co60': 4,
                 'Tc99': 5,
                 'HEUandTc99': 6}

    random_file = targetfile
    x = np.load('./train_integrations/' + random_file + '.npy')
    x = x[:, 1:]
    source_index = int(labels[random_file]['time'])
    source_type = labels[random_file]['source']

    if source_index <= 3:  # This means this is a background file
        source_index = np.random.randint(x.shape[0])
        source_type = 'Background'

    # left = source_index - 29
    # right = source_index + 30
    # left = source_index - 119
    # right = source_index + 120
    left = 0
    right = x.shape[0]
    if left < 0:
        left = 0
    if right >= x.shape[0]:
        right = x.shape[0]

    current_slice = x
    current_slice_counts = np.sum(current_slice, axis=1)
    # modified_slice_counts = reject_outliers(current_slice_counts, m=5)
    # current_slice_count_mean = np.mean(modified_slice_counts)
    current_slice_count_median = np.median(current_slice_counts)
    # current_slice_count_min = np.min(current_slice_counts)
    # current_slice_count_std = np.std(modified_slice_counts)
    hits = np.zeros((current_slice.shape[0], ), dtype=bool)

    # threshold = current_slice_count_min + (0.25 * current_slice_count_min)
    # threshold  = current_slice_count_mean + (0.5 * current_slice_count_std)
    threshold = current_slice_count_median
    # print(threshold)
    # threshold = current_slice_count_mean + (1 * current_slice_count_std)
    hits[current_slice_counts > threshold] = True

    machine = np.argwhere(hits == True)
    hits = np.zeros((current_slice.shape[0], ), dtype=bool)
    machine = machine.reshape((machine.shape[0], ))
    grouping = group_consecutives(machine)
    for group in grouping:
        if source_index in group:
            hits[group] = True
    if np.sum(hits) == 0:
        listofmiddles = []
        # print(grouping)
        for group in grouping:
            listofmiddles += [findMiddle(group)]
        # print(listofmiddles)
        listofmiddles = np.array(listofmiddles)
        delta = np.subtract(listofmiddles, np.tile(source_index, len(listofmiddles)))
        delta_squared = np.power(delta, 2)
        desired_index = np.argmin(delta_squared)
        hits[grouping[desired_index]] = True
    if np.sum(hits) == 1 or np.sum(hits) == 0:
        return

    tostore_spectra = current_slice
    tostore_labels = np.zeros((tostore_spectra.shape[0], ), dtype=int)
    tostore_labels[hits] = int(string2id[source_type])

    tostore_spectra = tostore_spectra[left:right, :]
    tostore_labels = tostore_labels[left:right]

    grp = filehandle.create_group(targetfile)
    grp.create_dataset('measured_spectra', data=tostore_spectra, compression='gzip')
    grp.create_dataset('labels', data=tostore_labels, compression='gzip')

    # Plot the hits:
    # for i in range(len(tostore_labels)):
    #     if tostore_labels[i] != 0:
    #         fig = plt.figure()
    #         plt.plot(tostore_spectra[i, :])
    #         plt.title('source{}_time{}'.format(source_type, source_index))
    #         plt.axis([0, 1024, 0, 50])
    #         fig.savefig('./plots/{}_{}'.format(targetfile, i))
    #         plt.close()

    return


    # print(source_type)
    # print(np.sum(x[source_index-2, :]),np.sum(x[source_index-1, :]),np.sum(x[source_index, :]),np.sum(x[source_index+1, :]),np.sum(x[source_index+2, :]))
    # print(np.mean(np.sum(current_slice, axis=1)), np.std(np.sum(current_slice, axis=1)))

    # fig = plt.figure()
    # for j in range(current_slice.shape[0]):
    #     if hits[j]:
    #         # plt.plot(x[j, :], color='r')
    #         plt.plot(j, current_slice_counts[j], 'r.')
    #     else:
    #         # plt.plot(x[j, :], color='b')
    #         plt.plot(j, current_slice_counts[j], 'b.')
    #     # plt.axis([0, 1024, 0, 50])
    # plt.title('type{}_current{}_target{}'.format(source_type, j, center))
    # fig.savefig('./{}/{:06d}_{:04d}.png'.format(outdir, int(targetfile), j))
    # plt.close()


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
    allfilelist.sort()

    bg_training_threshold = int(len(backgroundfilelist) / 2)
    s_training_threshold = int(len(sourcefilelist) / 2)
    all_training_threshold = int(len(allfilelist) / 2)
    backgroundfilelist_train = backgroundfilelist[:bg_training_threshold]
    backgroundfilelist_test = backgroundfilelist[bg_training_threshold:]
    sourcefilelist_train = sourcefilelist[:s_training_threshold]
    sourcefilelist_test = sourcefilelist[s_training_threshold:]

    training_filelist = allfilelist[all_training_threshold:]
    testing_filelist = allfilelist[:all_training_threshold]

    validatelist = glob.glob('./test_integrations/2*.npy')
    validatelist.sort()

    # print(labels)

    #NOTE: PARALLEL STUFF
    # Parallel(n_jobs=ncores)(delayed(make_spectral_plots)(item, 'train_plots', labels) for item in sourcefilelist_train[:100])
    # Parallel(n_jobs=ncores)(delayed(make_spectral_plots)(item, 'test_plots', labels) for item in sourcefilelist_test)

    #QUESTION: Keep or take out the background lists???
    f = h5py.File('sequential_dataset_relabel_allfiles.h5', 'w')
    train = f.create_group('train')
    test = f.create_group('test')
    validate = f.create_group('validate')
    a = time.time()
    # for i in range(len(sourcefilelist_train)):
    for i in range(len(training_filelist)):
        if i % 100 == 0:
            print('{} training samples done in {} s '.format(i, time.time() - a))
        # current_file = sourcefilelist_train[i]
        current_file = training_filelist[i]
        store_sequence(current_file, train, labels)
    for i in range(len(sourcefilelist_test)):
        if i % 100 == 0:
            print('{} testing samples done in {} s '.format(i, time.time() - a))
        # current_file = sourcefilelist_test[i]
        current_file = testing_filelist[i]
        store_sequence(current_file, test, labels)
    for i in range(len(validatelist)):
        random_file = validatelist[i]
        if i % 100 == 0:
            print('{} validation samples done in {} s '.format(i, time.time() - a))
        x = np.array(np.load(random_file))
        x = x[:, 1:]
        head, tail = os.path.split(random_file)
        runname = tail[:-4]
        tostore_spectra = x
        validate.create_dataset(runname, data=tostore_spectra, compression='gzip')
    return

main()
