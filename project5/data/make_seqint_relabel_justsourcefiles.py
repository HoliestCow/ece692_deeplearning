
import numpy as np
import matplotlib.pyplot as plt
import os.path
import glob

from joblib import Parallel, delayed
import h5py
from itertools import islice
import time
import scipy.signal


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
    len_limit = 32
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

def store_sequence(targetfile, filehandle, labels, plot_dir=None):


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

    # left = source_index - 14
    # right = source_index + 15
    # left = source_index - 29
    # right = source_index + 30
    # left = source_index - 44
    # right = source_index + 45
    # left = source_index - 59
    # right = source_index + 60
    # left = source_index - 89
    # right = source_index + 90
    # left = source_index - 119
    # right = source_index + 120
    # left = source_index - 149
    # right = source_index + 150
    left = 0
    right = x.shape[0]
    if left < 0:
        left = 0
    if right >= x.shape[0]:
        right = x.shape[0]

    current_slice = x
    current_slice_counts = np.sum(current_slice, axis=1)
    modified_slice_counts = reject_outliers(current_slice_counts, m=5)
    # current_slice_count_mean = np.mean(modified_slice_counts)
    current_slice_count_median = np.median(current_slice_counts)
    # current_slice_count_min = np.min(current_slice_counts)
    current_slice_count_std = np.std(modified_slice_counts)
    hits = np.zeros((current_slice.shape[0], ), dtype=bool)

    # NOTE: INSTEAD OF threshold based on counting. We should makr peaks using
    # scipy.signal.find_peaks_cwt(), find the peaks, find the nearest one, then somehow mark the entire peak.
    threshold = current_slice_count_median + (0.1 * current_slice_count_std)
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

    # ofinterest = []
    ofinterest = ['100033']

    # Plot the hits:
    if plot_dir is not None:
        hits = tostore_labels > 0.5
        x = np.arange(0, len(tostore_labels))
        fig = plt.figure()
        # print(x[~hits])
        # print(np.sum(tostore_spectra[~hits, :], axis=1))
        counts = np.sum(tostore_spectra, axis=1)
        plt.plot(x[hits], counts[hits], 'r.', label='threat')
        plt.plot(x[~hits], counts[~hits], 'b.', label='background')
        plt.plot(x[source_index], counts[source_index], 'g*', label='threat - PoCA')
        # plt.axis([0, 1024, 0, 50])
        plt.xlabel('Time (s)')
        plt.ylabel('Count Rate (cps)')
        plt.legend()
        plt.axis([0, x[-1], 0, np.max(np.sum(tostore_spectra, axis=1))])
        fig.savefig('./{}/{}_counts'.format(plot_dir, targetfile))
        plt.close()

        if targetfile in ofinterest:
            # x = np.arange(0, len(tostore_labels))
            for j in range(len(tostore_labels)):
                fig = plt.figure()
                if tostore_labels[j] == 0:
                    plt.plot(tostore_spectra[j, :], 'b.')
                elif j == source_index:
                    plt.plot(tostore_spectra[j, :], 'g*')
                else:
                    plt.plot(tostore_spectra[j, :], 'r.')
                try:
                    os.mkdir('./{}/{}'.format(plot_dir, targetfile))
                except:
                    pass
                # plt.title('source_{}_time_{}'.format(source_type, source_index))
                plt.axis([0, 1024, 0, 50])
                plt.xlabel('Channel Number')
                plt.ylabel('Counts')
                fig.savefig('./{}/{}/{:03d}'.format(plot_dir, targetfile, j))
                plt.close()

        # for i in range(len(tostore_labels)):
        #     if tostore_labels[i] != 0:
        #         fig = plt.figure()
        #         plt.plot(tostore_spectra[i, :])
        #         plt.title('source{}_time{}'.format(source_type, source_index))
        #         plt.axis([0, 1024, 0, 50])
        #         fig.savefig('./{}/{}_{}'.format(plot_dir, targetfile, i))
        #         plt.close()

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
    # filelist = glob.glob('/home/holiestcow/Documents/zephyr/datasets/muse/trainingData/runID-*.csv')
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

    bg_training_threshold = int(len(backgroundfilelist) / 2)
    s_training_threshold = int(len(sourcefilelist) / 2)
    backgroundfilelist_train = backgroundfilelist[:bg_training_threshold]
    backgroundfilelist_test = backgroundfilelist[bg_training_threshold:]
    sourcefilelist_train = sourcefilelist[:s_training_threshold]
    sourcefilelist_test = sourcefilelist[s_training_threshold:]

    validatelist = glob.glob('./test_integrations/runID-*.npy')
    validatelist.sort()

    # print(labels)

    #NOTE: PARALLEL STUFF
    # Parallel(n_jobs=ncores)(delayed(make_spectral_plots)(item, 'train_plots', labels) for item in sourcefilelist_train)
    # Parallel(n_jobs=ncores)(delayed(make_spectral_plots)(item, 'test_plots', labels) for item in sourcefilelist_test)

    #QUESTION: Keep or take out the background lists???
    f = h5py.File('sequential_dataset_relabel_allseconds.h5', 'w')
    train = f.create_group('train')
    test = f.create_group('test')
    # validate = f.create_group('validate')
    a = time.time()
    for i in range(len(sourcefilelist_train)):
        if i % 100 == 0:
            print('{} training samples done in {} s'.format(i, time.time() - a))
        current_file = sourcefilelist_train[i]
        store_sequence(current_file, train, labels, 'train_plots')
    # for i in range(len(backgroundfilelist_train)):
    #     if i % 100 == 0:
    #         print('{} training samples done in {} s '.format(i, time.time() - a))
    #     current_file = backgroundfilelist_train[i]
    #     store_sequence(current_file, train, labels)
    for i in range(len(sourcefilelist_test)):
        if i % 100 == 0:
            print('{} testing samples done in {} s'.format(i, time.time() - a))
        current_file = sourcefilelist_test[i]
        store_sequence(current_file, test, labels)
    # for i in range(len(backgroundfilelist_test)):
    #     if i % 100 == 0:
    #         print('{} training samples done in {} s '.format(i, time.time() - a))
    #     current_file = backgroundfilelist_test[i]
    #     store_sequence(current_file, train, labels)
    f.close()

    # NOTE: JUST TO CREATE THE VALIDATION. ONLY NEED TO DO THIS ONCE

    # g = h5py.File('sequential_dataset_relabel_validationonly.h5', 'w')
    # validate = g.create_group('validate')
    # for i in range(len(validatelist)):
    #     random_file = validatelist[i]
    #     if i % 100 == 0:
    #         print('{} validation samples done in {} s'.format(i, time.time() - a))
    #     x = np.array(np.load(random_file))
    #     x = x[:, 1:]
    #     head, tail = os.path.split(random_file)
    #     runname = tail[6:-4]
    #     tostore_spectra = x
    #     validate.create_dataset(runname, data=tostore_spectra, compression='gzip')
    # g.close()

    # h = h5py.File('sequential_dataset_relabel_testset_validationonly.h5', 'w')
    # validate_h = h.create_group('validate')
    # for i in range(len(sourcefilelist_test)):
    #     random_file = sourcefilelist_test[i]
    #     if i % 100 == 0:
    #         print('{} validation samples done in {} s'.format(i, time.time() - a))
    #     x = np.array(np.load('./train_integrations/' + random_file + '.npy'))
    #     x = x[:, 1:]
    #     # head, tail = os.path.split(random_file)
    #     runname = random_file
    #     # runname = tail[:-4]
    #     tostore_spectra = x
    #     # print(runname)
    #     validate_h.create_dataset(runname, data=tostore_spectra, compression='gzip')
    # h.close()
    return

main()
