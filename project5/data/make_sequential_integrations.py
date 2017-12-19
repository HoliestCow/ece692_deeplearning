
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


def main():
    # only need to do this once.
    # ncores = 4
    # nsamples = 25000
    # nsamples = 10000
    nsamples = 100
    # nsamples = 1
    # nsamples = 1000
    # nsamples = 100
    sequence_length = 15

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

    bg_training_threshold = int(len(backgroundfilelist) / 2)
    s_training_threshold = int(len(sourcefilelist) / 2)
    backgroundfilelist_train = backgroundfilelist[:bg_training_threshold]
    backgroundfilelist_test = backgroundfilelist[bg_training_threshold:]
    sourcefilelist_train = sourcefilelist[:s_training_threshold]
    sourcefilelist_test = sourcefilelist[s_training_threshold:]

    validatelist = glob.glob('./test_integrations/2*.npy')
    validatelist.sort()

    # print(labels)

    # Just make sequences for background files.
    f = h5py.File('sequential_dataset_balanced_newmethod.h5', 'w')
    train = f.create_group('train')
    # background = train.create_group('background')
    # sources = train.create_group('sources')  # This will be my testing dataset
    test = f.create_group('test')
    validate = f.create_group('validate')

    l_window = [2, 4, 6, 8]
    r_window = [3, 5, 7, 9]

    # source_spectra_train = np.zeros((len(sourcefilelist_train), 5, 1024))
    source_spectra_train = []
    source_label_train = np.zeros((len(sourcefilelist_train),))
    source_chooselr_train = np.zeros((len(sourcefilelist_train),), dtype=int)
    isgood = np.zeros((len(sourcefilelist_train),), dtype=bool)
    for i in range(len(sourcefilelist_train)):
        random_file = sourcefilelist_train[i]
        if i % 100 == 0:
            print(i)
        x = np.load('./integrations/' + random_file + '.npy')
        source_index = int(labels[random_file]['time'])

        choose_lr_window = np.random.randint(len(l_window))

        leftpos = l_window[choose_lr_window]
        rightpos = r_window[choose_lr_window]
        totalpos = leftpos + rightpos

        indices = np.arange(source_index - leftpos, source_index + rightpos)
        bg_to_subtract = np.tile(x[source_index - leftpos - 1, 1:], (totalpos, 1))
        try:
            x[indices, 1:]
            isgood[i] = True
        except:
            isgood[i] = False
            # source_spectra_train += []
            source_label_train[i] = 0
            source_chooselr_train[i] = 0
            continue
        temp_spectra = np.subtract(x[indices, 1:], bg_to_subtract)
        temp_spectra[temp_spectra < 0] = 0
        source_spectra_train += [temp_spectra]
        source_label_train[i] = int(string2id[labels[random_file]['source']])
        source_chooselr_train[i] = choose_lr_window
        # fig = plt.figure()
        # plt.plot(source_spectra_train[i, 3, :])
        # fig.savefig(sourcefilelist_train[i] + '.png')
        # plt.close()
    # source_spectra_train = source_spectra_train[isgood]
    source_label_train = source_label_train[isgood]
    source_chooselr_train = source_chooselr_train[isgood]

    source_spectra_test = []
    source_label_test = np.zeros((len(sourcefilelist_test),))
    source_chooselr_test = np.zeros((len(sourcefilelist_test),), dtype=int)
    isgood = np.zeros((len(sourcefilelist_test),), dtype=bool)
    for i in range(len(sourcefilelist_test)):
        random_file = sourcefilelist_test[i]
        if i % 100 == 0:
            print(i)
        x = np.load('./integrations/' + random_file + '.npy')
        source_index = int(labels[random_file]['time'])

        choose_lr_window = np.random.randint(len(l_window))

        leftpos = l_window[choose_lr_window]
        rightpos = r_window[choose_lr_window]
        totalpos = leftpos + rightpos

        indices = np.arange(source_index - leftpos, source_index + rightpos)
        bg_to_subtract = np.tile(x[source_index - leftpos - 1, 1:], (totalpos, 1))
        try:
            x[indices, 1:]
            isgood[i] = True
        except:
            isgood[i] = False
            source_label_test[i] = 0
            source_chooselr_test[i] = 0
            continue
        temp_spectra = np.subtract(x[indices, 1:], bg_to_subtract)
        temp_spectra[temp_spectra < 0] = 0
        source_spectra_test += [temp_spectra]
        source_label_test[i] = int(string2id[labels[random_file]['source']])
        source_chooselr_test[i] = choose_lr_window

    source_label_test = source_label_test[isgood]
    source_chooselr_test = source_chooselr_test[isgood]

    l_window = [2, 4, 6, 8]
    r_window = [3, 5, 7, 9]
    start_injection = [26, 24, 22, 20]
    middle_injection = 28
    end_injection = [31, 33, 35, 37]

    a = time.time()
    for i in range(nsamples):
        random_file = backgroundfilelist_train[np.random.randint(len(backgroundfilelist_train))]
        if i % 100 == 0:
            print('{} training samples done in {} s '.format(i, time.time() - a))
        x = np.load('./integrations/' + random_file + '.npy')
        # pure background
        background_spectra = x[:, 1:]
        # grab a random 30 second segment of the background
        background_index = np.random.randint(background_spectra.shape[0] - 44)
        current_background = background_spectra[background_index:background_index + 44, :]
        sourcetype_index = np.random.randint(len(source_spectra_train))
        leftstart = start_injection[source_chooselr_train[sourcetype_index]]
        rightend = end_injection[source_chooselr_train[sourcetype_index]]
        label_list = np.zeros((current_background.shape[0],))
        # start_injection = 26
        # end_injection = 31
        injection_indices = np.arange(leftstart, rightend)
        # label_list[injection_indices] = source_label_train[sourcetype_index] * np.ones((len(injection_indices),))
        label_list[middle_injection] = source_label_train[sourcetype_index]
        measured_spectra = current_background
        # NOTE: Not exactly an injection. The source has a different background.
        measured_spectra[injection_indices, :] = measured_spectra[injection_indices, :] + source_spectra_train[sourcetype_index] # , :, :]

        # TRUNCATING THE DATASET FOR CLASS BALANCE AND SPACE CONSIDERATIONS
        # cut the spectra shut that I get a 15, 15, 1024. label_list = 000001111100000
        # index = np.arange(measured_spectra.shape[0])

        # index_generator = window(index, n=sequence_length)
        # tostore_spectra = np.zeros((0, sequence_length, 1024))
        # tostore_bgspectra = np.zeros((0, 1024))
        # tostore_labels = []
        # for index_list in index_generator:
        #     tostore_spectra = np.concatenate((tostore_spectra, measured_spectra[index_list, :].reshape((1, sequence_length, 1024))))
        #     tostore_bgspectra = np.concatenate((tostore_bgspectra, background_spectra[index_list[-1], :].reshape(1, 1024)))
        #     tostore_labels += [label_list[list(index_list)[-1]]]
        # tostore_labels = np.array(tostore_labels)
        tostore_spectra = measured_spectra
        tostore_labels = label_list
        grp = train.create_group('sample_{}'.format(i))
        grp.create_dataset('measured_spectra', data=tostore_spectra, compression='gzip')
        # grp.create_dataset('target_spectra', data=tostore_bgspectra, compression='gzip')
        grp.create_dataset('labels', data=tostore_labels, compression='gzip')

        # NOTE: THIS BLOCK IS FOR CHECKING.
        # for j in range(tostore_spectra.shape[0]):
        #     fig = plt.figure()
        #     plt.plot(tostore_spectra[j, :])
        #     plt.title('current_{}_target_{}'.format(j, middle_injection))
        #     plt.axis([0, 1024, 0, 60])
        #     fig.savefig('{:03d}_{:03d}.png'.format(i, j))
        #     plt.close()
        # if i > 10:
        #     print(fuck)

    for i in range(int(2000)):
        random_file = backgroundfilelist_test[np.random.randint(len(backgroundfilelist_test))]
        if i % 100 == 0:
            print('{} testing samples done in {} s '.format(i, time.time() - a))
        x = np.load('./integrations/' + random_file + '.npy')
        # pure background
        background_spectra = x[:, 1:]
        # inject a source at some random
        background_index = np.random.randint(background_spectra.shape[0] - 44)
        current_background = background_spectra[background_index:background_index + 44, :]
        sourcetype_index = np.random.randint(len(source_spectra_test))
        leftstart = start_injection[source_chooselr_test[sourcetype_index]]
        rightend = end_injection[source_chooselr_test[sourcetype_index]]
        label_list = np.zeros((current_background.shape[0],))
        injection_indices = np.arange(leftstart, rightend)
        # label_list[injection_indices] = source_label_test[sourcetype_index] * np.ones((len(injection_indices),))
        label_list[middle_injection] = source_label_test[sourcetype_index]
        measured_spectra = current_background
        # NOTE: Not exactly an injection. The source has a different background.
        measured_spectra[injection_indices, :] = measured_spectra[injection_indices, :] + source_spectra_test[sourcetype_index] # , :, :]

        # Truncating the set for space and class balance
        # index = np.arange(measured_spectra.shape[0])
        #
        # index_generator = window(index, n=sequence_length)
        # tostore_spectra = np.zeros((0, sequence_length, 1024))
        # tostore_bgspectra = np.zeros((0, sequence_length, 1024))
        # tostore_labels = []
        # for index_list in index_generator:
        #     tostore_spectra = np.concatenate((tostore_spectra, measured_spectra[index_list, :].reshape((1, sequence_length, 1024))))
        #     tostore_bgspectra = np.concatenate((tostore_bgspectra, background_spectra[index_list, :].reshape((1, sequence_length, 1024))))
        #     tostore_labels += [label_list[list(index_list)[-1]]]
        # tostore_labels = np.array(tostore_labels)
        tostore_spectra = measured_spectra
        tostore_labels = label_list
        grp = test.create_group('sample_{}'.format(i))
        grp.create_dataset('measured_spectra', data=tostore_spectra, compression='gzip')
        # grp.create_dataset('target_spectra', data=tostore_bgspectra, compression='gzip')
        grp.create_dataset('labels', data=tostore_labels, compression='gzip')

        # NOTE: THIS BLOCK IS FOR CHECKING.
        # for j in range(tostore_spectra.shape[0]):
        #     fig = plt.figure()
        #     plt.plot(tostore_spectra[j, :])
        #     plt.title('current_{}_target_{}'.format(j, middle_injection))
        #     plt.axis([0, 1024, 0, 60])
        #     fig.savefig('{:03d}_{:03d}.png'.format(i, j))
        #     plt.close()
        # if i > 10:
        #     print(fuck)

    for i in range(len(validatelist)):
        tostore_spectra = np.zeros((0, sequence_length, 1024))
        random_file = validatelist[i]
        if i % 100 == 0:
            print('{} validation samples done in {} s '.format(i, time.time() - a))
        x = np.array(np.load(random_file))
        x = x[:, 1:]
        head, tail = os.path.split(random_file)
        runname = tail[:-4]
        # I now I have the spectra I want to use. Now I have to batch it out into fixed segments.
        # index = np.arange(x.shape[0])
        # index_generator = window(index, n=sequence_length)
        # for index_list in index_generator:
        #     tostore_spectra = np.concatenate((tostore_spectra, x[index_list, :].reshape((1, sequence_length, 1024))))
        tostore_spectra = x
        validate.create_dataset(runname, data=tostore_spectra, compression='gzip')
        # grp.create_dataset('target_spectra', data=tostore_bgspectra, compression='gzip')
        # grp.create_dataset('labels', data=tostore_labels, compression='gzip')

    f.close()

    return

main()
