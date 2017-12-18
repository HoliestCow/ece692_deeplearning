
import numpy as np
import matplotlib.pyplot as plt
import os.path
from rebin import rebin
import glob

from joblib import Parallel, delayed
import time
import h5py
from random import shuffle


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

def parse_datafiles(targetfile, binno, outdir):

    item = targetfile
    # for item in filelist:
    f = open(item, 'r')
    a = f.readlines()

    binnumber = 1024

    counter = 0

    spectra = np.zeros((0, binnumber))
    timetracker = 0
    energy_deposited = []
    for i in range(len(a)):
        b = a[i].strip()
        b_parsed = b.split(',')
        event_time = int(b_parsed[0])
        energy_deposited += [float(b_parsed[1])]

        timetracker += event_time

        # print(timetracker)

        if timetracker >= 1E6:
            timetracker = 0
            source_id = 0
            counts, energy_edges = np.histogram(energy_deposited, bins=binnumber, range=(0.0, 3000.0))
            spectra = np.vstack((spectra, counts))
            counter += 1
            energy_deposited = []
        # if counter >= 100:
        #     break
    # print(np.sum(spectra[0, :]))
    time = np.linspace(0, counter, counter)
    time = time.reshape((time.shape[0], 1))
    # print(time.shape, spectra.shape)
    tosave = np.hstack((time, spectra))

    f.close()
    head, tail = os.path.split(item)
    print(tail, spectra.shape)
    # f = open(os.path.join('./integrations', tail), 'w')
    # np.savetxt(f, tosave, delimiter=',')
    # f.close()
    np.save(os.path.join(outdir, tail[:-4] + '.npy'), tosave)
    return


def main():
    # only need to do this once.
    binnumber = 1024
    ncores = 4
    # nsamples = 100000
    # nsamples = 1
    filename = 'naive_dataset_justsources'

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

    # sequence_length = 30  # 30 seconds used to guess the next one
    filelist = glob.glob('./integrations/1*.npy')
    # shuffle(filelist)
    # Parallel(n_jobs=ncores)(delayed(parse_datafiles)(item, binnumber, './integrations') for item in filelist)
    test_filelist = glob.glob('./test_integrations/2*.npy')
    # Parallel(n_jobs=ncores)(delayed(parse_datafiles)(item, binnumber, './test_integrations') for item in test_filelist)

    labels = label_datasets()

    # NOTE: Slice for background segments
    f = h5py.File(filename + '.h5', 'w')
    train = f.create_group('training')
    test = f.create_group('testing')

    number_of_testing_files = 4800
    number_of_training_files = len(labels.keys()) - number_of_testing_files  # The last 10000 are for testing

    test2train_ratio = number_of_testing_files / number_of_training_files


    source_filelist = []
    counter = 0
    for item in filelist:
        head, tail = os.path.split(item)
        # print(labels[tail[:-4]]['source'])
        if labels[tail[:-4]]['source'] != 'Background':
            source_filelist += [item]
        counter += 1

    tostore_spectra = np.zeros((len(source_filelist), 1024))
    tostore_labels = np.zeros((len(source_filelist), 1))
    for i in range(int(len(source_filelist) / 2)):
        # create training dataset
        head, tail = os.path.split(source_filelist[i])
        current_key = tail[:-4]
        x = np.load('./integrations/' + current_key + '.npy')
        # time = x[:, 0]

        start = int(labels[current_key]['time'])
        source = labels[current_key]['source']

        spectra = x[start, 1:]
        tostore_spectra[i, :] = spectra
        tostore_labels[i] = int(string2id[source]) - 1
        # g = train.create_group('sample_' + str(i))
        # g.create_dataset('spectra', data=spectra, compression='gzip')
        # g.create_dataset('spectra', data=spectra)
        # g.create_dataset('label', data=int(string2id[source]))
    train.create_dataset('spectra', data=tostore_spectra, compression='gzip')
    train.create_dataset('labels', data=tostore_labels, compression='gzip')

    for i in range(int(len(source_filelist) / 2)):
        # create training dataset
        random_file = source_filelist[i + int(len(source_filelist) / 2)]
        head, tail = os.path.split(random_file)
        current_key = tail[:-4]
        x = np.load('./integrations/' + current_key + '.npy')
        # time = x[:, 0]

        start = int(labels[current_key]['time'])
        source = labels[current_key]['source']

        spectra = x[start, 1:]
        tostore_spectra[i, :] = spectra
        tostore_labels[i] = int(string2id[source]) - 1
        # g = test.create_group('sample_' + str(i))
        # g.create_dataset('spectra', data=spectra, compression='gzip')
        # g.create_dataset('label', data=int(string2id[source]))
    test.create_dataset('spectra', data=tostore_spectra, compression='gzip')
    test.create_dataset('labels', data=tostore_labels, compression='gzip')

    f.close()

    return

main()
