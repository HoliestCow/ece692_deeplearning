
import numpy as np
import matplotlib.pyplot as plt
import os.path
from rebin import rebin
import glob
from random import shuffle

from joblib import Parallel, delayed
# import time
import h5py


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
            # print(max(energy_deposited))
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
    nsamples = 50000
    # nsamples = 0
    filename = 'naive_dataset'

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
    filelist = glob.glob('/home/holiestcow/Documents/zephyr/datasets/muse/trainingData/1*.csv')
    # shuffle(filelist)
    # Parallel(n_jobs=ncores)(delayed(parse_datafiles)(item, binnumber, './integrations') for item in filelist)
    # test_filelist = glob.glob('/home/holiestcow/Documents/zephyr/datasets/muse/testData/2*.csv')
    # HACK: RIGHT HERE
    test_filelist = glob.glob('./test_integrations/2*.npy')
    # Parallel(n_jobs=ncores)(delayed(parse_datafiles)(item, binnumber, './test_integrations') for item in test_filelist)

    labels = label_datasets()

    # NOTE: Slice for background segments
    f = h5py.File(filename + '.h5', 'w')
    train = f.create_group('training')
    test = f.create_group('testing')
    validation = f.create_group('validation')

    number_of_testing_files = 4800
    number_of_training_files = len(labels.keys()) - number_of_testing_files  # The last 10000 are for testing

    test2train_ratio = number_of_testing_files / number_of_training_files

    tostore_spectra = np.zeros((nsamples, 1024))
    tostore_labels = np.zeros((nsamples, 1))
    filelist = list(labels.keys())
    for i in range(nsamples):
        # create training dataset
        random_file = filelist[np.random.randint(number_of_training_files)]
        if i % 100 == 0:
            print('training sample: {}'.format(i))
        x = np.load('./integrations/' + random_file + '.npy')
        # time = x[:, 0]

        start = np.random.randint(x.shape[0])
        source = 'Background'

        if labels[random_file]['source'] != 'Background' and start >= 30:
            start = int(labels[random_file]['time'])
            source = labels[random_file]['source']
        spectra = x[start, 1:]
        tostore_spectra[i, :] = spectra
        tostore_labels[i] = int(string2id[source])
        # g = train.create_group('sample_' + str(i))
        # g.create_dataset('spectra', data=spectra, compression='gzip')
        # g.create_dataset('spectra', data=spectra)
        # g.create_dataset('label', data=int(string2id[source]))
    train.create_dataset('spectra', data=tostore_spectra, compression='gzip')
    train.create_dataset('labels', data=tostore_labels, compression='gzip')

    tostore_spectra = np.zeros((int(nsamples * test2train_ratio), 1024))
    tostore_labels = np.zeros((int(nsamples * test2train_ratio), 1))
    for i in range(int(nsamples * test2train_ratio)):
        # create training dataset
        random_file = filelist[number_of_training_files + np.random.randint(number_of_testing_files)]
        if i % 100 == 0:
            print('testing sample: {}'.format(i))
        x = np.load('./integrations/' + random_file + '.npy')
        # time = x[:, 0]

        start = np.random.randint(x.shape[0])
        source = 'Background'

        if labels[random_file]['source'] != 'Background' and start >= 30:
            start = int(labels[random_file]['time'])
            source = labels[random_file]['source']

        spectra = x[start, 1:]
        tostore_spectra[i, :] = spectra
        tostore_labels[i] = int(string2id[source])
        # g = test.create_group('sample_' + str(i))
        # g.create_dataset('spectra', data=spectra, compression='gzip')
        # g.create_dataset('label', data=int(string2id[source]))
    test.create_dataset('spectra', data=tostore_spectra, compression='gzip')
    test.create_dataset('labels', data=tostore_labels, compression='gzip')

    # this is for the validation set, where i have to analyze
    #     each file individual
    for i in range(len(test_filelist)):
        if i % 100 == 0:
            print('validation sample {}'.format(i))
        filename = test_filelist[i]
        head, tail = os.path.split(filename)
        dataname = tail[:-4]
        x = np.load(os.path.join('./test_integrations', dataname + '.npy'))
        t = x[:, 0]
        spectra = x[:, 1:]
        file_sample = validation.create_group(dataname)
        file_sample.create_dataset('time', data=t, compression='gzip')
        file_sample.create_dataset('spectra', data=spectra, compression='gzip')


    f.close()

    return

main()
