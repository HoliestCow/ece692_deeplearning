
import numpy as np
import matplotlib.pyplot as plt
import os.path
from rebin import rebin
import glob

from joblib import Parallel, delayed
import time
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


def parse_datafiles(targetfile, binno):

    # energy = np.linspace(0, 3000, 1024)

    # filelist = ['/home/holiestcow/Documents/zephyr/datasets/muse/trainingData/109798.csv']

    # process all files.
    # filelist = glob.glob('/home/holiestcow/Documents/zephyr/datasets/muse/trainingData/*.csv')

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
            counts, energy_edges = np.histogram(energy_deposited, bins=binnumber, range=(0.0, 4000.0))
            spectra = np.vstack((spectra, counts))
            counter += 1
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
    np.save(os.path.join('./integrations', tail[:-4] + '.npy'), tosave)

    # features = nscrad_rebin(spectra, 64)
    # features_bg = features[:30, :]
    # nuisance = nscrad_get_nuisance()
    # # energy = energy_edges[1:]
    # nuisance_matrix = np.zeros((0, binno))
    # for key in nuisance:
    #     fig = plt.figure()
    #     plt.plot(nuisance[key]['energy'], nuisance[key]['counts'])
    #     fig.savefig('nuisance_{}.png'.format(key))
    #     print([0] + nuisance[key]['energy'])
    #     print(len(nuisance[key]['energy']), len(nuisance[key]['counts']))
    #
    #     out = np.array(rebin(np.array([0] + nuisance[key]['energy']), np.array(nuisance[key]['counts']),
    #                 energy_edges))
    #     out = out.reshape((1, len(out)))
    #     nuisance_matrix = np.vstack((nuisance_matrix, out))
    #
    # nuisance_matrix = nscrad_rebin(nuisance_matrix, 64)
    # nuisance_matrix = nuisance_matrix.reshape((nuisance_matrix.shape[1], nuisance_matrix.shape[0]))
    # bullshit = nscrad_get_parameters(features, features_bg, nuisance_matrix)
    return

def quickplot(random_file, labels):

    x = np.load('./integrations/' + random_file + '.npy')
    # time = x[:, 0]

    prefix = 'source{}'.format(labels[random_file]['source'])

    source = labels[random_file]['source']
    spectra = x[:, 1:]

    directory = '{}_{}'.format(prefix, random_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    print(labels[random_file]['time'])
    print(labels[random_file]['source'])
    for j in range(spectra.shape[0]):
        fig = plt.figure()
        plt.plot(spectra[j, :])
        plt.title('{}_{}_{}'.format(source, j, labels[random_file]['time']))
        plt.axis([0, 1024, 0, 50])
        fig.savefig(os.path.join(directory, '{0:03d}.png'.format(j)))
        plt.close()
    return

def main():
    # only need to do this once.
    ncores = 4
    nsamples = 20

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

    # print(labels)

    # NOTE: Slice for background segments
    # f = h5py.File('dataset.h5', 'w')
    Parallel(n_jobs=ncores)(delayed(quickplot)(filename, labels) for filename in sourcefilelist[:nsamples])
    return

main()
