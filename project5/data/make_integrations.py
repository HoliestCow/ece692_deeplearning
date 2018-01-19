
import numpy as np
import matplotlib.pyplot as plt
import os.path
from rebin import rebin
import glob

from joblib import Parallel, delayed
import time


def nscrad_get_parameters(x, x_bg, x_nuisance):
    # print(np.sum(x[0, :]))
    # stop
    # mean_bg = np.mean(x_bg, axis=0)
    # bg_covariance = np.matmul(mean_bg.reshape((len(mean_bg), 1)),
    #                           mean_bg.reshape((1, len(mean_bg))))
    bg_covariance = np.matmul(x_nuisance.reshape((x_nuisance.shape[0], x_nuisance.shape[1])),
                              x_nuisance.reshape((x_nuisance.shape[1], x_nuisance.shape[0])))
    mean_bg = bg_covariance
    feature_length = x.shape[1]
    out = np.zeros((x.shape[0], feature_length))
    T = np.zeros((feature_length, feature_length))
    # T[:, 0] = np.ones((feature_length, 1))
    for i in range(feature_length):
        if i != feature_length-1:
            T[i, i+1] = - mean_bg[0] / mean_bg[i]
        T[i, 0] = 1

    # FEATURES
    c = np.matmul(x, T)
    # nuisance
    A = np.matmul(T, x_nuisance)
    # BG Covariance
    print(bg_covariance.shape, T.shape)
    print(bg_covariance)
    np.linalg.inv(bg_covariance)
    stop
    # sigma = np.matmul(bg_covariance, T.reshape((T.shape[1], T.shape[0])))
    sigma = np.matmul(T, bg_covariance)
    sigma = np.matmul(sigma, bg_covariance)

    A_t = A.reshape(A.shape[1], A.shape[0])
    sigma_inv = np.linalg.inv(sigma)
    middle = np.linalg.inv(np.matmul(A_t, np.matmul(sigma_inv, A)))
    left = np.matmul(A, middle)
    right = np.matmul(A_t, sigma_int)
    P = np.matmul(left, right)
    print(P.shape)
    stop
    return


def nscrad_get_nuisance():
    f = open('./nuisance.csv', 'r')
    a = f.readlines()
    nuisance_sources = {}
    for i in range(len(a)):
        b = a[i].strip()
        b_parsed = b.split(',')
        isotope = b_parsed[0]
        if i == 0:
            current_isotope = isotope
            energy = []
            counts = []

        if current_isotope != isotope:
            nuisance_sources[current_isotope] = {}
            nuisance_sources[current_isotope]['energy'] = energy
            nuisance_sources[current_isotope]['counts'] = counts
            current_isotope = isotope
            energy = []
            counts = []
        elif len(b_parsed) == 0 or i == len(a) - 1:
            nuisance_sources[current_isotope] = {}
            nuisance_sources[current_isotope]['energy'] = energy
            nuisance_sources[current_isotope]['counts'] = counts
            current_isotope = isotope
            energy = []
            counts = []
        else:
            energy += [float(b_parsed[2])]
            counts += [float(b_parsed[3])]
    return nuisance_sources


def nscrad_rebin(x, n):
    # print(np.sum(x[0, :]))
    # stop
    feature_length = x.shape[1] / n
    out = np.zeros((x.shape[0], feature_length))
    for i in range(feature_length):
        out[:, i] = np.sum(x[:, (i * n) - 1: (i * n)], axis=1)
    return out


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
        source_time = parsed[1]
        source_labels[filename] = {'source': id2string[int(source)],
                                   'time': float(source_time)}
    f.close()
    return source_labels


def parse_datafiles(targetfile, binno, outdir):
    # energy = np.linspace(0, 3000, 1024)

    # filelist = ['/home/holiestcow/Documents/zephyr/datasets/muse/trainingData/109798.csv']

    # process all files.
    # filelist = glob.glob('/home/holiestcow/Documents/zephyr/datasets/muse/trainingData/*.csv')

    item = targetfile
    # for item in filelist:
    f = open(item, 'r')
    a = f.readlines()

    # binno = 1024

    counter = 0

    spectra = np.zeros((0, binno))
    timetracker = 0
    energy_deposited = []

    totaltimetracker = 0
    for i in range(len(a)):
        b = a[i].strip()
        b_parsed = b.split(',')
        event_time = int(b_parsed[0])
        energy_deposited += [float(b_parsed[1])]

        timetracker += event_time
        totaltimetracker += event_time

        # print(timetracker)

        if timetracker >= 1E6:
            timetracker = 0
            source_id = 0
            counts, energy_edges = np.histogram(energy_deposited, bins=1024, range=(0.0, 4000.0))
            spectra = np.vstack((spectra, counts))
            energy_deposited = []
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
    print(tail, spectra.shape, totaltimetracker / 1E6)
    # f = open(os.path.join('./integrations', tail), 'w')
    # np.savetxt(f, tosave, delimiter=',')
    # f.close()
    np.save(os.path.join(outdir, tail[:-4] + '.npy'), tosave)

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


def main():
    # only need to do this once.
    ncores = 4
    binnumber = 1024
    filelist = glob.glob('/home/holiestcow/Documents/zephyr/datasets/muse/trainingData/1*.csv')
    Parallel(n_jobs=ncores)(delayed(parse_datafiles)(item, binnumber, 'train_integrations') for item in filelist)
    filelist = glob.glob('/home/holiestcow/Documents/zephyr/datasets/muse/listData/runID*.csv')
    Parallel(n_jobs=ncores)(delayed(parse_datafiles)(item, binnumber, 'test_integrations') for item in filelist)
    # labels = label_datasets()
    # print(labels)

    return

main()