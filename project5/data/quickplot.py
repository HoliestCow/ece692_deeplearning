
import numpy as np
import matplotlib.pyplot as plt
import os

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
    labels = label_datasets()
    filename = ['106524', '109000', '100032', '105000', '107839']
    for item in filename:
        x = np.load('./integrations/{}.npy'.format(item))
        spectra = x[:, 1:]
        os.mkdir(os.path.join('poke', item))
        for i in range(spectra.shape[0]):
            fig = plt.figure()
            plt.plot(spectra[i, :])
            plt.axis([0, 1024, 0, 50])
            plt.xlabel('Channel Number (Energy Deposited)')
            plt.ylabel('Counts in Bin (Frequency)')
            plt.title('source{}_targettime{}_currenttime{}'.format(labels[item]['source'], labels[item]['time'], i))
            fig.savefig(os.path.join('poke', item, '{:03d}.png'.format(i)))
            plt.close()
    x = np.load('./test_integrations/213262.npy')
    # spectra= x[:, 1:]
    # for i in range(spectra.shape[0]):
    #     fig = plt.figure()
    #     plt.plot(spectra[i, :])
    #     fig.savefig('./poke_test/{:03d}.png'.format(i))
    return

main()
