
import numpy as np
import matplotlib.pyplot as plt
import re
import glob


def main():
    filelist = glob.glob('*.out')
    container = {}
    keys = []
    for item in filelist:
        print(item)
        if item == 'sgd.out' or item == 'nodatanormoraugmetnation.out':
            continue
        f = open(item, 'r')
        a = f.readlines()
        epochs = []
        training_acc = []
        testing_acc = []
        for line in a:
            if re.search('50000/50000', line):
                yolo = line.strip()
                parsed = yolo.split(' | ')

                epochs += [int(parsed[1][-4:])]

                training_acc += [float(parsed[2][-6:])]

                temp = parsed[-1].split(' -- ')[0]

                testing_acc += [float(temp[-6:])]
        print(epochs)
        print(training_acc)
        print(testing_acc)

        container[item[:-4]] = {'epochs': np.array(epochs),
                                'training_accuracy': np.array(training_acc),
                                'testing_accuracy': np.array(testing_acc)}
        keys += [item[:-4]]
        f.close()
    fig = plt.figure()
    for key in keys:
        plt.plot(container[key]['epochs'], container[key]['testing_accuracy'])
    plt.xlabel('epoch number')
    plt.ylabel('testing_accuracy')
    plt.legend(keys)
    fig.savefig('testing_accuracy.png')

    fig = plt.figure()
    for key in keys:
        plt.plot(container[key]['epochs'], container[key]['training_accuracy'])
    plt.xlabel('epoch number')
    plt.ylabel('training_accuracy')
    plt.legend(keys)
    fig.savefig('training_accuracy.png')
    return


main()