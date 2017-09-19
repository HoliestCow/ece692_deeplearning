# Python 3.6
from tensorflow.examples.tutorials.mnist import input_data
import mnist_loader  # This is nielson
from network import Network as TF_Network
from np_network import Network as NP_Network  # this is nielson
from time import time
# import numpy as np
import pandas as pd
# from itertools import product, tee, repeat
import os.path
# from tabulate import tabulate
# from multiprocessing import Process, Pool
from multiprocessing import Pool
from itertools import product
from glob import glob


def main():
    # dataset
    ncores = 4  # This is used for NP_NN
    # run_name = 'NP_sigmoid'
    run_name = 'NP_relu'
    # run_name = 'TF_relu'
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    input_number = mnist.train.images.shape[1]
    output_number = mnist.train.labels.shape[1]

    # training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # FOR TF
    # learning_rate = [1E-3, 0.1, 0.3]
    # hidden_layers = [
    #     [],
    #     [30],
    #     [100, 30],
    #     [100, 300, 100]]
    # epoch_number = [50, 100, 200]
    # batch_number = [50, 100, 200]

    # FOR NP
    learning_rate = [1E-3, 0.1, 0.3]
    hidden_layers = [[], [30]]
    epoch_number = [50, 100]
    batch_number = [50, 100]

    for row in range(len(hidden_layers)):
        hidden_layers[row] = [input_number] + hidden_layers[row] + [output_number]

    # if not os.path.isfile(run_name + '.p'):
    #     tf_data = TF_NN(hidden_layers, mnist, learning_rate, epoch_number, batch_number)
    #     tf_data.to_pickle(run_name + '.p')

    # HACK: I left mnist data import intrinsic to NP_NN since the import intrinsically uses
    #       a generator (python 2 to 3 issue).
    # print('calcing np_data')
    np_data = NP_NN(hidden_layers, learning_rate, epoch_number, batch_number, ncores=ncores)
    # print(np_data)
    np_data.to_pickle(run_name + '.p')
    # analysis portion
    # tf_data = pd.read_pickle(run_name + '.p')
    # print(tf_data)
    # np_data = pd.read_pickle(run_name + '.p')
    # print('TF')
    # gambit(tf_data, learning_rate=1E-3, prefix=run_name)
    # print('NP')
    # gambit(np_data, learning_rate=0.1, prefix=run_name)
    return


def NP_NN(hidden_layers, learning_rate, epoch_number, batch_number, ncores=4):
    # for hidden_layer, learning, epoch, batchsize in zip(hidden_layers,
    #                                                     learning_rate,
    #                                                     epoch_number,
    #                                                     batch_number):
    todo = list(product(*[hidden_layers, learning_rate, epoch_number, batch_number]))
    # training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    for i in range(len(todo)):
        todo[i] = (i,) + todo[i]
    if not len(todo) == len(glob('*.out')):
        with Pool(processes=ncores) as p:
            p.starmap(execute_NP_NN, todo)
            [p.apply_async(execute_NP_NN, todo) for i in range(ncores)]
    data = parse_txtfiles()
    data = pd.DataFrame.from_records(
        data,
        columns=['model', 'learning_rate', 'score', 'total_time', 'train_time', 'eval_time',
                 'epoch_number', 'batch_number'])
    return data


def parse_txtfiles():
    filelist = glob('*.out')
    output_container = []
    for item in filelist:
        f = open(item, 'r')
        a = f.readlines()
        line = a[0].strip()  # one line per file.
        [model, learning_rate, score, total_time, train_time, eval_time,
         epoch_number, batch_number] = line.split(', ')
        learning_rate = float(learning_rate)
        score = float(score)
        total_time = float(total_time)
        train_time = float(train_time)
        eval_time = float(eval_time)
        epoch_number = int(epoch_number)
        batch_number = int(batch_number)
        output_container += [(model, learning_rate, score, total_time,
                              train_time, eval_time, epoch_number, batch_number)]
    return output_container


def execute_NP_NN(run_id, hidden_layer, learning, epoch, batch):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print('NP Calculating:\nNN: {}\nLR: {}\nEN: {}\nBA: {}'.format(hidden_layer,
                                                                   learning,
                                                                   epoch,
                                                                   batch))
    network_label = network_label_generator(hidden_layer)
    net = NP_Network(hidden_layer)
    bigtic = time()
    net.SGD(training_data, epoch, batch, learning)
    # def SGD(self, training_data, epochs, mini_batch_size, eta,
            # test_data=None)
    train_time = net.train_time
    tic = time()
    score = net.evaluate(test_data)
    toc = time()
    eval_time = toc-tic
    bigtoc = time()
    total_time = bigtoc-bigtic
    # output_container = (network_label,
    #                     learning,
    #                     score,
    #                     total_time,
    #                     train_time,
    #                     eval_time,
    #                     epoch,
    #                     batch)
    # return output_container
    # CSV style for multiprocessing
    f = open('{}.out'.format(run_id), 'w')
    f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
        network_label, learning, score, total_time, train_time, eval_time, epoch, batch))
    f.close()
    return


def TF_NN(hidden_layers, data, learning_rate, epoch_number, batch_number):

    output_container = []
    # for hidden_layer, learning, epoch, batch in zip(hidden_layers,
    #                                                 learning_rate,
    #                                                 epoch_number,
    #                                                 batch_number):
    todo = list(product(*[hidden_layers, learning_rate, epoch_number, batch_number]))
    for item in todo:
        hidden_layer = item[0]
        learning = item[1]
        epoch = item[2]
        batch = item[3]
        print('TF Calculating:\nNN: {}\nLR: {}\nEN: {}\nBA: {}'.format(hidden_layer,
                                                                     learning,
                                                                     epoch,
                                                                     batch))
        network_label = network_label_generator(hidden_layer)
        [score, total_time, train_time, eval_time] = execute_neural_network(
            dataset=data,
            neural_network=hidden_layer,
            learning_rate=learning,
            epoch_number=epoch,
            batch_number=batch)
        output_container += [(network_label,
                              learning,
                              score,
                              total_time,
                              train_time,
                              eval_time,
                              epoch,
                              batch)]
    data = pd.DataFrame.from_records(
        output_container,
        columns=['model', 'learning_rate', 'score', 'total_time', 'train_time', 'eval_time',
                 'epoch_number', 'batch_number'])

    return data


def network_label_generator(x):
    out = ''
    for i in range(len(x)):
        # out += str(x[i])+'_'
        out += '{}_'.format(x[i])
    out = out[:-1]
    return out


def execute_neural_network(dataset=None, neural_network=None, learning_rate=0.5,
                           epoch_number=100, batch_number=1000):
    total_tic = time()
    model = TF_Network(dataset)
    model.define_layers(output_numbers=neural_network)
    model.build_graph(learning_rate=learning_rate)
    train_tic = time()
    model.train(epoch_number=epoch_number, batch_number=batch_number)
    train_toc = time()
    eval_tic = time()
    score = model.eval()
    eval_toc = time()
    total_toc = time()
    total_time = total_toc - total_tic
    train_time = train_toc - train_tic
    eval_time = eval_toc - eval_tic
    return score, total_time, train_time, eval_time

main()

