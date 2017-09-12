# Python 3.6
from tensorflow.examples.tutorials.mnist import input_data
import mnist_loader  # This is nielson
from network import Network as TF_Network
from np_network import Network as NP_Network  # this is nielson
from time import time
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
# from itertools import product, tee, repeat
import os.path
# from tabulate import tabulate
# from multiprocessing import Process, Pool
from multiprocessing import Pool
from itertools import product
from glob import glob

def main():
    # dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    ncores = 4
    input_number = mnist.train.images.shape[1]
    output_number = mnist.train.labels.shape[1]

    # training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # FOR TF
    # learning_rate = [1E-3, 0.1, 0.3]
    # hidden_layers = [
    #     [],
    #     [5],
    #     [50, 25],
    #     [30]]
    # epoch_number = [100, 200, 500, 1000]
    # batch_number = [10, 50, 100]

    # FOR NP
    learning_rate = [0.1, 0.3]
    hidden_layers = [[], [30]]
    epoch_number = [10, 20]
    batch_number = [50, 100]

    for row in range(len(hidden_layers)):
        hidden_layers[row] = [input_number] + hidden_layers[row] + [output_number]

    if not os.path.isfile('tf_data.p'):
        tf_data = TF_NN(hidden_layers, mnist, learning_rate, epoch_number, batch_number)
        print(tf_data)
        tf_data.to_pickle('tf_data.p')
    if not os.path.isfile('np_data.p'):
        # HACK: I left mnist data import intrinsic to NP_NN since the import intrinsically uses
        #       a generator (python 2 to 3 issue).
        np_data = NP_NN(hidden_layers, learning_rate, epoch_number, batch_number, ncores=ncores)
        print(np_data)
        np_data.to_pickle('np_data.p')
    # analysis portion
    tf_data = pd.read_pickle('tf_data.p')
    np_data = pd.read_pickle('np_data.p')
    gambit(tf_data, learning_rate=0.1, prefix='TF')
    gambit(np_data, learning_rate=0.1, prefix='NP')
    return


def gambit(data, learning_rate=0.3, prefix=None):
    print(data)
    # print(tabulate(data, headers='keys', tablefmt='psql'))
    toplot = data[data['learning_rate'] == learning_rate]
    fig, ax = poke_dataframe(toplot,
                             'total_time',
                             'score',
                             labeler='model')
    fig.savefig('{}_lr{}_model_totaltime.png'.format(prefix, learning_rate))
    plt.close(fig)

    fig, ax = poke_dataframe(toplot,
                             'train_time',
                             'score',
                             labeler='model')
    fig.savefig('{}_lr{}_model_traintime.png'.format(prefix, learning_rate))
    plt.close(fig)

    fig, ax = poke_dataframe(toplot,
                             'eval_time',
                             'score',
                             labeler='model')
    fig.savefig('{}_lr{}_model_evaltime.png'.format(prefix, learning_rate))
    plt.close(fig)

    # fig, ax = poke_dataframe(data,
    #                          'learning_rate',
    #                          'score',
    #                          labeler='model',
    #                          specific_label='784_30_10')
    # fig.savefig('{}_model_784_10_totaltime.png'.format(prefix))
    # plt.close(fig)

    fig, ax = poke_dataframe(data,
                             'learning_rate',
                             'score',
                             specific_model='784_30_10')
    fig.savefig('{}_model_784_30_10_totaltime.png'.format(prefix))
    plt.close(fig)

    fig, ax = poke_dataframe(data,
                             'epoch_number',
                             'score',
                             labeler='batch_number',
                             specific_model='784_30_10')
    fig.savefig('{}_epoch_784_30_10_epoch.png'.format(prefix))
    plt.close(fig)

    fig, ax = poke_dataframe(data,
                             'epoch_number',
                             'score',
                             labeler='batch_number',
                             specific_model='784_30_10')
    fig.savefig('{}_epoch_784_30_10_epoch.png'.format(prefix))
    plt.close(fig)
    return


def NP_NN(hidden_layers, learning_rate, epoch_number, batch_number, ncores=4):
    output_container = []
    # for hidden_layer, learning, epoch, batchsize in zip(hidden_layers,
    #                                                     learning_rate,
    #                                                     epoch_number,
    #                                                     batch_number):
    todo = list(product(*[hidden_layers, learning_rate, epoch_number, batch_number]))
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print(training_data, validation_data, test_data)
    # index = list(range(len(todo)))
    # todo = list(zip(index, todo, repeat(training_data), repeat(test_data)))
    # zip(index, todo, repeat(
    for i in range(len(todo)):
        todo[i] = (i,) + todo[i] + (training_data,) + (test_data,)
    with Pool(processes=ncores) as p:
        p.starmap(execute_NP_NN, todo)
        # result = [p.apply_async(execute_NP_NN, todo) for i in range(ncores)]
    # p = Process(target=execute_NP_NN, args=todo)
    # p.start()
    # p.join()
    print('we made it fam')
    # for item in todo:
    #     [run_id, hidden_layer, learning, epoch, batch] = item
    #     print('NP Calculating:\nNN: {}\nLR: {}\nEN: {}\nBA: {}'.format(hidden_layer,
    #                                                                 learning,
    #                                                                 epoch,
    #                                                                 batch))
    #     output_container += [execute_NP_NN(run_id, hidden_layer, learning, epoch, batch)]

    # I need to figure out the csv parsing glob shit here.
    data = parse_txtfiles()
    print(data)
    stop
    data = pd.DataFrame.from_records(
        output_container,
        columns=['model', 'learning_rate', 'score', 'total_time', 'train_time', 'eval_time',
                 'epoch_number', 'batch_number'])
    return data


def parse_txtfiles():
    filelist = glob('./*.txt')
    output_container = []
    for item in filelist:
        f = open(item, 'r')
        a = f.readlines()
        line = a[0]  # one line per file.
        output_container += line.split(',')
    return output_container


def execute_NP_NN(run_id, hidden_layer, learning, epoch, batch,
                  training_data, test_data):
    # training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
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


def poke_dataframe(data, x, y, labeler=None, specific_label=None, specific_model=None):
    fig, ax = plt.subplots()
    color_wheel = mcd.XKCD_COLORS
    color_list = list(mcd.XKCD_COLORS.keys())[::10]
    # toplot = data.loc[0.5]
    if specific_model:
        data = data[data['model'] == specific_model]
    if labeler:
        if specific_label:
            ax = data[data[labeler] == specific_label].plot(
                x=x,
                y=y,
                ax=ax,
                kind='scatter',
                label=specific_label)
        else:
            unique_labels = list(set(data[labeler]))
            counter = 0
            for item in unique_labels:
                # print(toplot[toplot['model'] == item])
                ax = data[data[labeler] == item].plot(
                    x=x,
                    y=y,
                    ax=ax,
                    kind='scatter',
                    label=item,
                    c=color_wheel[color_list[counter]])
                counter += 1
    else:
        ax = data.plot(
            x=x,
            y=y,
            ax=ax,
            kind='scatter')
    return fig, ax


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

