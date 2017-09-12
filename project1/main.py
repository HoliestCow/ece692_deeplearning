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
from itertools import product
import os.path
from multiprocessing import Pool


def main():
    # dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    input_number = mnist.train.images.shape[1]
    output_number = mnist.train.labels.shape[1]

    # training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    learning_rate = [1E-3, 0.1, 0.3]
    hidden_layers = [
        [],
        [5],
        [10, 10],
        [30]]
    epoch_number = [5, 15, 50]
    batch_number = [5, 50, 100]

    for row in range(len(hidden_layers)):
        hidden_layers[row] = [input_number] + hidden_layers[row] + [output_number]

    if not os.path.isfile('tf_data.p'):
        tf_data = TF_NN(hidden_layers, mnist, learning_rate, epoch_number, batch_number)
        print(tf_data)
        tf_data.to_pickle('tf_data.p')
    if not os.path.isfile('np_data.p'):
        # HACK: I left mnist data import intrinsic to NP_NN since the import intrinsically uses
        #       a generator (python 2 to 3 issue).
        np_data = NP_NN(hidden_layers, learning_rate, epoch_number, batch_number)
        print(np_data)
        np_data.to_pickle('np_data.p')
    # analysis portion
    tf_data = pd.read_pickle('tf_data.p')
    np_data = pd.read_pickle('np_data.p')
    gambit(tf_data, prefix='TF')
    gambit(np_data, prefix='NP')
    return


def gambit(data, prefix=None):
    toplot = data[data['learning_rate'] == 0.5]
    fig, ax = poke_dataframe(toplot,
                             'total_time',
                             'score',
                             labeler='model')
    fig.savefig('{}_lr0.5_model_totaltime.png'.format(prefix))
    plt.close(fig)

    fig, ax = poke_dataframe(toplot,
                             'train_time',
                             'score',
                             labeler='model')
    fig.savefig('{}_lr0.5_model_traintime.png'.format(prefix))
    plt.close(fig)

    fig, ax = poke_dataframe(toplot,
                             'eval_time',
                             'score',
                             labeler='model')
    fig.savefig('{}_lr0.5_model_evaltime.png'.format(prefix))
    plt.close(fig)

    fig, ax = poke_dataframe(data,
                             'learning_rate',
                             'score',
                             labeler='model',
                             specific_label='784_10')
    fig.savefig('{}_model_784_10_totaltime.png'.format(prefix))
    plt.close(fig)
    return


def NP_NN(hidden_layers, learning_rate, epoch_number, batch_number):
    output_container = []
    # for hidden_layer, learning, epoch, batchsize in zip(hidden_layers,
    #                                                     learning_rate,
    #                                                     epoch_number,
    #                                                     batch_number):
    # p = Pool(6)
    todo = list(product(*[hidden_layers, learning_rate, epoch_number, batch_number]))
    for i in range(len(todo)):
        todo[i] = (i,) + todo[i]
    print(todo)
    # with Pool(processes=6) as p:
    #     p.starmap(execute_NP_NN, todo)

    for item in todo:
        [run_id, hidden_layer, learning, epoch, batch] = item
        print('Calculating:\nNN: {}\nLR: {}\nEN: {}\nBA: {}'.format(hidden_layer,
                                                                    learning,
                                                                    epoch,
                                                                    batch))
        print('Calculating {}'.format(hidden_layer))
        output_container += [execute_NP_NN(run_id, hidden_layer, learning, epoch, batch)]

    data = pd.DataFrame.from_records(
        output_container,
        columns=['model', 'learning_rate', 'score', 'total_time', 'train_time', 'eval_time',
                 'epoch_number', 'batch_number'])
    return data


def execute_NP_NN(run_id, hidden_layer, learning, epoch, batch):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
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
    output_container = (network_label,
                        learning,
                        score,
                        total_time,
                        train_time,
                        eval_time,
                        epoch,
                        batch)
    # CSV style for multiprocessing
    # f = open('{}.out'.format(run_id))
    # f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
    #     network_label, learning, score, total_time, train_time, eval_time, epoch, batch))
    # f.close()
    return output_container


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
        print('Calculating:\nNN: {}\nLR: {}\nEN: {}\nBA: {}'.format(hidden_layer,
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


def poke_dataframe(data, x, y, labeler=None, specific_label=None):
    fig, ax = plt.subplots()
    color_wheel = mcd.XKCD_COLORS
    color_list = list(mcd.XKCD_COLORS.keys())[::10]
    # toplot = data.loc[0.5]
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

