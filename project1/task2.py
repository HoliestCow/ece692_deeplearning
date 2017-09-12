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



def main():
    # dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    input_number = mnist.train.images.shape[1]
    output_number = mnist.train.labels.shape[1]

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    learning_rate = [1E-4, 1E-3, 0.1, 0.3, 0.5, 0.7, 1.0]
    hidden_layers = [
        [],
        [5],
        [10, 10],
        [50, 25, 10]]

    for row in range(len(hidden_layers)):
        hidden_layers[row] = [input_number] + hidden_layers[row] + [output_number]

    tf_data = TF_NN(hidden_layers, learning_rate, mnist)
    np_data = NP_NN(hidden_layers, learning_rate, training_data, test_data)
    #### analysis portion ####
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


def NP_NN(hidden_layers, learning_rate, training_data, test_data):
    output_container = []
    for i in range(len(hidden_layers)):
        print('Calculating {}'.format(hidden_layers[i]))
        for j in range(len(learning_rate)):
            network_label = network_label_generator(hidden_layers[i])
            # training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
            net = NP_Network(hidden_layers[i])
            bigtic = time()
            net.SGD(training_data, 10, 100, learning_rate[j], test_data=test_data)
            # def SGD(self, training_data, epochs, mini_batch_size, eta,
                    # test_data=None)
            train_time = net.train_time
            tic = time()
            score = net.evaluate(test_data)
            toc = time()
            eval_time = toc-tic
            bigtoc = time()
            total_time = bigtoc-bigtic
            output_container += [(network_label,
                                  learning_rate[j],
                                  score,
                                  total_time,
                                  train_time,
                                  eval_time)]
    data = pd.DataFrame.from_records(
        output_container,
        columns=['model', 'learning_rate', 'score', 'total_time', 'train_time', 'eval_time'])
    return data


def TF_NN(hidden_layers, learning_rate, data):

    output_container = []
    for i in range(len(hidden_layers)):
        for j in range(len(learning_rate)):
            network_label = network_label_generator(hidden_layers[i])
            [score, total_time, train_time, eval_time] = execute_neural_network(
                dataset=data,
                neural_network=hidden_layers[i],
                learning_rate=learning_rate[j])
            output_container += [(network_label,
                                  learning_rate[j],
                                  score,
                                  total_time,
                                  train_time,
                                  eval_time)]
        # this is [str, float, float, float, float]
    # data = pd.DataFrame.from_records(
    #     output_container,
    #     columns=['model', 'learning_rate', 'score', 'total_time', 'train_time', 'eval_time'],
    #     index=['learning_rate'])
    data = pd.DataFrame.from_records(
        output_container,
        columns=['model', 'learning_rate', 'score', 'total_time', 'train_time', 'eval_time'])

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

