# Python 3.6
from tensorflow.examples.tutorials.mnist import input_data
from network import Network
from time import time
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd


def main():
    # dataset
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # building the networks
    layers = [
        # Here I'm messing with network structure
        [],
        [5],
        [10, 10],
        [5, 15, 5],
        [10, 5, 10],
        [4],
        [500, 250, 100],
        [500, 250, 50, 20],
        # Here I'm messing with learning rate
        [],
        [],
        [],
        [],
        []
        ]
    # learning rate settings
    learning_rate = [
        # Here I'm messing with network structure
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        # Here I'm messing with learning rate
        0.7,
        1.0,
        0.25,
        0.1,
        1E-3
        ]
    input_number = mnist.train.images.shape[1]
    output_number = mnist.train.labels.shape[1]
    output_container = []
    for i in range(len(layers)):
        network_label = network_label_generator(layers[i], input_number, output_number)
        [score, total_time, train_time, eval_time] = execute_neural_network(
            dataset=mnist,
            neural_network=layers[i],
            learning_rate=learning_rate[i])
        output_container += [(network_label,
                              learning_rate[i],
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
    # data.sort_index(level=['learning_rate'], ascending=[1], inplace=True)

    #### analysis portion ####
    toplot = data[data['learning_rate'] == 0.5]
    fig, ax = poke_dataframe(toplot,
                             'total_time',
                             'score',
                             labeler='model')
    fig.savefig('lr0.5_model_totaltime.png')
    plt.close(fig)

    fig, ax = poke_dataframe(toplot,
                             'train_time',
                             'score',
                             labeler='model')
    fig.savefig('lr0.5_model_traintime.png')
    plt.close(fig)

    fig, ax = poke_dataframe(toplot,
                             'eval_time',
                             'score',
                             labeler='model')
    fig.savefig('lr0.5_model_evaltime.png')
    plt.close(fig)

    fig, ax = poke_dataframe(data[data['model'] == '784_10'],
                             'learning_rate',
                             'score',
                             labeler='model',
                             specific_label='784_10')
    fig.savefig('model_784_10_totaltime.png')
    plt.close(fig)

    return


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


def network_label_generator(x, input_number, output_number):
    out = ''
    out += '{}_'.format(input_number)
    for i in range(len(x)):
        # out += str(x[i])+'_'
        out += '{}_'.format(x[i])
    out += '{}'.format(output_number)
    return out


def execute_neural_network(dataset=None, neural_network=[], learning_rate=0.5):
    total_tic = time()
    model = Network(dataset)
    model.define_layers(output_numbers=neural_network)
    model.build_graph(learning_rate=learning_rate)
    train_tic = time()
    model.train()
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

