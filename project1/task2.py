# Python 3.6
from tensorflow.examples.tutorials.mnist import input_data
from network import Network
from time import time
# import numpy as np
import pandas as pd


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
        0.25
        ]
    output_container = []
    for i in range(len(layers)):
        model_label = 'nn{}_lr{}'.format(layers[i], learning_rate[i])
        # model_label = ''
        # for j in range(len(layers[i])):
        #     model_label += '{}_'.format(layers[i][j])
        model_label = model_label[:-1]
        [score, total_time, train_time, eval_time] = execute_neural_network(
            dataset=mnist,
            neural_network=layers[i],
            learning_rate=learning_rate[i])
        output_container += [(model_label,
                              learning_rate[i],
                              score,
                              total_time,
                              train_time,
                              eval_time)]
        # this is [str, float, float, float, float]
    data = pd.DataFrame.from_records(
        output_container,
        columns=['model', 'learning_rate', 'score', 'total_time', 'train_time', 'eval_time'],
        index=['model', 'learning_rate'])

    print(data)
    import ipdb; ipdb.set_trace()
    # data.plot(kind='bar',x=data.loc[:, 0.5].index.values,y=data.loc[:, 0.5].column.values)

    return


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
    total_time = total_tic - total_toc
    train_time = train_tic - train_toc
    eval_time = eval_tic - eval_toc
    return score, total_time, train_time, eval_time

main()

