# Python 3.6
from network import Network
from time import time
# import numpy as np
import pandas as pd
# from itertools import product, tee, repeat
# from multiprocessing import Process, Pool
from multiprocessing import Pool
from itertools import product
from glob import glob


def main():
    # dataset
    run_name = cnn_test
    # FOR TF
    learning_rate = [1E-3, 0.1, 0.3]
    hidden_layers = [
        [],
        [30],
        [100, 30],
        [100, 300, 100]]
    epoch_number = [50, 100, 200]
    batch_number = [50, 100, 200]

    return

def NN(hidden_layers, data, learning_rate, epoch_number, batch_number):

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
    model = Network(dataset)
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

