import pandas as pd
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from itertools import repeat
import numpy as np


def main():
    # input_list = glob('*.p')
    # for stuff in input_list:
    #     data = pd.read_pickle(stuff)
    #     prefix = sub('.p', '', stuff)
    #     data[data['score'] >= 1.0] = (data['score'] / 10000)
    #     gambit(data, learning_rate=1E-3, prefix=prefix)
    tf_sigmoid = pd.read_pickle('TF_sigmoid.p')
    tf_relu = pd.read_pickle('TF_relu.p')
    tf_softmax = pd.read_pickle('TF_softmax.p')

    tf_sigmoid = tf_sigmoid.assign(activation_function=list(repeat('sigmoid', len(tf_sigmoid))))
    tf_relu = tf_relu.assign(activation_function=list(repeat('relu', len(tf_relu))))
    tf_softmax = tf_softmax.assign(activation_function=list(repeat('softmax', len(tf_softmax))))

    tf_sigmoid = tf_sigmoid.assign(approach=list(repeat('tensorflow', len(tf_sigmoid))))
    tf_relu = tf_relu.assign(approach=list(repeat('tensorflow', len(tf_relu))))
    tf_softmax = tf_softmax.assign(approach=list(repeat('tensorflow', len(tf_softmax))))

    tf = pd.concat([tf_softmax, tf_sigmoid, tf_relu])

    np_sigmoid = pd.read_pickle('NP_sigmoid.p')
    np_relu = pd.read_pickle('NP_relu.p')

    np_sigmoid = np_sigmoid.assign(activation_function=list(repeat('sigmoid', len(np_sigmoid))))
    np_relu = np_relu.assign(activation_function=list(repeat('relu', len(np_relu))))

    np_sigmoid = np_sigmoid.assign(approach=list(repeat('numpy', len(np_sigmoid))))
    np_relu = np_relu.assign(approach=list(repeat('numpy', len(np_relu))))

    np = pd.concat([np_sigmoid, np_relu])
    np['score'] = np['score'] / 10000

    allruns = pd.concat([tf, np])
    # allruns = tf
    gambit(allruns)

    return


def gambit(data):
    # changing structure
    architecture_score(data, epoch_number=100, batch_number=50, approach='tensorflow')
    architecture_traintime(data, epoch_number=100, batch_number=50, approach='tensorflow')
    architecture_evaltime(data, epoch_number=100, batch_number=50, approach='tensorflow')
    # Hyper params
    epoch_batch(data, model='784_30_10', learning_rate=1E-3, approach='tensorflow')
    learning_rate(data, model='784_30_10', batch_number=100, approach='tensorflow')

    # when the np data is done.
    architecture_score(data, epoch_number=100, batch_number=50, approach='numpy')
    architecture_traintime(data, epoch_number=100, batch_number=50, approach='numpy')
    architecture_evaltime(data, epoch_number=100, batch_number=50, approach='numpy')
    epoch_batch(data, model='784_30_10', learning_rate=1E-3, approach='numpy')
    learning_rate(data, model='784_30_10', batch_number=100, approach='numpy')
    return


def learning_rate(data, model='784_30_10', batch_number=100, activation_function='sigmoid',
                  approach='tensorflow'):
    a = data['model'] == model
    b = data['batch_number'] == batch_number
    c = data['approach'] == approach
    # d = data['activation_function'] == activation_function
    toplot = data[a & b & c]
    print(toplot)
    epochs = np.unique(toplot['epoch_number'])
    fig, ax = plt.subplots()
    for i in range(len(epochs)):
        selected = toplot[toplot['epoch_number'] == epochs[i]]
        ax.scatter(selected['learning_rate'], selected['score'],
                   label='epochs {}'.format(epochs[i]))
    ax.set_title('{}_{}_bn{}_af{}.png'.format(approach, model, batch_number, activation_function))
    ax.set_xlabel('learning_rate')
    ax.set_ylabel('score')
    ax.legend()
    fig.savefig('{}_learningrate.png'.format(approach))
    return


def epoch_batch(data, model='784_30_10', activation_function='sigmoid', learning_rate=1E-3,
                approach='tensorflow'):
    a = data['model'] == model
    b = data['approach'] == approach
    c = data['activation_function'] == activation_function
    d = data['learning_rate'] == learning_rate
    toplot = data[a & b & c & d]
    print(toplot)
    batchsize = np.unique(toplot['batch_number'])
    fig, ax = plt.subplots()
    for i in range(len(batchsize)):
        selected = toplot[toplot['batch_number'] == batchsize[i]]
        ax.scatter(selected['epoch_number'], selected['train_time'],
                   label='batchnumber {}'.format(batchsize[i]))
    ax.set_title('model_{}'.format(model))
    ax.set_xlabel('number of epochs')
    ax.set_ylabel('training time (s)')
    plt.legend()
    fig.savefig('{}_epochvstraintime_{}.png'.format(approach, model))

    fig, ax = plt.subplots()
    for i in range(len(batchsize)):
        selected = toplot[toplot['batch_number'] == batchsize[i]]
        ax.scatter(selected['epoch_number'], selected['eval_time'],
                   label='batchnumber {}'.format(batchsize[i]))
    ax.set_title('model_{}'.format(model))
    ax.set_xlabel('number of epochs')
    ax.set_ylabel('evaluation time (s)')
    plt.legend()
    fig.savefig('{}_epochvsevaltime_{}.png'.format(approach, model))

    fig, ax = plt.subplots()
    for i in range(len(batchsize)):
        selected = toplot[toplot['batch_number'] == batchsize[i]]
        ax.scatter(selected['epoch_number'], selected['score'],
                   label='batchnumber {}'.format(batchsize[i]))
    ax.set_title('model_{}'.format(model))
    ax.set_xlabel('number of epochs')
    ax.set_ylabel('score')
    plt.legend()
    fig.savefig('{}_epochvsscore_{}.png'.format(approach, model))
    return


def architecture_traintime(data, learning_rate=1E-3, epoch_number=100, batch_number=100,
                           approach='tensorflow'):
    # color_wheel = mcd.XKCD_COLORS
    # color_list = list(mcd.XKCD_COLORS.keys())[::5]
    # model, learning_rate, score, total_time, train_time, eval_time, epoch_number, batch_number
    ##################################################
    # First plot studies the structure of architecture
    ##################################################
    a = data['learning_rate'] == learning_rate
    b = data['epoch_number'] == epoch_number
    c = data['batch_number'] == batch_number
    d = data['approach'] == approach
    toplot = data[a & b & c & d]
    print(toplot)
    labels = np.unique(toplot['activation_function'])
    xtick_labels = np.unique(toplot['model'])
    ind = np.arange(len(xtick_labels))
    width = 1 / (len(xtick_labels) + 1.0)
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        selected = toplot[toplot['activation_function'] == labels[i]]
        ax.bar(ind + (i * width), selected['train_time'], width, label=labels[i])
    ax.set_ylabel('Training Time (s)')
    ax.set_xticks(ind + width)
    ax.set_xticklabels((selected['model']), rotation=10)
    ax.legend()

    ax.set_title('lr{}_en{}_bn{}_{}'.format(learning_rate, epoch_number, batch_number, approach))

    fig.savefig('{}_architecture_traintime.png'.format(approach))
    return


def architecture_evaltime(data, learning_rate=1E-3, epoch_number=100, batch_number=100,
                          approach='tensorflow'):
    # color_wheel = mcd.XKCD_COLORS
    # color_list = list(mcd.XKCD_COLORS.keys())[::5]
    # model, learning_rate, score, total_time, train_time, eval_time, epoch_number, batch_number
    ##################################################
    # First plot studies the structure of architecture
    ##################################################
    a = data['learning_rate'] == learning_rate
    b = data['epoch_number'] == epoch_number
    c = data['batch_number'] == batch_number
    d = data['approach'] == approach
    toplot = data[a & b & c & d]
    labels = np.unique(toplot['activation_function'])
    xtick_labels = np.unique(toplot['model'])
    width = 1 / (len(xtick_labels) + 1.0)
    ind = np.arange(len(xtick_labels))
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        selected = toplot[toplot['activation_function'] == labels[i]]
        ax.bar(ind + (i* width), selected['eval_time'], width, label=labels[i])
    ax.set_ylabel('Evaluation Time (s)')
    ax.set_xticks(ind + width)
    ax.set_xticklabels((selected['model']), rotation=10)
    ax.legend()
    ax.set_title('lr{}_en{}_bn{}_{}'.format(learning_rate, epoch_number, batch_number, approach))
    fig.savefig('{}_architecture_evaltime.png'.format(approach))
    return


def architecture_score(data, learning_rate=1E-3, epoch_number=100, batch_number=100,
                       approach='tensorflow'):
    a = data['learning_rate'] == learning_rate
    b = data['epoch_number'] == epoch_number
    c = data['batch_number'] == batch_number
    d = data['approach'] == approach
    toplot = data[a & b & c & d]
    labels = np.unique(toplot['activation_function'])
    xtick_labels = np.unique(toplot['model'])
    ind = np.arange(len(xtick_labels))
    width = 1 / (len(xtick_labels) + 1.0)
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        selected = toplot[toplot['activation_function'] == labels[i]]
        print(selected)
        print(ind + (i * width))
        print(selected['score'])
        ax.bar(ind + (i * width), selected['score'], width, label=labels[i])
    ax.set_ylabel('Score')
    ax.set_xticks(ind + width)
    ax.set_xticklabels((selected['model']), rotation=10)
    ax.legend()

    ax.set_title('lr{}_en{}_bn{}_{}'.format(learning_rate, epoch_number, batch_number, approach))

    fig.savefig('{}_architecture_score.png'.format(approach))

    return

main()
