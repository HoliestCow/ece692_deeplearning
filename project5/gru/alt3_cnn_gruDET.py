
import tensorflow as tf
import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import itertools
# from copy import deepcopy

# import os
# import os.path
from collections import OrderedDict
import pickle
from itertools import islice

# NOTE: For python2
# import cPickle as pickle

# from tensorflow.examples.tutorials.mnist import input_data
#i mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class cnnMNIST(object):
    def __init__(self):
        self.use_gpu = False
        self.lr = 1e-4
        self.epochs = 1000
        self.runname = 'meh'
        self.build_graph()
        self.dataset_filename = './sequential_dataset_relabel.h5'

    def onehot_labels(self, labels):
        out = np.zeros((labels.shape[0], 2))
        for i in range(labels.shape[0]):
            out[i, :] = np.eye(2, dtype=int)[int(labels[i])]
        return out

    def onenothot_labels(self, labels):
        out = np.zeros((labels.shape[0],))
        for i in range(labels.shape[0]):
            out[i] = np.argmax(labels[i, :])
        return out

    def get_data(self):
        # data_norm = True
        # data_augmentation = False
        try:
            f = h5py.File(self.dataset_filename, 'r')
        except:
            # f = h5py.File('/home/holiestcow/Documents/2017_fall/ne697_hayward/lecture/datacompetition/sequential_dataset_balanced.h5', 'r')
            f = h5py.File('../data/{}'.format(self.dataset_filename), 'r')

        X = f['train']
        X_test = f['test']

        self.x_train = X
        self.x_test = X_test
        # NOTE: always use the keylist to get data
        self.data_keylist = list(X.keys())

        return

    def batch(self, iterable, n=1, shuffle=True, small_test=True, usethesekeys = None, shortset=False):
        if shuffle:
            self.shuffle()
        if usethesekeys is None:
            keylist = self.data_keylist
        else:
            keylist = usethesekeys
            if shortset:
                keylist = usethesekeys[:100]

        sequence_length = 15

        # l = len(iterable)
        for i in range(len(keylist)):
            self.current_key = keylist[i]
            x = np.array(iterable[keylist[i]]['measured_spectra'])
            y = np.array(iterable[keylist[i]]['labels'])
            mask = y >= 0.5
            y[mask] = 1

            index = np.arange(x.shape[0])

            index_generator = self.window(index, n=sequence_length)
            tostore_spectra = np.zeros((0, sequence_length, 1024))
            tostore_labels = []
            for index_list in index_generator:
                tostore_spectra = np.concatenate((tostore_spectra, x[index_list, :].reshape((1, sequence_length, 1024))))
                tostore_labels += [y[list(index_list)[-1]]]
            tostore_labels = np.array(tostore_labels)

            x = tostore_spectra
            y = self.onehot_labels(tostore_labels)
            self.current_batch_length = x.shape[0]

            yield x, y

    def memory_batch(self, iterable, n=1, shuffle=True, small_test=True, usethesekeys = None, shortset=False):
        if shuffle:
            self.shuffle()
        if usethesekeys is None:
            keylist = self.data_keylist
        else:
            keylist = usethesekeys
            if shortset:
                keylist = usethesekeys[:100]

        max_batch_size = 128

        sequence_length = 15

        # l = len(iterable)
        for i in range(len(keylist)):
            self.current_key = keylist[i]
            x = np.array(iterable[keylist[i]]['measured_spectra'])
            y = np.array(iterable[keylist[i]]['labels'])
            mask = y >= 0.5
            y[mask] = 1

            index = np.arange(x.shape[0])

            index_generator = self.window(index, n=sequence_length)
            tostore_spectra = np.zeros((0, sequence_length, 1024))
            tostore_labels = []
            for index_list in index_generator:
                tostore_spectra = np.concatenate((tostore_spectra, x[index_list, :].reshape((1, sequence_length, 1024))))
                tostore_labels += [y[list(index_list)[-1]]]
            tostore_labels = np.array(tostore_labels)

            self.howmanytimes = int(np.ceil(tostore_spectra.shape[0] / max_batch_size))
            # self.remainder = tostore_spectra.shape[0] % max_batch_size
            for j in range(self.howmanytimes + 1):
                start = j * max_batch_size
                end = ((j + 1) * max_batch_size)
                if end > tostore_spectra.shape[0]:
                    end = tostore_spectra.shape[0]
                x = tostore_spectra[start:end, :, :]
                if x.shape[0] == 0:
                    continue
                y = self.onehot_labels(tostore_labels[start:end])
                yield x, y

            # for j in range(self.current_batch_length):
            #     stuff = y[j,:]
            #     stuff = stuff.reshape((1, 2))
            #     yield x[j, :], stuff, z[j]

    def memory_validation_batcher(self):
        # f = h5py.File('./sequential_dataset_validation.h5', 'r')
        # NOTE: for using cnnfeatures sequential dataset
        # f = h5py.File('sequential_dataset_validation.h5', 'r')
        try:
            f = h5py.File(self.dataset_filename, 'r')
        except:
            f = h5py.File('../data/{}'.format(self.dataset_filename), 'r')
        g = f['validate']
        samplelist = list(g.keys())
        # samplelist = samplelist[:10]

        sequence_length = 15
        max_batch_size = 128

        for i in range(len(samplelist)):
            self.current_sample_name = samplelist[i]
            data = np.array(g[samplelist[i]])
            index = np.arange(data.shape[0])

            index_generator = self.window(index, n=sequence_length)
            tostore_spectra = np.zeros((0, sequence_length, 1024))
            for index_list in index_generator:
                tostore_spectra = np.concatenate((tostore_spectra, data[index_list, :].reshape((1, sequence_length, 1024))))
            # yield tostore_spectra, samplelist[i]

            self.howmanytimes = int(np.ceil(tostore_spectra.shape[0] / max_batch_size))

            for j in range(self.howmanytimes + 1):
                start = j * max_batch_size
                end = (j + 1) * max_batch_size
                if end > tostore_spectra.shape[0]:
                    end = tostore_spectra.shape[0]
                x = tostore_spectra[start:end, :, :]
                if x.shape[0] == 0:
                    continue
                y = self.onehot_labels(tostore_labels[start:end])
                yield x, y

    def window(self, seq, n=2):
        "Returns a sliding window (of width n) over data from the iterable"
        "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    def build_graph(self):
        # NOTE: CNN
        # self.x = tf.placeholder(tf.float32, shape=[None, 1024])
        # self.y_ = tf.placeholder(tf.float32, shape=[None, 2])
        self.x = tf.placeholder(tf.float32, shape=[None, 15, 1024])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2])
        # self.keep_prob = tf.placeholder(tf.float32, shape=[])
        # self.weights = tf.placeholder(tf.float32, shape=[30])

        feature_map1 = 32
        feature_map2 = 64

        num_units = 64
        num_layers = 2

        # x_image = self.hack_1dreshape(self.x)
        # print(x_image.shape)
        # define conv-layer variables
        W_conv1 = self.weight_variable([3, 9, 1, feature_map1])    # first conv-layer has 32 kernels, size=5
        b_conv1 = self.bias_variable([feature_map1])
        x_expanded = tf.expand_dims(self.x, 3)
        W_conv2 = self.weight_variable([3, 9, feature_map1, feature_map2])
        b_conv2 = self.bias_variable([feature_map2])

        # x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_expanded, W_conv1) + b_conv1)
        # h_pool1 = self.max_pool_2x2(h_conv1)
        h_pool1 = self.max_pool_spectra(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        # h_pool2 = self.max_pool_2x2(h_conv2)
        h_pool2 = self.max_pool_spectra(h_conv2)

        # h_fc1 = tf.contrib.layers.flatten(h_pool2)
        h_fc1 = tf.reshape(h_pool2, [-1, 15, 256 * feature_map2])
        # dropped_h_fc1 = tf.nn.dropout(h_fc1, self.keep_prob)

        cnn_output = h_fc1
        # cnn_output = dropped_h_fc1

        cells = []
        for _ in range(num_layers):
          cell = tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
          # cell = tf.contrib.rnn.DropoutWrapper(
          #     cell, output_keep_prob=1.0 - self.keep_prob)
          cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        # output, state = tf.nn.dynamic_rnn(cell, cnn_output, dtype=tf.float32)

        output, _ = tf.nn.dynamic_rnn(cell, cnn_output, dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        out_size = self.y_.get_shape()[1].value
        # logit = tf.contrib.layers.fully_connected(
        #     last, out_size, activation_fn=None)
        # self.y_conv = tf.nn.softmax(logit)
        # self.loss = tf.reduce_sum(tf.losses.softmax_cross_entropy(self.y_, self.y_conv))

        self.y_conv = tf.contrib.layers.fully_connected(last, out_size, activation_fn=None)

        classes_weights = tf.constant([1.0, 1.0])
        # classes_weights = tf.constant([0.1, 1.5])  # I haven't tried this one yet.
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=self.y_conv, targets=self.y_, pos_weight=classes_weights)
        self.loss = tf.reduce_mean(cross_entropy)

        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        # self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def shuffle(self):
        np.random.shuffle(self.data_keylist)
        return

    def train(self):
        if self.use_gpu:
            # use half of  the gpu memory
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.eval()  # creating evaluation
        a = time.time()
        for i in range(self.epochs):
            if i % 100 == 0 and i != 0:
                counter = 0
                sum_acc = 0
                sum_loss = 0
                hits = 0
                meh = 0
                x_generator_test = self.batch(self.x_test,
                                              usethesekeys=list(self.x_test.keys()), shortset=True)
                # first, second = next(x_generator_test)
                # for outerloop in range(self.howmanytimes):
                for first, second in x_generator_test:
                #     if outerloop != 0:
                        # first, second = next(x_generator_test)
                    # NOTE: quick and dirty preprocessing once again
                    # feedme = j / j.sum(axis=-1, keepdims=True)
                    feedme = j
                    accuracy, test_loss, prediction = self.sess.run([self.accuracy, self.loss, self.prediction], feed_dict={self.x: feedme,
                               self.y_: k})
                            #    self.weights: z})
                    sum_loss += np.sum(test_loss)
                    hits += np.sum(prediction)
                    sum_acc += accuracy
                    counter += feedme.shape[0]
                    meh += 1
                train_acc, train_loss = self.sess.run([self.accuracy, self.loss],
                                                      feed_dict={self.x: x, self.y_: y})
                b = time.time()
                print('step {}:\navg test acc {} | avg train acc {}\navg test loss {} | avg train loss {}\ntotalhits {}\ntime elapsed: {} s'.format(i, sum_acc / meh, train_acc, sum_loss / counter, train_loss, hits, b-a))
            # NOTE: QUick and dirty preprocessing. normalize to counts
            # x = x / x.sum(axis=-1, keepdims=True)
            x_generator = self.batch(self.x_train, shuffle=True)
            # x, y = next(x_generator)
            for x, y in x_generator:
            # for j in range(self.howmanytimes):
            #     if j != 0:
            #         x, y = next(x_generator)
                # print(self.current_key, x.shape)
                # for j in range(self.current_batch_length):
                    # x, y, z = next(x_generator)
                self.sess.run([self.train_step], feed_dict={
                               self.x: x,
                               self.y_: y})
                           #   self.weights: z})
            # self.shuffle()

    def eval(self):
        # self.time_index = np.arange(self.y_conv.get_shape()[0])
        self.prediction = tf.argmax(self.y_conv, 1)
        truth = tf.argmax(self.y_, 1)
        correct_prediction = tf.equal(self.prediction, truth)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # def test_eval(self):
    #     self.eval()
    #     x_generator = self.batch(self.x_test, n=100, shuffle=False)
    #     y_generator = self.batch(self.y_test, n=100, shuffle=False)
    #     test_acc = []
    #     counter = 0
    #     for data in x_generator:
    #         test_acc += [self.sess.run(self.accuracy, feed_dict={
    #         self.x: data, self.y_: next(y_generator), self.keep_prob: 1.0})]
    #     total_test_acc = sum(test_acc) / float(len(test_acc))
    #     print('test accuracy %g' % total_test_acc)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def hack_1dreshape(self, x):
        # expand its dimensionality to fit into conv2d
        tensor_expand = tf.expand_dims(x, -1)
        # tensor_expand = tf.expand_dims(tensor_expand, 1)
        return tensor_expand

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

    def max_pool_spectra(self, x):
        return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                              strides=[1, 1, 2, 1], padding='SAME')

    def get_label_predictions(self):
        x_batcher = self.batch(self.x_test, n=1000, shuffle=False,
                               usethesekeys=list(self.x_test.keys()))
        # y_batcher = self.batch(self.y_test, n=1000, shuffle=False)
        predictions = []
        correct_predictions = np.zeros((0, 2))
        counter = 0
        a = time.time()
        for x, y in x_batcher:
            counter += 1
            # x_features = x / x.sum(axis=-1, keepdims=True)
            if counter % 1000 == 0:
                print('label predictions done: {} in {} s'.format(counter, time.time() - a))
            x_features = x
            temp_predictions, score = self.sess.run(
            [self.prediction, self.y_conv],
            feed_dict={self.x: x_features})
            predictions += temp_predictions.tolist()
            correct_predictions = np.vstack((correct_predictions, y))
        return predictions, correct_predictions, score

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    cnn = cnnMNIST()
    validate_please = True
    characterize = True
    cnn.use_gpu = True
    cnn.lr = 1e-4
    cnn.epochs = 1
    cnn.runname = 'cnndetalt3_wdiffs_lr{}_ep{}'.format(cnn.lr, cnn.epochs)
    runname = cnn.runname
    a = time.time()
    print('Retrieving data')
    cnn.get_data()
    b = time.time()
    print('Built the data in {} s'.format(b-a))

    a = time.time()
    cnn.train()
    b = time.time()
    print('Training time: {} s'.format(b-a))
    # cnn.test_eval()

    if characterize:
        predictions, y, score = cnn.get_label_predictions()
        predictions_decode = predictions
        labels_decode = cnn.onenothot_labels(y)
        np.save('{}_predictions.npy'.format(runname), predictions_decode)
        np.save('{}_ground_truth.npy'.format(runname), labels_decode)
        print('Confusion matrix data saved')

    if validate_please:
        # Validation time
        # I have to rewrite this. Pickle is exceeding the swap space.
        # I could probably write the deck directly from here now that I think about it.
        # g = h5py.File('vanilla_dataset.h5', 'r')
        # h = g['validation']
        # spectra_list = list(h.keys())
        a = time.time()
        validation_data = cnn.validation_batcher()
        counter = 0
        answers = OrderedDict()
        for sample, samplename in validation_data:
            if counter % 100 == 0 and counter != 0:
                print('{} validation samples done in {} s'.format(counter, time.time() - a))
            x = sample
            print(x.shape)
            predictions = cnn.sess.run(
                cnn.prediction,
                feed_dict = {cnn.x: x})
            time_index = np.arange(predictions.shape[0])
            mask = predictions >= 0.5

            answers[samplename] = {'time': 0,
                                   'spectra': 0}

            if np.sum(mask) != 0:
                counts = np.squeeze(x[:, -1, :])
                counts = np.sum(counts, axis=1)
                t = time_index[mask]
                t = [int(i) for i in t]
                index_guess = np.argmax(counts[t])
                current_spectra = np.squeeze(x[index_guess, -1, :])
                current_time = t[index_guess] + 15
                answers[samplename] = {'time': current_time,
                                       'spectra': current_spectra}
            counter += 1
        save_obj(answers, '{}_hits'.format(runname))
        print('Validation written in {} s'.format(time.time() - a))

    return

main()

