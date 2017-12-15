
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
# import cPickle as pickle

# from tensorflow.examples.tutorials.mnist import input_data
#i mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class cnnMNIST(object):
    def __init__(self):
        self.lr = 1e-3
        self.epochs = 1000
        self.runname = 'grudetcnnalt3_{}'.format(self.epochs)
        self.build_graph()

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
            f = h5py.File('./sequential_dataset_balanced.h5', 'r')
        except:
            f = h5py.File('/home/holiestcow/Documents/2017_fall/ne697_hayward/lecture/datacompetition/sequential_dataset_balanced.h5', 'r')

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


        # l = len(iterable)
        for i in range(len(keylist)):
            self.current_key = keylist[i]
            x = np.array(iterable[keylist[i]]['measured_spectra'])
            y = np.array(iterable[keylist[i]]['labels'])
            # NOTE: For using cnnfeatures sequential dataset
            # x = np.array(iterable[keylist[i]]['features'])
            # y = np.array(iterable[keylist[i]]['labels'])
            mask = y >= 0.5
            y[mask] = 1
            z = np.ones((y.shape[0],))
            z[mask] = 5.0
            y = self.onehot_labels(y)
            self.current_batch_length = x.shape[0]

            yield x, y, z

            # for j in range(self.current_batch_length):
            #     stuff = y[j,:]
            #     stuff = stuff.reshape((1, 2))
            #     yield x[j, :], stuff, z[j]

    def validation_batcher(self):
        # f = h5py.File('./sequential_dataset_validation.h5', 'r')
        # NOTE: for using cnnfeatures sequential dataset
        # f = h5py.File('sequential_dataset_validation.h5', 'r')
        try:
            f = h5py.File('sequential_dataset_balanced.h5', 'r')
        except:
            f = h5py.File('/home/holiestcow/Documents/2017_fall/ne697_hayward/lecture/datacompetition/sequential_dataset_balanced.h5', 'r')
        g = f['validate']
        samplelist = list(g.keys())
        # samplelist = samplelist[:10]

        for i in range(len(samplelist)):
            self.current_sample_name = samplelist[i]
            data = np.array(g[samplelist[i]])
            self.current_batch_length = data.shape[0]
            yield data
            # for j in range(self.current_batch_length):
            #     current_x = np.squeeze(data[j, :, :])
            #     yield current_x

    def build_graph(self):
        # NOTE: CNN
        # self.x = tf.placeholder(tf.float32, shape=[None, 1024])
        # self.y_ = tf.placeholder(tf.float32, shape=[None, 2])
        self.x = tf.placeholder(tf.float32, shape=[None, 15, 1024])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2])
        # self.weights = tf.placeholder(tf.float32, shape=[30])

        feature_map1 = 32
        feature_map2 = 64

        num_units = 64
        num_layers = 2

        # x_image = self.hack_1dreshape(self.x)
        # print(x_image.shape)
        # define conv-layer variables
        W_conv1 = self.weight_variable([1, 9, 1, feature_map1])    # first conv-layer has 32 kernels, size=5
        b_conv1 = self.bias_variable([feature_map1])
        x_expanded = tf.expand_dims(self.x, 3)
        W_conv2 = self.weight_variable([1, 9, feature_map1, feature_map2])
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

        cnn_output = h_fc1

        # dropout = tf.placeholder(tf.float32)
        cells = []
        for _ in range(num_layers):
          cell = tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
        #   cell = tf.contrib.rnn.DropoutWrapper(
        #       cell, output_keep_prob=1.0 - dropout)
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

        # classes_weights = tf.constant([0.1, 0.6])
        # classes_weights = tf.constant([0.1, 1.0])  # works ok after 300 epochs
        classes_weights = tf.constant([0.1, 1.5])
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=self.y_conv, targets=self.y_, pos_weight=classes_weights)
        self.loss = tf.reduce_sum(cross_entropy)

        # self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def shuffle(self):
        np.random.shuffle(self.data_keylist)
        return

    def train(self):
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
                for j, k, z in x_generator_test:
                    # NOTE: quick and dirty preprocessing once again
                    # feedme = j / j.sum(axis=-1, keepdims=True)
                    feedme = j
                    accuracy, train_loss, prediction = self.sess.run([self.accuracy, self.loss, self.prediction], feed_dict={self.x: feedme,
                               self.y_: k})
                            #    self.weights: z})
                    sum_loss += np.sum(train_loss)
                    hits += np.sum(prediction)
                    sum_acc += accuracy
                    counter += feedme.shape[0]
                    meh += 1
                b = time.time()
                print('step {}:\navg acc {}\navg loss {}\ntotalhits {}\ntime elapsed: {} s'.format(i, sum_acc / meh, sum_loss / counter, hits, b-a))
            # NOTE: QUick and dirty preprocessing. normalize to counts
            # x = x / x.sum(axis=-1, keepdims=True)
            x_generator = self.batch(self.x_train, shuffle=True)
            x, y, z = next(x_generator)
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
        for x, y, z in x_batcher:
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

    predictions, y, score = cnn.get_label_predictions()

    predictions_decode = predictions
    labels_decode = cnn.onenothot_labels(y)
    #
    np.save('{}_predictions.npy'.format(runname), predictions_decode)
    np.save('{}_ground_truth.npy'.format(runname), labels_decode)
    # Validation time
    # I have to rewrite this. Pickle is exceeding the swap space.
    # I could probably write the deck directly from here now that I think about it.
    a = time.time()
    validation_data = cnn.validation_batcher()

    print('Validation written in {} s'.format(time.time() - a))

    return

main()

