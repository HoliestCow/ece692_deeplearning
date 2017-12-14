
# Sample code implementing LeNet-5 from Liu Liu

import tensorflow as tf
import numpy as np
import time
import h5py
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from copy import deepcopy

import os
import os.path

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class cnnMNIST(object):
    def __init__(self):
        self.lr = 1e-3
        self.epochs = 100
        self.runname = 'cnnseqdetandsid_{}'.format(self.epochs)
        self.build_graph()

    def onehot_labels(self, labels):
        out = np.zeros((len(labels), 7))
        for i in range(len(labels)):
            out[i, :] = np.eye(7)[int(labels[i])]
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
            x = np.array(iterable[keylist[i]]['measured_spectra'])
            y = np.array(iterable[keylist[i]]['labels'])
            # NOTE: For using cnnfeatures sequential dataset
            # x = np.array(iterable[keylist[i]]['features'])
            # y = np.array(iterable[keylist[i]]['labels'])
            mask = y >= 0.5
            # y[mask] = 1
            z = np.ones((y.shape[0],))
            # z[mask] = 6.0
            y = self.onehot_labels(y)
            self.current_batch_length = x.shape[0]
            yield x, y, z

            # for j in range(self.current_batch_length):
            #     stuff = y[j,:]
            #     stuff = stuff.reshape((1, 7))
            #     yield x[j, :], stuff, z[j]

    def validation_batcher(self):
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

        feature_map1 = 32
        feature_map2 = 64

        final_hidden_nodes = 1024

        self.x = tf.placeholder(tf.float32, shape=[None, 15, 1024])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 7])

        x_image = self.hack_1dreshape(self.x)
        # define conv-layer variables
        # Try 17, 3??
        print(x_image)
        W_conv1 = self.weight_variable([1, 9, 1, feature_map2])
        b_conv1 = self.bias_variable([feature_map2])
        W_conv2 = self.weight_variable([3, 1, feature_map2, feature_map1])
        b_conv2 = self.bias_variable([feature_map1])

        # x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        # h_pool1 = self.max_pool_2x2(h_conv1)
        h_pool1 = self.max_pool_spectra(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        # h_pool2 = self.max_pool_2x2(h_conv2)
        h_pool2 = self.max_pool_time(h_conv2)

        # W_conv3 = self.weight_variable([1, 3, feature_map2, feature_map3])
        # b_conv3 = self.bias_variable([feature_map3])
        # W_conv4 = self.weight_variable([1, 3, feature_map3, feature_map4])
        # b_conv4 = self.bias_variable([feature_map4])

        # h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)
        # h_pool3 = self.max_pool_2x2(h_conv3)
        # h_conv4 = tf.nn.relu(self.conv2d(h_pool3, W_conv4) + b_conv4)
        # h_pool4 = self.max_pool_2x2(h_conv4)

        # densely/fully connected layer
        print(h_pool2.shape)
        sizing_variable = 512 * 8
        W_fc1 = self.weight_variable([sizing_variable * feature_map1, final_hidden_nodes])
        b_fc1 = self.bias_variable([final_hidden_nodes])

        h_pool2_flat = tf.reshape(h_pool2, [-1, sizing_variable * feature_map1])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # W_fc1 = self.weight_variable([64 * feature_map4, final_hidden_nodes])
        # b_fc1 = self.bias_variable([final_hidden_nodes])

        # h_pool4_flat = tf.reshape(h_pool4, [-1, 64 * feature_map4])
        # h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

        # dropout regularization
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # linear classifier
        W_fc2 = self.weight_variable([final_hidden_nodes, 7])
        b_fc2 = self.bias_variable([7])

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Now I have to weight to logits
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def shuffle(self):
        # rng_state = np.random.get_state()
        # np.random.set_state(rng_state)
        np.random.shuffle(self.data_keylist)
        # np.random.set_state(rng_state)
        # np.random.shuffle(self.y_train)
        return

    def train(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.eval() # creating evaluation
        a = time.time()
        for i in range(self.epochs):
            if i % 10 == 0 and i != 0:
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
                    accuracy, train_loss, prediction = self.sess.run(
                        [self.accuracy, self.loss, self.y_conv],
                        feed_dict={self.x: feedme,
                            self.y_: k,
                            self.keep_prob: 1.0})
                            #    self.weights: z})
                    sum_loss += np.sum(train_loss)
                    hits += np.sum(prediction)
                    sum_acc += accuracy
                    counter += feedme.shape[0]
                    meh += 1
                b = time.time()
                print('step {}:\navg acc {}\navg loss {}\ntotalhits {}\ntime elapsed: {} s'.format(i, sum_acc / meh, sum_loss / counter, hits, b-a))
            x_generator = self.batch(self.x_train, n=128)
            current_x, current_y, current_z = next(x_generator)
            self.sess.run([self.train_step], feed_dict={self.x: current_x,
                                                        self.y_: current_y,
                                                        self.keep_prob: 0.50})
            # self.shuffle()

    def eval(self):
        # self.time_index = np.arange(self.y_conv.get_shape()[0])
        self.prediction = tf.argmax(self.y_conv, 1)
        truth = tf.argmax(self.y_, 1)
        correct_prediction = tf.equal(self.prediction, truth)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def test_eval(self):
        self.eval()
        x_generator = self.batch(self.x_test, n=100, shuffle=False)
        y_generator = self.batch(self.y_test, n=100, shuffle=False)
        test_acc = []
        counter = 0
        for data in x_generator:
            test_acc += [self.sess.run(self.accuracy, feed_dict={
            self.x: data, self.y_: next(y_generator), self.keep_prob: 1.0})]
        total_test_acc = sum(test_acc) / float(len(test_acc))
        print('test accuracy %g' % total_test_acc)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def hack_1dreshape(self, x):
        # expand its dimensionality to fit into conv2d
        # tensor_expand = tf.expand_dims(x, 1)
        tensor_expand = tf.expand_dims(x, -1)
        return tensor_expand

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # def max_pool_2x2(self, x):
    #     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
    #                             strides=[1, 2, 2, 1], padding='SAME')
    def max_pool_spectra(self, x):
        return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                              strides=[1, 1, 2, 1], padding='SAME')

    def max_pool_time(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                              strides=[1, 2, 1, 1], padding='SAME')

    def get_label_predictions(self):
        x_batcher = self.batch(self.x_test, n=1000, shuffle=False)
        # y_batcher = self.batch(self.y_test, n=1000, shuffle=False)
        predictions = np.zeros((0, 1))
        for data in x_batcher:
            temp_predictions = self.sess.run(
            self.prediction,
            feed_dict={self.x: data,
                       self.keep_prob: 1.0})
            temp_predictions = temp_predictions.reshape((temp_predictions.shape[0], 1))
            predictions = np.vstack((predictions, temp_predictions))
        return predictions

def main():
    cnn = cnnMNIST()
    a = time.time()
    print('Retrieving data')
    cnn.get_data()
    b = time.time()
    print('Built the data in {} s'.format(b-a))

    validation_data = cnn.validation_batcher()

    a = time.time()
    cnn.train()
    b = time.time()
    print('Training time: {} s'.format(b-a))
    # cnn.test_eval()

    predictions = cnn.get_label_predictions()

    predictions_decode = predictions
    labels_decode = cnn.onenothot_labels(cnn.y_test)

    np.save('{}_predictions.npy'.format(cnn.runname), predictions_decode)
    np.save('{}_ground_truth.npy'.format(cnn.runname), labels_decode)

    answers = open('approach1c_answers.csv', 'w')
    answers.write('RunID,SourceID,SourceTime,Comment\n')
    # counter = 0
    for sample in validation_data:
        x = np.array(sample['spectra'])
        x = x[30:, :]
        predictions = cnn.sess.run(
            cnn.prediction,
            feed_dict = {cnn.x: x,
                         cnn.keep_prob: 1.0})
        time_index = np.arange(predictions.shape[0])
        mask = predictions >= 0.5

        runname = sample.name.split('/')[-1]
        if np.sum(mask) != 0:
            counts = np.sum(x, axis=1)
            # fig = plt.figure()
            t = time_index[mask]
            t = [int(i) for i in t]
            index_guess = np.argmax(counts[t])

            current_predictions = predictions[mask]

            answers.write('{},{},{},\n'.format(
                runname, current_predictions[index_guess], t[index_guess] + 30))
        else:
            answers.write('{},{},{},\n'.format(
                runname, 0, 0))
    answers.close()
    return

main()

