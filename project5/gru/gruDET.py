
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
from collections import OrderedDict
import pickle
# import cPickle as pickle

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class cnnMNIST(object):
    def __init__(self):
        self.lr = 1e-3
        self.epochs = 100
        self.build_graph()

    def onehot_labels(self, labels):
        out = np.zeros((labels.shape[0], 2))
        for i in range(labels.shape[0]):
            out[i, :] = np.eye(2)[int(labels[i])]
        return out

    def onenothot_labels(self, labels):
        out = np.zeros((labels.shape[0],))
        for i in range(labels.shape[0]):
            out[i] = np.argmax(labels[i, :])
        return out

    def get_data(self):
        # data_norm = True
        # data_augmentation = False

        f = h5py.File('./sequential_dataset.h5', 'r')

        X = f['train']
        X_test = f['test']

        self.x_train = X
        self.x_test = X_test
        # NOTE: always use the keylist to get data
        self.data_keylist = list(X.keys())

        return

    def batch(self, iterable, n=1, shuffle=True, small_test=True, usethesekeys = None):
        if shuffle:
            self.shuffle()
        if usethesekeys is None:
            keylist = self.data_keylist
        else:
            keylist = usethesekeys

       
        # l = len(iterable)
        for i in range(len(keylist)):
            x = np.array(iterable[keylist[i]]['measured_spectra'])
            y = np.array(iterable[keylist[i]]['labels'])
            mask = y >= 0.5
            z = np.ones((y.shape[0],))
            z[mask] = 10.0
            y = self.onehot_labels(y)
            yield x, y, z

    def validation_batcher(self):
        f = h5py.File('./sequential_dataset_validation.h5', 'r')
        # f = h5py.File('/home/holiestcow/Documents/2017_fall/ne697_hayward/lecture/datacompetition/sequential_dataset_validation.h5', 'r')
        samplelist = list(f.keys())
        # samplelist = samplelist[:10]

        for i in range(len(samplelist)):
            data = f[samplelist[i]]
            yield data


    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 15, 1024])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2])
        self.weights = tf.placeholder(tf.float32)

        num_units = 32
        num_layers = 1
        # dropout = tf.placeholder(tf.float32)

        cells = []
        for _ in range(num_layers):
            cell = tf.contrib.rnn.GRUCell(num_units)  # Or LSTMCell(num_units)
            # cell = tf.contrib.rnn.DropoutWrapper(
            #            cell, output_keep_prob=1.0)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        output, state = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        out_size = self.y_.get_shape()[1].value
        self.y_conv = tf.contrib.layers.fully_connected(
            last, out_size, activation_fn=None)
        # self.y_conv = tf.nn.softmax(logit) # probably a mistake here
        ratio = 1.0 / 1000000.0
        class_weight = tf.constant([ratio, 1.0 - ratio])
        weighted_logits = tf.multiply(self.y_conv, class_weight) # shape [batch_size, 2]
        self.loss = tf.nn.softmax_cross_entropy_with_logits(
                 logits=weighted_logits, labels=self.y_, name="xent_raw")
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        # loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.y_, logits=self.y_conv, pos_weight=self.weights))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def shuffle(self):
        np.random.shuffle(self.data_keylist)
        return

    def train(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.eval() # creating evaluation
        a = time.time()
        for i in range(self.epochs):
            # batch = mnist.train.next_batch(50)
            x_generator = self.batch(self.x_train, shuffle=True)

            if i % 10 == 0 and i != 0:
                counter = 0
                sum_acc = 0
                sum_loss = 0
                x_generator_test = self.batch(self.x_test,
                                              usethesekeys=list(self.x_test.keys()))
                for j, k, z in x_generator_test:
                    train_acc, train_loss = self.sess.run([self.accuracy, self.loss],feed_dict={self.x: j,
                                                                       self.y_: k})
                    sum_acc += np.sum(train_acc)
                    sum_loss += np.sum(train_loss)
                    counter += 1
                b = time.time()
                print('step {}:\navg testing loss {}\navg accuracy {}\ntime elapsed: {} s'.format(i, sum_acc / counter, sum_loss / counter, b-a))
            x, y, z = next(x_generator)
            self.sess.run([self.train_step], feed_dict={
                              self.x: x,
                              self.y_: y,
                              self.weights: 1000})
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
        tensor_expand = tf.expand_dims(x, 1)
        tensor_expand = tf.expand_dims(tensor_expand, -1)
        return tensor_expand

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

    def get_label_predictions(self):
        x_batcher = self.batch(self.x_test, n=1000, shuffle=False,
                               usethesekeys=list(self.x_test.keys()))
        # y_batcher = self.batch(self.y_test, n=1000, shuffle=False)
        predictions = []
        correct_predictions = np.zeros((0, 2))
        for x, y, z in x_batcher:
            temp_predictions = self.sess.run(
            self.prediction,
            feed_dict={self.x: x})
            predictions += temp_predictions.tolist()
            correct_predictions = np.vstack((correct_predictions, y))
        return predictions, correct_predictions


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    cnn = cnnMNIST()
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

    predictions, y = cnn.get_label_predictions()
     
    predictions_decode = predictions
    labels_decode = cnn.onenothot_labels(y)
    #
    np.save('grudet_predictions2.npy', predictions_decode)
    np.save('grudet_ground_truth2.npy', labels_decode)

    # Validation time
    validation_data = cnn.validation_batcher()
    answers = OrderedDict()
    for sample in validation_data:
        x = np.array(sample)
        predictions = cnn.sess.run(
            cnn.prediction,
            feed_dict = {cnn.x: x})
        time_index = np.arange(predictions.shape[0])
        mask = predictions >= 0.5

        runname = sample.name.split('/')[-1]
        if np.sum(mask) != 0:
            counts = np.sum(np.squeeze(x[:, -1, :]), axis=-1)
            t = time_index[mask]
            t = [int(i) for i in t]
            index_guess = np.argmax(counts[t])

            current_spectra = np.squeeze(x[t[index_guess], -1, :])
            current_time = t[index_guess] + 15 
            print(current_time)
            answers[runname] = {'time': current_time,
                                'spectra': current_spectra}
        else:
            answers[runname] = {'time': 0,
                                'spectra': 0}
    save_obj(answers, 'hits')

    return

main()

