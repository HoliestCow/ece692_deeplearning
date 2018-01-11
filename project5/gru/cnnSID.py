
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

# import cPickle as pickle
import pickle

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class cnnMNIST(object):
    def __init__(self):
        self.lr = 1e-4
        self.epochs = 10000
        self.build_graph()

    def onehot_labels(self, labels):
        out = np.zeros((labels.shape[0], 6))
        for i in range(labels.shape[0]):
            out[i, :] = np.eye(6)[labels[i]]
        return out

    def onenothot_labels(self, labels):
        out = np.zeros((labels.shape[0],))
        for i in range(labels.shape[0]):
            out[i] = np.argmax(labels[i, :])
        return out

    def get_data(self):
        # data_norm = True
        # data_augmentation = False

        f = h5py.File('../data/sequential_dataset_relabel_justsources.h5', 'r')
        g = f['train']
        X = np.array(g['measured_spectra'])
        Y = self.onehot_labels(np.array(g['labels'], dtype=np.int32))

        g = f['test']
        X_test = np.array(g['measured_spectra'])
        Y_test = self.onehot_labels(np.array(g['labels'], dtype=np.int32))

        # img_prep = ImagePreprocessing()
        # if data_norm:
        #     img_prep.add_featurewise_zero_center()
        #     img_prep.add_featurewise_stdnorm()
        #
        # img_aug = ImageAugmentation()
        # if data_augmentation:
        #     img_aug.add_random_flip_leftright()
        #     img_aug.add_random_rotation(max_angle=30.)
        #     img_aug.add_random_crop((32, 32), 6)

        self.x_train = X
        self.y_train = Y

        self.x_test = X_test
        self.y_test = Y_test

        f.close()

        return

    def batch(self, iterable, n=1, shuffle=True):
        if shuffle:
            self.shuffle()
        # l = len(iterable)
        l = iterable.shape[0]
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l), :]
    
    def validation_batcher(self):
        f = h5py.File('../data/sequential_dataset_relabel_justsources.h5', 'r')
        g = f['validate']
        samplelist = list(g.keys())

        for i in range(len(samplelist)):
            data = g[samplelist[i]]
            yield data


    def build_graph(self):
        
        final_nodes = 4096

        self.x = tf.placeholder(tf.float32, shape=[None, 1024])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 6])

        x_image = self.hack_1dreshape(self.x)
        # define conv-layer variables
        W_conv1 = self.weight_variable([1, 5, 1, 32])    # first conv-layer has 32 kernels, size=5
        b_conv1 = self.bias_variable([32])
        W_conv2 = self.weight_variable([1, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        # x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # densely/fully connected layer
        W_fc1 = self.weight_variable([256 * 64, final_nodes])
        b_fc1 = self.bias_variable([final_nodes])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 256 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout regularization
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # linear classifier
        W_fc2 = self.weight_variable([final_nodes, 6])
        b_fc2 = self.bias_variable([6])

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)

    def shuffle(self):
        rng_state = np.random.get_state()
        np.random.set_state(rng_state)
        np.random.shuffle(self.x_train)
        np.random.set_state(rng_state)
        np.random.shuffle(self.y_train)
        return

    def train(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.eval() # creating evaluation
        for i in range(self.epochs):
            # batch = mnist.train.next_batch(50)
            x_generator = self.batch(self.x_train, n=128)
            y_generator = self.batch(self.y_train, n=128)
            # print(batch[0].shape)
            # print(batch[1].shape)
            if i % 100 == 0 and i != 0:
                train_acc = self.sess.run(self.accuracy,feed_dict={self.x: self.x_test[:1000, :],
                    self.y_: self.y_test[:1000, :],
                                                                   self.keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_acc))
            self.sess.run([self.train_step], feed_dict={self.x: next(x_generator),
                                                        self.y_: next(y_generator),
                                                        self.keep_prob: 0.5})
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
        tensor_expand = tf.expand_dims(x, 1)
        tensor_expand = tf.expand_dims(tensor_expand, -1)
        return tensor_expand

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

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

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    # interest = 'cnndetalt3_wdiffs_lr0.0001_ep1000'
    # interest = 'cnndetalt3_relabel_lr0.0001_ep500'
    # interest = 'cnndetalt3_relabel_lr1e-05_ep50000'
    # interest = 'cnndetalt3_relabel_lr1e-05_ep10000'
    interest = 'cnndetalt3_relabel_lr0.0001_ep10000_datasequential_dataset_relabel_allseconds.h5_sequential_dataset_relabel_allsecond'
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
    cnn.test_eval()

    predictions = cnn.get_label_predictions()

    predictions_decode = predictions
    labels_decode = cnn.onenothot_labels(cnn.y_test)

    np.save('sid_predictions.npy', predictions_decode)
    np.save('sid_ground_truth.npy', labels_decode)
    stop
    # counter = 0
    # hits = load_obj('normalgru_hits')
    hits = load_obj('{}_hits'.format(interest))
    answers = open('approach3_answers.csv', 'w')
    answers.write('RunID,SourceID,SourceTime,Comment\n')


    for sample in hits:
        key = sample
        data = hits[key]
        if data['time'] == 0:
            answers.write('{},{},{},\n'.format(key, 0, 0))
            continue
        x = np.array(data['spectra'])
        x = x.reshape((1, len(x)))
        predictions = cnn.sess.run(
            cnn.prediction,
            feed_dict = {cnn.x: x,
                         cnn.keep_prob: 1.0})
        t = np.array(data['time'])
        answers.write('{},{},{},\n'.format(
            key, predictions[0] + 1, t))
    answers.close()
    return

main()

