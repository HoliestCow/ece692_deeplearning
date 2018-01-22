
# Sample code implementing LeNet-5 from Liu Liu

import tensorflow as tf
import numpy as np
import time
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from copy import deepcopy

import os
import os.path

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class cnnMNIST(object):
    def __init__(self):
        self.use_gpu = True
        self.lr = 1e-3
        self.epochs = 100
        self.runname = 'cnndetandsid_{}'.format(self.epochs)
        self.dataset_filename = 'sequential_dataset_relabel_allseconds.h5'
        self.build_graph()

    def onehot_labels(self, labels):
        out = np.zeros((labels.shape[0], 7))
        for i in range(labels.shape[0]):
            out[i, :] = np.eye(7)[labels[i]]
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

        training = f['train']
        testing = f['test']

        training_dataset = []
        training_labels = []
        for item in training:
            training_dataset += [np.array(training[item]['measured_spectra'])]
            training_labels += [np.array(training[item]['labels'])]
        training_dataset = np.concatenate(training_dataset, axis=0)
        training_labels = np.array(training_labels)
        training_labels = np.concatenate(training_labels, axis=0)

        testing_dataset = []
        testing_labels = []
        for item in testing:
            testing_dataset += [np.array(testing[item]['measured_spectra'])]
            testing_labels += [np.array(testing[item]['labels'])]
        testing_dataset = np.concatenate(testing_dataset, axis=0)
        testing_labels = np.concatenate(testing_labels, axis=0)

        self.x_train = training_dataset
        self.y_train = self.onehot_labels(training_labels)
        self.x_test = testing_dataset
        self.y_test = self.onehot_labels(testing_labels)
        return

    def naive_get_data(self):
        # data_norm = True
        # data_augmentation = False

        f = h5py.File('naive_dataset.h5', 'r')
        g = f['training']
        X = np.array(g['spectra'])
        Y = self.onehot_labels(np.array(g['labels'], dtype=np.int32))

        g = f['testing']
        X_test = np.array(g['spectra'])
        Y_test = self.onehot_labels(np.array(g['labels'], dtype=np.int32))

        self.x_train = X
        self.y_train = Y

        self.x_test = X_test
        self.y_test = Y_test

        f.close()

        return

    def batch(self, iterable, n=1):
        # l = len(iterable)
        l = iterable.shape[0]
        for ndx in range(0, l, n):
            data = iterable[ndx:min(ndx + n, l), :]
            # normalization = np.linalg.norm(data, 1, axis=1)
            # for j in range(data.shape[0]):
            #     data[j, :] = np.divide(data[j, :], normalization[j])
            yield data


    def validation_batcher(self, testing=False):
        if testing:
            f = h5py.File('../data/{}'.format(self.dataset_filename), 'r')
            g = f['test']
        else:
            f = h5py.File('../data/{}'.format('sequential_dataset_relabel_testset_validationonly.h5'), 'r')
            g = f['validate']
        samplelist = list(g.keys())

        for i in range(len(samplelist)):
            if testing:
                data = np.array(g[samplelist[i]]['measured_spectra'])
            else:
                data = np.array(g[samplelist[i]])
            yield data, samplelist[i]


    def build_graph(self):
        feature_map = 8

        self.x = tf.placeholder(tf.float32, shape=[None, 1024])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 7])
        self.keep_prob = tf.placeholder(tf.float32)

        x_image = self.hack_1dreshape(self.x)
        # define conv-layer variables
        W_conv1 = self.weight_variable([1, 3, 1, feature_map])    # first conv-layer has 32 kernels, size=5
        b_conv1 = self.bias_variable([feature_map])
        W_conv2 = self.weight_variable([1, 3, feature_map, feature_map])
        b_conv2 = self.bias_variable([feature_map])
        W_conv3 = self.weight_variable([1, 3, 1, feature_map])    # first conv-layer has 32 kernels, size=5
        b_conv3 = self.bias_variable([feature_map])
        W_conv4 = self.weight_variable([1, 3, feature_map, feature_map])
        b_conv4 = self.bias_variable([feature_map])
        W_conv5 = self.weight_variable([1, 3, feature_map, feature_map])
        b_conv5 = self.bias_variable([feature_map])
        W_conv6 = self.weight_variable([1, 3, feature_map, feature_map])
        b_conv6 = self.bias_variable([feature_map])
        W_conv7 = self.weight_variable([1, 3, feature_map, feature_map])
        b_conv7 = self.bias_variable([feature_map])
        W_conv8 = self.weight_variable([1, 3, feature_map, feature_map])
        b_conv8 = self.bias_variable([feature_map])
        W_conv9 = self.weight_variable([1, 3, feature_map, feature_map])
        b_conv9 = self.bias_variable([feature_map])
        W_conv10 = self.weight_variable([1, 3, feature_map, feature_map])
        b_conv10= self.bias_variable([feature_map])



        # x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_pool1_dropped = tf.nn.dropout(h_pool1, self.keep_prob)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1_dropped, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_pool2_dropped = tf.nn.dropout(h_pool2, self.keep_prob)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2_dropped, W_conv3) + b_conv3)
        h_pool3 = self.max_pool_2x2(h_conv3)
        h_pool3_dropped = tf.nn.dropout(h_pool3, self.keep_prob)
        h_conv4 = tf.nn.relu(self.conv2d(h_pool3_dropped, W_conv4) + b_conv4)
        h_pool4 = self.max_pool_2x2(h_conv4)
        h_pool4_dropped = tf.nn.dropout(h_pool4, self.keep_prob)
        h_conv5 = tf.nn.relu(self.conv2d(h_pool4_dropped, W_conv5) + b_conv5)
        h_pool5 = self.max_pool_2x2(h_conv5)
        h_pool5_dropped = tf.nn.dropout(h_pool5, self.keep_prob)
        h_conv6 = tf.nn.relu(self.conv2d(h_pool5_dropped, W_conv6) + b_conv6)
        h_pool6 = self.max_pool_2x2(h_conv6)
        h_pool6_dropped = tf.nn.dropout(h_pool6, self.keep_prob)
        h_conv7 = tf.nn.relu(self.conv2d(h_pool6_dropped, W_conv7) + b_conv7)
        h_pool7 = self.max_pool_2x2(h_conv7)
        h_pool7_dropped = tf.nn.dropout(h_pool7, self.keep_prob)
        h_conv8 = tf.nn.relu(self.conv2d(h_pool7_dropped, W_conv8) + b_conv8)
        h_pool8 = self.max_pool_2x2(h_conv8)
        h_pool8_dropped = tf.nn.dropout(h_pool8, self.keep_prob)
        h_conv9 = tf.nn.relu(self.conv2d(h_pool8_dropped, W_conv9) + b_conv9)
        h_pool9 = self.max_pool_2x2(h_conv9)
        h_pool9_dropped = tf.nn.dropout(h_pool9, self.keep_prob)
        h_conv10 = tf.nn.relu(self.conv2d(h_pool9_dropped, W_conv10) + b_conv10)
        h_pool10 = self.max_pool_2x2(h_conv10)
        # h_pool10_dropped = tf.nn.dropout(h_pool2, self.keep_prob)

        # densely/fully connected layer
        # W_fc1 = self.weight_variable([1 * feature_map, fc1])
        # b_fc1 = self.bias_variable([fc1])
        #
        # h_pool2_flat = tf.reshape(h_pool2, [-1, 256 * feature_map])
        # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout regularization
        # h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # linear classifier

        W_fc2 = self.weight_variable([feature_map, 7])
        b_fc2 = self.bias_variable([7])

        h_fc2 = tf.matmul(h_pool10, W_fc2) + b_fc2
        # h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

        # W_fc3 = self.weight_variable([fc2, 7])
        # b_fc3 = self.bias_variable([7])

        # y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
        self.y_conv = h_fc2

        # Now I have to weight to logits
        # class_weights = tf.constant([0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        # class_weights = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        # self.y_conv = tf.multiply(y_conv, class_weights)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        # reg = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1)
        # beta = 0.01
        # cross_entropy = tf.reduce_mean(cross_entropy + reg * beta)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)

    def shuffle(self):
        rng_state = np.random.get_state()
        np.random.set_state(rng_state)
        np.random.shuffle(self.x_train)
        np.random.set_state(rng_state)
        np.random.shuffle(self.y_train)
        # permutation = np.random.permutation(self.x_train.shape[0])
        # self.x_train = self.x_train[permutation, :]
        # self.y_train = self.y_train[permutation, :]
        return

    def train(self):
        if self.use_gpu:
            # use half of  the gpu memory
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.sess = tf.Session()
        # self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.eval() # creating evaluation
        a = time.time()
        for i in range(self.epochs):
            # batch = mnist.train.next_batch(50)
            x_generator = self.batch(self.x_train, n=128)
            y_generator = self.batch(self.y_train, n=128)
            # print(batch[0].shape)
            # print(batch[1].shape)
            if i % 10 == 0 and i != 0:
                test_acc = self.sess.run(self.accuracy,feed_dict={self.x: self.x_test[:200, :],
                    self.y_: self.y_test[:200, :],
                    self.keep_prob: 1.0})
                train_acc = self.sess.run(self.accuracy, feed_dict={self.x: current_x,
                                                                   self.y_: current_y,
                                                                   self.keep_prob: 1.0})
                print('step %d, training accuracy %g, testing accuracy %g, elapsed time %f' % (i, train_acc, test_acc, time.time()-a))
            for current_x in x_generator:
               current_y = next(y_generator)
               self.sess.run([self.train_step], feed_dict={self.x: current_x,
                                                           self.y_: current_y,
                                                           self.keep_prob: 0.01})

            self.shuffle()

    def eval(self):
        # self.time_index = np.arange(self.y_conv.get_shape()[0])
        self.prediction = tf.argmax(self.y_conv, 1)
        truth = tf.argmax(self.y_, 1)
        correct_prediction = tf.equal(self.prediction, truth)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def test_eval(self):
        self.eval()
        x_generator = self.batch(self.x_test, n=100)
        y_generator = self.batch(self.y_test, n=100)
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
        x_batcher = self.batch(self.x_test, n=256)
        predictions = np.zeros((0, 1))
        for data in x_batcher:
            temp_predictions = self.sess.run(
            self.prediction,
            feed_dict={self.x: data,
                       self.keep_prob: 1.0})
            temp_predictions = temp_predictions.reshape((temp_predictions.shape[0], 1))
            predictions = np.vstack((predictions, temp_predictions))
        return predictions

def label_datasets():

    # targetfile = '/home/holiestcow/Documents/zephyr/datasets/muse/trainingData/answers.csv'
    targetfile = '../data/answers.csv'
    head, tail = os.path.split(targetfile)

    # filename = []
    source_labels = {}

    id2string = {0: 'Background',
                 1: 'HEU',
                 2: 'WGPu',
                 3: 'I131',
                 4: 'Co60',
                 5: 'Tc99',
                 6: 'HEUandTc99'}


    f = open(targetfile, 'r')
    a = f.readlines()
    for i in range(len(a)):
        line = a[i].strip()
        if line[0] == 'R':
            continue
        parsed = line.split(',')
        filename = parsed[0]
        source = parsed[1]
        source_time = parsed[2]
        source_labels[filename] = {'source': id2string[int(source)],
                                   'time': float(source_time)}
    f.close()
    return source_labels

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def longest(l):
    print('length: ', len(l))
    if len(l) == 0:
        return None, None

    # if(not isinstance(l, list)): return(0)
    # return(max([len(l),] + [len(subl) for subl in l if isinstance(subl, list)] +
    #     [longest(subl) for subl in l]))
    max_index = -1
    max_length = 0

    counter = 0
    for item in l:
        current_index = counter
        current_length = len(item)
        if current_length > max_length:
            max_index = current_index
            max_length = current_length
        counter += 1

    return max_index, max_length

def main():

    isTest = True
    analyze_answers = True

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
    cnn.test_eval()

    predictions = cnn.get_label_predictions()

    predictions_decode = predictions
    labels_decode = cnn.onenothot_labels(cnn.y_test)

    np.save('{}_{}_predictions.npy'.format(cnn.runname, cnn.dataset_filename[:-4]), predictions_decode)
    np.save('{}_{}_ground_truth.npy'.format(cnn.runname, cnn.dataset_filename[:-4]), labels_decode)

    answers = open('approach1_answers_{}_{}.csv'.format(cnn.runname, cnn.dataset_filename[:-4]), 'w')
    answers.write('RunID,SourceID,SourceTime,Comment\n')
    counter = 0
    # if isTest:
    #     for sample, runname in testing_data:
    #         x = sample
    #         x = x[30:, :]
    #         predictions = cnn.sess.run(
    #             cnn.prediction,
    #             feed_dict = {cnn.x: x,
    #                          cnn.keep_prob: 1.0})
    #         time_index = np.arange(predictions.shape[0])
    #         mask = predictions >= 0.5
    #
    #         # runname = sample.name.split('/')[-1]
    #         # runname = sample.name
    #         if np.sum(mask) != 0:
    #             counts = np.sum(x, axis=1)
    #             # fig = plt.figure()
    #             t = time_index[mask]
    #             t = [int(i) for i in t]
    #             index_guess = np.argmax(counts[t])
    #
    #             current_predictions = predictions[mask]
    #
    #         if counter < 30 and np.sum(mask) != 0:
    #             fig = plt.figure()
    #             plt.plot(counts, 'b.')
    #             plt.plot(counts[mask], 'r.')
    #             plt.plot(counts[index_guess], 'g*')
    #             plt.plot(counts[int(label_dict[runname]['time']) - 30], 'm*')
    #             fig.savefig('hitcounts_{}.png'.format(counter))
    #         else:
    #             break
    #         counter += 1
    #     answers.close()
    #     return
    for sample, runname in validation_data:
        x = sample
        # x = x[30:, :]
        predictions = cnn.sess.run(
            cnn.prediction,
            feed_dict = {cnn.x: x,
                         cnn.keep_prob: 1.0})
        time_index = np.arange(predictions.shape[0])
        # mask = predictions >= 0.5
        machine = np.argwhere(predictions >= 0.5)
        hits = np.zeros((x.shape[0], ), dtype=bool)
        # hits = mask
        machine = machine.reshape((machine.shape[0], ))
        grouping = group_consecutives(machine)

        group_index, group_length = longest(grouping)
        print(group_index, group_length)
        if group_index is not None:
            hits[grouping[group_index]] = True
        # for group in grouping:
        #     if source_index in group:
        #         hits[group] = True
        #NOTE: I left off right here. I haven't figured out if there is no hits.ckligh

        # runname = sample.name.split('/')[-1]
        # runname = sample.name
        if np.sum(hits) != 0:
            counts = np.sum(x, axis=1)
            # fig = plt.figure()
            t = time_index[hits]
            t = [int(i) for i in t]
            index_guess = np.argmax(counts[t])

            current_predictions = predictions[hits]

            answers.write('{},{},{},\n'.format(
                runname, current_predictions[index_guess], t[index_guess]))
        else:
            answers.write('{},{},{},\n'.format(
                runname, 0, 0))

        if counter % 1000 == 0:
            print('{} validation samples complete'.format(counter))
        counter += 1
    answers.close()

    if analyze_answers:
        id2string = {0: 'Background',
                 1: 'HEU',
                 2: 'WGPu',
                 3: 'I131',
                 4: 'Co60',
                 5: 'Tc99',
                 6: 'HEUandTc99'}

        a = open('approach1_answers_{}_{}.csv'.format(cnn.runname, cnn.dataset_filename[:-4]), 'r')
        b = a.readlines()
        b = b[1:]
        predicted = {}
        for line in b:
            raw_line = line.strip()
            parsed = raw_line.split(',')
            print(parsed)
            name = parsed[0]
            predicted[name] = {'source': id2string[int(parsed[1])],
                               'time': float(parsed[2])}
        truth = label_datasets()

        delta = {}
        TP = 0
        FP = 0
        P = 0
        TN = 0
        FN = 0
        N = 0

        locale = 0
        locale_threshold = 10  # within 5 seconds on either side.
        counter = 0
        for item in predicted:
            locale = np.sqrt(np.power((predicted[item]['time'] - truth[item]['time']), 2))
            if predicted[item]['source'] == 0 and truth[item]['source'] == 1:
                FN += 1
                N += 1
            elif predicted[item]['source'] == 0 and truth[item]['source'] == 0:
                TN += 1
                N += 1
            elif predicted[item]['source'] != 0 and truth[item]['source'] != 0 and locale < locale_threshold:
                TP += 1
                P += 1
            else:
                FP += 1
                P += 1
            counter += 1

        print('TPR: {}\nFPR: {}\nTNR: {}\nFNR: {}\n'.format(
            float(TP) / float(P),
            float(FP) / float(P),
            float(TN) / float(N),
            float(FN) / float(N)))

    return

main()

