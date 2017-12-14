
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
        self.lr = 1e-2
        self.epochs = 1000
        self.runname = 'grudetcnnalt3_{}'.format(self.epochs)
        self.build_graph()

    def onehot_labels(self, labels):
        out = np.zeros((labels.shape[0], 7))
        for i in range(labels.shape[0]):
            out[i, :] = np.eye(7, dtype=int)[int(labels[i])]
        return out

    def onenothot_labels(self, labels):
        out = np.zeros((labels.shape[0],))
        for i in range(labels.shape[0]):
            out[i] = np.argmax(labels[i, :])
        return out

    def get_data(self):
        # data_norm = True
        # data_augmentation = False

        # f = h5py.File('/home/holiestcow/Documents/2017_fall/ne697_hayward/lecture/datacompetition/sequential_dataset_balanced.h5', 'r')
        f = h5py.File('./sequential_dataset_balanced.h5', 'r')

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
                keylist = usethesekeys[:1000]


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
            z[mask] = 6.0
            y = self.onehot_labels(y)
            self.current_batch_length = x.shape[0]

            for j in range(self.current_batch_length):
                stuff = y[j,:]
                stuff = stuff.reshape((1, 7))
                yield x[j, :], stuff, z[j]
            # yield x, y, z

    def validation_batcher(self):
        # f = h5py.File('./sequential_dataset_validation.h5', 'r')
        # NOTE: for using cnnfeatures sequential dataset
        # f = h5py.File('sequential_dataset_validation.h5', 'r')
        # f = h5py.File('/home/holiestcow/Documents/2017_fall/ne697_hayward/lecture/datacompetition/sequential_dataset_balanced.h5', 'r')
        f = h5py.File('sequential_dataset_balanced.h5', 'r')
        g = f['validate']
        samplelist = list(g.keys())
        # samplelist = samplelist[:10]

        for i in range(len(samplelist)):
            self.current_sample_name = samplelist[i]
            data = np.array(g[samplelist[i]])
            self.current_batch_length = data.shape[0]
            for j in range(self.current_batch_length):
                current_x = np.squeeze(data[j, :, :])
                yield current_x

    def build_graph(self):
        # NOTE: CNN
        # self.x = tf.placeholder(tf.float32, shape=[None, 1024])
        # self.y_ = tf.placeholder(tf.float32, shape=[None, 7])
        self.x = tf.placeholder(tf.float32, shape=[15, 1024])
        self.y_ = tf.placeholder(tf.float32, shape=[1, 7])
        self.weights = tf.placeholder(tf.float32, shape=[])

        feature_map1 = 16
        feature_map2 = 32

        final_hidden_nodes = 256

        num_units = 16
        num_layers = 1

        x_image = self.hack_1dreshape(self.x)
        # define conv-layer variables
        W_conv1 = self.weight_variable([1, 3, 1, feature_map1])    # first conv-layer has 32 kernels, size=5
        b_conv1 = self.bias_variable([feature_map1])
        W_conv2 = self.weight_variable([1, 3, feature_map1, feature_map2])
        b_conv2 = self.bias_variable([feature_map2])

        # x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # densely/fully connected layer
        W_fc1 = self.weight_variable([256 * feature_map2, final_hidden_nodes])
        b_fc1 = self.bias_variable([final_hidden_nodes])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 256 * feature_map2])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # the features are now the h_fc1
        h_fc1 = tf.expand_dims(h_fc1, 0)

        # dropout regularization
        # self.keep_prob = tf.placeholder(tf.float32)
        # h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # linear classifier
        # W_fc2 = self.weight_variable([final_hidden_nodes, 7])

        lstm_in = tf.transpose(h_fc1, [1, 0, 2])
        lstm_in = tf.reshape(lstm_in, [-1, final_hidden_nodes])
        lstm_in = tf.layers.dense(lstm_in, num_units, activation=None)

        lstm_in = tf.split(lstm_in, 15, 0)

        lstm = tf.contrib.rnn.GRUCell(num_units)
        cell = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers)

        batch_size = tf.shape(h_fc1)[0]
        initial_state = cell.zero_state(batch_size, tf.float32)

        output, state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32, initial_state=initial_state)

        self.y_conv = tf.layers.dense(output[-1], 7, name='logits')
        # self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
        #     logits=self.y_conv, labels=self.y_))


        # self.y_conv = tf.nn.softmax(logit) # probably a mistake here
        # ratio = 1.0 / 1000000.0
        # ratio = 1.0 / ratio
        # class_weight = tf.constant([ratio,
        #                             1.0 - ratio,
        #                             1.0 - ratio,
        #                             1.0 - ratio,
        #                             1.0 - ratio,
        #                             1.0 - ratio,
        #                             1.0 - ratio])
        # weighted_logits = tf.multiply(self.y_conv, class_weight) # shape [batch_size, 2]
        # self.loss = tf.nn.softmax_cross_entropy_with_logits(
                 # logits=weighted_logits, labels=self.y_, name="xent_raw")
        # NOTE: Normal gru
        pure_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)
        # NOTE Normal gru with summing instead of mean
        # self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        # NOTE: Weighted gru
        # self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.y_, logits=self.y_conv, pos_weight=200.0))
        # NOTE: Weighted gru with summing instead of mean
        # self.loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(
        #     targets=self.y_,
        #     logits=self.y_conv,
        #     pos_weight=10000.0))
        # loss_per_example = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_conv, labels=self.y_)
        self.loss = tf.reduce_sum(tf.multiply(self.weights, pure_loss))
        # self.loss = tf.reduce_sum(loss_per_example)

        # self.loss = tf.reduce_sum(tf.losses.sparse_softmax_cross_entropy(labels=self.y_), logits=self.y_conv, weights=self.weights))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

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
            # batch = mnist.train.next_batch(50)
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
                    accuracy, train_loss, prediction = self.sess.run([self.accuracy, self.loss, self.prediction], feed_dict={self.x: feedme,
                               self.y_: k,
                               self.weights: z})
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
            for j in range(self.current_batch_length):
                # x, y, z = next(x_generator)
                self.sess.run([self.train_step], feed_dict={
                                  self.x: x,
                                  self.y_: y,
                                  self.weights: z})
                x, y, z = next(x_generator)
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
        correct_predictions = np.zeros((0, 7))
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

    # predictions, y, score = cnn.get_label_predictions()

    # predictions_decode = predictions
    # labels_decode = cnn.onenothot_labels(y)
    #
    # np.save('{}_predictions.npy'.format(runname), predictions_decode)
    # np.save('{}_ground_truth.npy's.format(runname), labels_decode)
    # Validation time
    # I have to rewrite this. Pickle is exceeding the swap space.
    # I could probably write the deck directly from here now that I think about it.
    a = time.time()
    validation_data = cnn.validation_batcher()
    answers = open('approach3_answers.csv', 'w')
    answers.write('RunID,SourceID,SourceTime,Comment\n')
    
    counter = 0
    for sample in validation_data:
        if counter % 100 == 0:
            print('{} samples processed in {} s'.format(counter, time.time() - a))
        counter += 1
        print('start ' + cnn.current_sample_name)
        sample_prediction_list = np.zeros((cnn.current_batch_length,))
        sample_score_list = np.zeros((cnn.current_batch_length, 7))
        for j in range(cnn.current_batch_length - 1):
            predictions, score = cnn.sess.run(
                [cnn.prediction, cnn.y_conv],
                feed_dict = {cnn.x: sample})
            sample_prediction_list[j] = predictions[0]
            sample_score_list[j, :] = score[0, :]
            sample = next(validation_data)
        time_index = np.arange(sample_prediction_list.shape[0])
        mask = sample_prediction_list >= 0.5

        # runname = sample.name.split('/')[-1]
        runname = cnn.current_sample_name
        print('end ' + runname)

        # Current spectra needs to be vanilla
        # current time needs to be zero

        if np.sum(mask) != 0:
            counts = np.argmax(sample_score_list, axis=1)
            score_list = []
            for j in counts:
                score_list += [sample_score_list[j, counts[j]]]
            score_list = np.array(score_list)
            t = time_index[mask]
            t = [int(i) for i in t]
            index_guess = np.argmax(score_list[t])
            # print(score_list)
            # print(predictions[mask])
            # print(index_guess)

            # current_spectra = np.squeeze(x[t[index_guess], -1, :])
            current_time = t[index_guess] + 15  # Not sure if its + 15?
            current_prediction = sample_prediction_list[mask][index_guess]

            answers.write('{},{},{},\n'.format(runname, current_time, current_prediction))
        else:
            answers.write('{},0,0,\n'.format(runname))
    answers.close()

    print('Validation written in {} s'.format(time.time() - a))

    return

main()

