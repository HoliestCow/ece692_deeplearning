
# Sample code implementing LeNet-5 from Liu Liu

import tensorflow as tf
import numpy as np
import time
import h5py
# import matplotlib.pyplot as plt
import pickle

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

        f = h5py.File('../gru/sequential_dataset_balanced.h5', 'r')

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
            mask = y >= 0.5
            z = np.ones((y.shape[0],))
            z[mask] = 100000.0
            y = self.onehot_labels(y)
            yield x, y, z

    def validation_batcher(self):
        f = h5py.File('../gru/sequential_dataset_validation.h5', 'r')
        # f = h5py.File('/home/holiestcow/Documents/2017_fall/ne697_hayward/lecture/datacompetition/sequential_dataset_validation.h5', 'r')
        samplelist = list(f.keys())
        # samplelist = samplelist[:10]

        for i in range(len(samplelist)):
            data = f[samplelist[i]]
            yield data

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 15, 1024])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2])

        x_image = self.hack_1dseqreshape(self.x)
        # define conv-layer variables
        W_conv1 = self.weight_variable([3, 5, 1, 32])    # first conv-layer has 32 kernels, size=5
        b_conv1 = self.bias_variable([32])
        W_conv2 = self.weight_variable([3, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        # x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)


        # densely/fully connected layer
        W_fc1 = self.weight_variable([256 * 64 * 4, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 256 * 64 * 4])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout regularization
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # linear classifier
        W_fc2 = self.weight_variable([1024, 2])
        b_fc2 = self.bias_variable([2])

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.train_step = tf.train.MomentumOptimizer(
            self.lr, 0.4, use_nesterov=True).minimize(self.loss)


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
            x_generator = self.batch(self.x_train, shuffle=True)
            x, y, z = next(x_generator)
            if i % 10 == 0 and i != 0:
                counter = 0
                sum_loss = 0
                sum_acc = 0
                x_generator_test = self.batch(self.x_test,
                                              usethesekeys=list(self.x_test.keys()), shortset=True)
                for j, k, z in x_generator_test:
                    train_loss, accuracy = self.sess.run(
                        [self.loss, self.accuracy],
                        feed_dict={self.x: j,
                                   self.y_: k,
                                   self.keep_prob: 1.0})
                    sum_loss += np.sum(train_loss)
                    sum_acc += np.sum(accuracy)
                    counter += 1
                b = time.time()
                print('step {}:\navg loss {}\navg acc {}. Time elapsed {} s\n'.format(i, sum_loss / counter, sum_acc / counter, b-a))
            self.sess.run([self.train_step], feed_dict={self.x: x,
                                                        self.y_: y,
                                                        self.keep_prob: 0.5})
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

    def hack_1dseqreshape(self, x):
        # NOTE: I'm considering the channel number as pixel location
        # Which means the only channel is "counts"
        tensor_expand = tf.expand_dims(x, -1)
        return tensor_expand

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def get_label_predictions(self):
        x_batcher = self.batch(self.x_test, n=1000, shuffle=False,
                               usethesekeys=list(self.x_test.keys()))
        # y_batcher = self.batch(self.y_test, n=1000, shuffle=False)
        predictions = []
        correct_predictions = np.zeros((0, 2))
        for x, y, z in x_batcher:
            temp_predictions = self.sess.run(
                self.prediction,
                feed_dict={self.x: x,
                           self.keep_prob: 1.0})
            predictions += temp_predictions.tolist()
            correct_predictions = np.vstack((correct_predictions, y))
        return predictions, correct_predictions


def save_obj(obj, name):
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

    predictions, y = cnn.get_label_predictions()

    # scores = np.zeros((score.shape[0],))
    # for i in range(len(scores)):
    #     scores[i] = score[i, int(predictions[i])]

    predictions_decode = predictions
    labels_decode = cnn.onenothot_labels(y)

    np.save('cnnseqsiddet_predictions.npy', predictions_decode)
    np.save('cnnseqsiddet_ground_truth.npy', labels_decode)

    stop

    validation_data = cnn.validation_batcher()
    answers = open('approach1_answers.csv', 'w')
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

