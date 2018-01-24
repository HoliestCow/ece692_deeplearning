
# Sample code implementing LeNet-5 from Liu Liu

import tensorflow as tf
import numpy as np
import time
import h5py
from collections import OrderedDict
from itertools import islice
# import matplotlib.pyplot as plt

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class cnnMNIST(object):
    def __init__(self):
        self.lr = 1e-3
        self.epochs = 100
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

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 16, 1024])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 7])

        x_image = tf.reshape(self.x, [-1, 16, 1024, 1])

        feature_map1 = 20
        feature_map2 = 40
        feature_map3 = 40
        feature_map4 = 60
        feature_map5 = 80

        self.keep_prob = tf.placeholder(tf.float32)

        # define conv-layer variables
        W_conv1 = self.weight_variable([3, 3, 1, feature_map1])    # first conv-layer has 32 kernels, size=5
        b_conv1 = self.bias_variable([feature_map1])
        W_conv2 = self.weight_variable([3, 3, feature_map1, feature_map2])
        b_conv2 = self.bias_variable([feature_map2])
        W_conv3 = self.weight_variable([3, 3, feature_map2, feature_map3])    # first conv-layer has 32 kernels, size=5
        b_conv3 = self.bias_variable([feature_map3])
        W_conv4 = self.weight_variable([3, 3, feature_map3, feature_map4])
        b_conv4 = self.bias_variable([feature_map4])
        W_conv5 = self.weight_variable([3, 3, feature_map4, feature_map5])
        b_conv5 = self.bias_variable([feature_map5])
        # W_conv6 = self.weight_variable([3, 3, feature_map5, feature_map6])
        # b_conv6 = self.bias_variable([feature_map6])
        # W_conv7 = self.weight_variable([3, 3, feature_map6, feature_map7])
        # b_conv7 = self.bias_variable([feature_map7])
        # W_conv8 = self.weight_variable([3, 3, feature_map7, feature_map8])
        # b_conv8 = self.bias_variable([feature_map8])
        # W_conv9 = self.weight_variable([3, 3, feature_map8, feature_map9])
        # b_conv9 = self.bias_variable([feature_map9])
        # W_conv10 = self.weight_variable([3, 3, feature_map9, feature_map10])
        # b_conv10= self.bias_variable([feature_map10])

        # x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool(h_conv1, [1, 2, 4, 1], [1, 2, 4, 1])
        h_pool1_dropped = tf.nn.dropout(h_pool1, self.keep_prob)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1_dropped, W_conv2) + b_conv2)
        h_pool2 = self.max_pool(h_conv2, [1, 2, 4, 1], [1, 2, 4, 1])
        h_pool2_dropped = tf.nn.dropout(h_pool2, self.keep_prob)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2_dropped, W_conv3) + b_conv3)
        h_pool3 = self.max_pool(h_conv3, [1, 2, 4, 1], [1, 2, 4, 1])
        h_pool3_dropped = tf.nn.dropout(h_pool3, self.keep_prob)
        h_conv4 = tf.nn.relu(self.conv2d(h_pool3_dropped, W_conv4) + b_conv4)
        h_pool4 = self.max_pool(h_conv4, [1, 2, 4, 1], [1, 2, 4, 1])
        h_pool4_dropped = tf.nn.dropout(h_pool4, self.keep_prob)
        h_conv5 = tf.nn.relu(self.conv2d(h_pool4_dropped, W_conv5) + b_conv5)
        h_pool5 = self.max_pool(h_conv5, [1, 1, 4, 1], [1, 1, 4, 1])
        h_pool5_dropped = tf.nn.dropout(h_pool5, self.keep_prob)
        h_pool5_flat = tf.reshape(h_pool5_dropped, [-1, feature_map5])

        # h_pool5_dropped = tf.nn.dropout(h_pool5, self.keep_prob)
        # h_conv6 = tf.nn.relu(self.conv2d(h_pool5_dropped, W_conv6) + b_conv6)
        # h_pool6 = self.max_pool_2x2(h_conv6)
        # h_pool6_dropped = tf.nn.dropout(h_pool6, self.keep_prob)
        # h_conv7 = tf.nn.relu(self.conv2d(h_pool6_dropped, W_conv7) + b_conv7)
        # h_pool7 = self.max_pool_2x2(h_conv7)
        # h_pool7_dropped = tf.nn.dropout(h_pool7, self.keep_prob)
        # h_conv8 = tf.nn.relu(self.conv2d(h_pool7_dropped, W_conv8) + b_conv8)
        # h_pool8 = self.max_pool_2x2(h_conv8)
        # h_pool8_dropped = tf.nn.dropout(h_pool8, self.keep_prob)
        # h_conv9 = tf.nn.relu(self.conv2d(h_pool8_dropped, W_conv9) + b_conv9)
        # h_pool9 = self.max_pool_2x2(h_conv9)
        # h_pool9_dropped = tf.nn.dropout(h_pool9, self.keep_prob)
        # h_conv10 = tf.nn.relu(self.conv2d(h_pool9_dropped, W_conv10) + b_conv10)
        # h_pool10 = self.max_pool_2x2(h_conv10)
        # h_pool10_flat = tf.reshape(h_pool10, [-1, feature_map10])

        # # h_pool10_dropped = tf.nn.dropout(h_pool2, self.keep_prob)

        # densely/fully connected layer
        # W_fc1 = self.weight_variable([1 * feature_map, fc1])
        # b_fc1 = self.bias_variable([fc1])
        #
        # h_pool2_flat = tf.reshape(h_pool2, [-1, 256 * feature_map])
        # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout regularization
        # h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # linear classifier
        W_fc2 = self.weight_variable([feature_map5, 7])
        b_fc2 = self.bias_variable([7])

        h_fc2 = tf.matmul(h_pool5_flat, W_fc2) + b_fc2
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
        np.random.shuffle(self.data_keylist)
        return

    def get_data(self):
        # data_norm = True
        # data_augmentation = False

        f = h5py.File('../data/sequential_dataset_relabel_allseconds.h5', 'r')

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

        sequence_length = 16
        max_batch_size = 64

        # l = len(iterable)
        for i in range(len(keylist)):
            self.current_key = keylist[i]
            x = np.array(iterable[keylist[i]]['measured_spectra'])
            y = np.array(iterable[keylist[i]]['labels'])
            # mask = y >= 0.5
            # y[mask] = 1

            index = np.arange(x.shape[0])

            index_generator = self.window(index, n=sequence_length)
            # tostore_spectra = np.zeros((0, sequence_length, 1024))
            tostore_spectra = []
            tostore_labels = []
            for index_list in index_generator:
                # tostore_spectra = np.concatenate((tostore_spectra, x[index_list, :].reshape((1, sequence_length, 1024))))
                tostore_spectra += [x[index_list, :].reshape((1, sequence_length, 1024))]
                # tostore_labels += [y[list(index_list)[-1]]]  # last label is the the correct label (current spectra in question)
                tostore_labels += [y[list(index_list)[int(sequence_length / 2)]]]  # middle frame in question
            tostore_spectra = np.concatenate(tostore_spectra, axis=0)
            tostore_labels = np.array(tostore_labels)

            self.howmanytimes = int(np.ceil(tostore_spectra.shape[0] / max_batch_size))

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

            # x = tostore_spectra
            # y = self.onehot_labels(tostore_labels)
            # self.current_batch_length = x.shape[0]

            # yield x, y

    def memory_validation_batcher(self):
        # f = h5py.File('./sequential_dataset_validation.h5', 'r')
        # NOTE: for using cnnfeatures sequential dataset
        # f = h5py.File('sequential_dataset_validation.h5', 'r')
        try:
            f = h5py.File(self.dataset_filename, 'r')
        except:
            f = h5py.File('../data/{}'.format('sequential_dataset_relabel_testdata_validationonly.h5'), 'r')
        g = f['validate']
        samplelist = list(g.keys())
        # samplelist = samplelist[:100]

        sequence_length = 16
        max_batch_size = 32

        for i in range(len(samplelist)):
            self.current_sample_name = samplelist[i]
            data = np.array(g[samplelist[i]])
            index = np.arange(data.shape[0])

            index_generator = self.window(index, n=sequence_length)
            # tostore_spectra = np.zeros((0, sequence_length, 1024))
            tostore_spectra = []
            for index_list in index_generator:
                # tostore_spectra = np.concatenate((tostore_spectra, data[index_list, :].reshape((1, sequence_length, 1024))))
                tostore_spectra += [data[index_list, :].reshape((1, sequence_length, 1024))]
            # yield tostore_spectra, samplelist[i]
            tostore_spectra = np.concatenate(tostore_spectra, axis=0)

            self.howmanytimes = int(np.ceil(tostore_spectra.shape[0] / max_batch_size))

            for j in range(self.howmanytimes + 1):
                start = j * max_batch_size
                end = (j + 1) * max_batch_size
                if end > tostore_spectra.shape[0]:
                    end = tostore_spectra.shape[0]
                x = tostore_spectra[start:end, :, :]
                if x.shape[0] == 0:
                    continue
                yield x

    def validation_batcher(self):
        f = h5py.File('../data/sequential_dataset_relabel_testset_validationonly.h5', 'r')
        # f = h5py.File('/home/holiestcow/Documents/2017_fall/ne697_hayward/lecture/datacompetition/sequential_dataset_validation.h5', 'r')
        samplelist = list(f.keys())
        # samplelist = samplelist[:10]

        for i in range(len(samplelist)):
            data = f[samplelist[i]]
            yield data

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

    def train(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.eval() # creating evaluation
        for i in range(self.epochs):
            # batch = mnist.train.next_batch(50)
            x_generator = self.batch(self.x_train, n=32)
            # y_generator = self.batch(self.y_train, n=32)
            # print(batch[0].shape)
            # print(batch[1].shape)
            if i % 10 == 0 and i != 0:
                train_acc = self.sess.run(self.accuracy,feed_dict={self.x: self.x_test[:50, :],
                    self.y_: self.y_test[:50, :],
                                                                   self.keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_acc))
            for x, y in x_generator:
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

    def max_pool(self, x, ksize, strides):
        # return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
        return tf.nn.max_pool(x, ksize=ksize,
                                strides=strides, padding='SAME')

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
    cnn.test_eval()

    predictions, score = cnn.get_label_predictions()

    scores = np.zeros((score.shape[0],))
    for i in range(len(scores)):
        scores[i] = score[i, int(predictions[i])]

    predictions_decode = predictions
    labels_decode = cnn.onenothot_labels(cnn.y_test)

    np.save('cnndetandidseq_predictions.npy', predictions_decode)
    np.save('cnndetandidseq_prediction_scores.npy', scores)
    np.save('cnndetandidseq_ground_truth.npy', labels_decode)

    validation_data = cnn.memory_validation_batcher()
    answers = open('approach3_answers.csv', 'w')
    answers.write('RunID,SourceID,SourceTime,Comment\n')
    # counter = 0
    for sample in validation_data:
        x = np.array(sample)
        predictions = cnn.sess.run(
            cnn.prediction,
            feed_dict = {cnn.x: x,
                         cnn.keep_prob: 1.0})
        time_index = np.arange(predictions.shape[0])
        mask = predictions >= 0.5

        runname = sample.name.split('/')[-1]
        if np.sum(mask) != 0:
            counts = np.squeeze(np.sum(x[:, -1, :], axis=2))
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

