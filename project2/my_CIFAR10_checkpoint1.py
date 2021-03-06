# Sample code implementing LeNet-5 from Liu Liu

import tensorflow as tf
import numpy as np
from CIFAR10 import CIFAR10
import time

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class cnnCIFAR10(object):
    def __init__(self, data):
        self.lr = 1E-3  # This achieves the speed I want
        self.epochs = 1000
        self.batch_size = 100
        self.data = data
        self.num_channels = 3
        self.pixel_width = int(np.sqrt(self.data.input_size / self.num_channels))
        # self.pixel_width = self.data.input_size
        # self.test_batch, self.test_labels = self.data.get_test_data()
        self.build_graph()

    def build_graph(self):
        # reverse order of 1 and 2
        num_kernels_1 = 32
        num_kernels_2 = 64
        num_neurons_final = 1024

        self.x = tf.placeholder(tf.float32, shape=[None, self.data.input_size])

        self.y_ = tf.placeholder(tf.float32, shape=[None, self.data.output_size])

        # define conv-layer variables
        W_conv1 = self.weight_variable([5, 5, self.num_channels, num_kernels_1])    # first conv-layer has 32 kernels, size=5
        # W_conv1 = self.weight_variable([5, 5, 32, self.num_channels])
        b_conv1 = self.bias_variable([num_kernels_1])
        W_conv2 = self.weight_variable([5, 5, num_kernels_1, num_kernels_2])
        b_conv2 = self.bias_variable([num_kernels_2])

        print(self.x.shape)
        x_image = tf.reshape(self.x, [-1, self.pixel_width, self.pixel_width, self.num_channels])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        print(h_pool2.shape)

        # densely/fully connected layer
        W_fc1 = self.weight_variable([int(h_pool2.shape[1] * h_pool2.shape[1]) * num_kernels_2, num_neurons_final])
        b_fc1 = self.bias_variable([num_neurons_final])

        h_pool2_flat = tf.reshape(h_pool2, [-1, int(h_pool2.shape[1] * h_pool2.shape[1]) * num_kernels_2])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout regularization
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # linear classifier
        W_fc2 = self.weight_variable([num_neurons_final, self.data.output_size])
        b_fc2 = self.bias_variable([self.data.output_size])

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)
        # constant_learning_rate  = 1E-4
        # self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy)


    def train(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.eval()  # creating evaluation
        batch = self.data.get_batch(self.batch_size, isHorizontalFlip=True)
        test_batch = self.data.get_test_data()
        learning_rate = self.lr
        for i in range(self.epochs):
            if i % 100 == 0:
                train_acc = []
                for j in range(self.data.num_test_epochs):
                    try:
                        test_x = next(test_batch)
                        test_y = next(test_batch)
                    except:
                        test_batch = self.data.get_test_data()
                        test_x = next(test_batch)
                        test_y = next(test_batch)
                    train_acc += [self.sess.run(self.accuracy, feed_dict={self.x: test_x, self.y_: test_y, self.keep_prob: 1.0})]
                print('step %d, training accuracy %g' % (i, sum(train_acc)/len(train_acc)))
                # half the learning rate
                # learning_rate /= 2
                # self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)
                # self.train_step = tf.train.GradientDescentOptimizer(self.lr, name='GradientDescent').minimize(self.cross_entropy)
            try:
                x = next(batch)
                y = next(batch)
            except:
                batch = self.data.get_batch(self.batch_size, isHorizontalFlip=True)
                x = next(batch)
                y = next(batch)
            self.sess.run([self.train_step], feed_dict={self.x: x, self.y_: y, self.keep_prob: 0.5})
            # self.sess.run(self.train_step, feed_dict={self.x: x, self.y_: y, self.keep_prob: 0.5})

    def eval(self):
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def test_eval(self):
        self.eval()
        # test_acc = self.sess.run(self.accuracy, feed_dict={
        #         self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0})
        # print(self.test_batch.shape, self.test_labels.shape)
        test_acc = []
        test_batch = self.data.get_test_data()
        for i in range(self.data.num_test_epochs):
            test_acc += [self.sess.run(self.accuracy, feed_dict={self.x: next(test_batch),
                                                           self.y_: next(test_batch),
                                                           self.keep_prob: 1.0})]
        print('test accuracy %g' % (sum(test_acc) / len(test_acc)))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        # initial = tf.constant(0.1, shape=shape)
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')


def main():
    data = CIFAR10()
    a = time.time()
    cnn = cnnCIFAR10(data)
    cnn
    cnn.train()
    b = time.time()
    training_time = b-a
    cnn.test_eval()
    c = time.time()
    timeperimage = (c-b) / data.num_test_samples 
    print('Training time: {}s per epoch'.format(training_time / cnn.epochs))
    print('Evaluation time per sample: {}s per sample'.format(timeperimage))  
    return

main()
