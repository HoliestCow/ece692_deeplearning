# Sample code implementing LeNet-5 from Liu Liu

import tensorflow as tf
import numpy as np
from CIFAR10 import get_data, get_proper_images, onehot_labels, unpickle

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from skimage import data

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class cnnCIFAR10(object):
    def __init__(self, x, y, x_test, y_test, img_prep, img_aug):
        self.lr = 1e-3
        self.epochs = 100
        self.batch_size = 100
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.img_prep = img_prep
        self.img_aug = img_aug
        self.num_channels = 3
        self.pixel_width = int(np.sqrt(self.data.input_size / self.num_channels))
        # self.pixel_width = self.data.input_size
        self.test_batch, self.test_labels = self.data.get_test_data()
        self.build_graph()

    def build_graph(self):
        num_kernels_1 = 32
        num_kernels_2 = 64
        num_neurons_final = 1024

        self.x = tf.placeholder(tf.float32, shape=[None, self.data.input_size])

        self.y_ = tf.placeholder(tf.float32, shape=[None, self.data.output_size])
        # autoencoder stuff
        W_conv1 = self.weight_variable([5, 5, 3, 32]) # first conv-layer has 32 kernels, size=5
        b_conv1 = self.bias_variable([32])
        W_deconv1 = self.weight_variable([5, 5, 3, 32])
        b_deconv1 = self.bias_variable([3])

        h_conv1 = tf.nn.relu(self.conv2d(noise, W_conv1) + b_conv1)
        self.reconst1 = tf.nn.tanh(self.deconv2d(h_conv1, W_deconv1, [self.batch_size, 32, 32, 3]) + b_deconv1)

        self.ae_loss1 = tf.reduce_mean(tf.squared_difference(self.x, self.reconst1))
        self.ae_train_step1 = tf.train.AdamOptimizer(self.lr).minimize(self.ae_loss1)

        # define conv-layer variables
        # W_conv1 = self.weight_variable([5, 5, self.num_channels, num_kernels_1])    # first conv-layer has 32 kernels, size=5
        # W_conv1 = self.weight_variable([5, 5, 32, self.num_channels])
        # b_conv1 = self.bias_variable([num_kernels_1])
        W_conv2 = self.weight_variable([5, 5, num_kernels_1, num_kernels_2])
        b_conv2 = self.bias_variable([num_kernels_2])

        # print(self.x.shape)
        x_image = tf.reshape(self.x, [-1, self.pixel_width, self.pixel_width, self.num_channels])

        # h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
        # print(h_pool2.shape)

        # densely/fully connected layer
        W_fc1 = self.weight_variable([int(h_pool2.shape[1] * h_pool2.shape[1]) * num_kernels_2, num_neurons_final])
        b_fc1 = self.bias_variable([num_neurons_final])

        h_pool2_flat = tf.reshape(h_pool2, [-1, int(h_pool2.shape[1] * h_pool2.shape[1]) * num_kernels_2])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout regularization
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # linear classifier
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)

    def train(self):
        augseq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.CoarseDropout(p=0.1, size_percent=0.1)
        ])
        batch_loader = ia.BatchLoader(self.load_batch)
        bg_augmenter = ia.BackgroundAugmenter(batch_loader, augseq)

        # Run until load_batches() returns nothing anymore. This also allows infinite
        # training.
        while True:
            print("Next batch...")
            batch = bg_augmenter.get_batch()
            if batch is None:
                print("Finished epoch.")
                break
            images_aug = batch.images_aug

            # print("Image IDs: ", batch.data)
            # misc.imshow(np.hstack(list(images_aug)))


        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.eval()  # creating evaluation
        # batch = self.data.get_batch(self.batch_size)
        for i in range(self.epochs):
            # batch = mnist.train.next_batch(self.batch_size)
            batch = bg_augmenter.get_batch()
            if batch is None:
                continue
            images_aug = batch.images_aug

            if i % 100 == 0:
                train_acc = self.sess.run(self.accuracy, feed_dict={self.x: self.x_test, self.y_: self.y_test, self.keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_acc))
            self.sess.run([self.ae_train_step1, self.train_step], feed_dict={self.x: batch.images_aug, self.y_: batch.data, self.keep_prob: 0.5})
        batch_loader.terminate()
        bg_augmenter.terminate()

    # def get_batch(self):
    #     index = np.random.randomint(self.x.shape[0], size=self.batchsize)
    #     x = self.x[index, :]
    #     return x, y

# # Example augmentation sequence to run in the background.

# A generator that loads batches from the hard drive.
    def load_batch(self):
        # Here, load 10 batches of size 4 each.
        # You can also load an infinite amount of batches, if you don't train
        # in epochs.
        # batch_size = 4
        # nb_batches = 1

        # Here, for simplicity we just always use the same image.
        # astronaut = data.astronaut()
        images = self.x
        label = self.y
        # astronaut = ia.imresize_single_image(astronaut, (64, 64))

        # for i in range(nb_batches):
        # A list containing all images of the batch.
        # A list containing IDs of images in the batch. This is not necessary
        # for the background augmentation and here just used to showcase that
        # you can transfer additional information.

        # Add some images to the batch.
        # for b in range(self.batch_size):
        #     batch_images.append(astronaut)
        #     batch_data.append((i, b))
        index = np.random.randint(self.x.shape[0], (self.batch_size,))
        images = self.x[index, :, :, :]
        label = self.y[index]

            # Create the batch object to send to the background processes.
        batch = ia.Batch(
            images=np.array(images, dtype=np.uint8),
            data=label
        )

        return batch

# background augmentation consists of two components:
#  (1) BatchLoader, which runs in a Thread and calls repeatedly a user-defined
#      function (here: load_batches) to load batches (optionally with keypoints
#      and additional information) and sends them to a queue of batches.
#  (2) BackgroundAugmenter, which runs several background processes (on other
#      CPU cores). Each process takes batches from the queue defined by (1),
#      augments images/keypoints and sends them to another queue.
# The main process can then read augmented batches from the queue defined
# by (2).

    def eval(self):
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def test_eval(self):
        self.eval()
        # test_acc = self.sess.run(self.accuracy, feed_dict={
        #         self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0})
        print(self.test_batch.shape, self.test_labels.shape)
        test_acc = self.sess.run(self.accuracy, feed_dict={self.x: self.test_batch,
                                                           self.y_: self.test_labels,
                                                           self.keep_prob: 1.0})
        print('test accuracy %g' % test_acc)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def deconv2d(self, x, W):
        return tf.nn.conv2d_transpose(x, W, strides=[1, 1, 1, 1,], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')


def main():
    x, y, x_test, y_test, img_prep, img_aug = get_data()
    cnn = cnnCIFAR10(data)
    cnn
    cnn.train()
    cnn.test_eval()
    return

main()
