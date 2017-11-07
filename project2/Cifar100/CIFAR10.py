import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.models.dnn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.optimizers import SGD
import numpy as np
import tensorflow as tf
import urllib

def get_proper_images(raw):
    raw_float = np.array(raw, dtype=float)
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])
    return images

def onehot_labels(labels):
    return np.eye(10)[labels]

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def get_data():
    data_norm = True
    data_augmentation = True

    data1  = unpickle('../cifar-10-batches-py/data_batch_1')
    data2  = unpickle('../cifar-10-batches-py/data_batch_2')
    data3  = unpickle('../cifar-10-batches-py/data_batch_3')
    data4  = unpickle('../cifar-10-batches-py/data_batch_4')
    data5  = unpickle('../cifar-10-batches-py/data_batch_5')

    X = np.concatenate((get_proper_images(data1['data']),
                        get_proper_images(data2['data']),
                        get_proper_images(data3['data']),
                        get_proper_images(data4['data']),
                        get_proper_images(data5['data'])))
    Y = np.concatenate((onehot_labels(data1['labels']),
                        onehot_labels(data2['labels']),
                        onehot_labels(data3['labels']),
                        onehot_labels(data4['labels']),
                        onehot_labels(data5['labels'])))

    X_test = get_proper_images(unpickle('../cifar-10-batches-py/test_batch')['data'])
    Y_test = onehot_labels(unpickle('../cifar-10-batches-py/test_batch')['labels'])

    img_prep = ImagePreprocessing()
    if data_norm:
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()

    img_aug = ImageAugmentation()
    if data_augmentation:
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=30.)
        img_aug.add_random_crop((32, 32), 6)

    return X, Y, X_test, Y_test, img_prep, img_aug

def my_model(img_prep, img_aug):

    # dropout_probability = 0.5
    dropout_probability = 1.0
    initial_learning_rate = 0.0001
    learning_decay=1E-5
    initializer = 'uniform_scaling'
    # initializer = 'truncated_normal'
    activation_function = 'relu'
#     activation_function = 'sigmoid'
    objective_function = 'categorical_crossentropy'
#     objective_function = 'mean_square'

    network = input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, strides=1, padding='same', activation=activation_function,
                      bias=True, bias_init='zeros', weights_init=initializer)
    network = max_pool_2d(network, 2, strides=None, padding='same')
    network = conv_2d(network, 64, 3, strides=1, padding='same', activation=activation_function,
                      bias=True, bias_init='zeros', weights_init=initializer)
    network = conv_2d(network, 64, 3, strides=1, padding='same', activation=activation_function,
                      bias=True, bias_init='zeros', weights_init=initializer)
    network = max_pool_2d(network, 2, strides=None, padding='same')
    network = fully_connected(network, 512, activation=activation_function)
    network = dropout(network, dropout_probability)
    network = fully_connected(network, 10, activation='softmax')
    # network = regression(network, optimizer='adam',
    #                      loss=objective_function,
    #                      learning_rate=initial_learning_rate)
    sgd = SGD(learning_rate=initial_learning_rate, lr_decay=learning_decay, decay_step=90)
    network = regression(network, optimizer=sgd,
                         loss='categorical_crossentropy')
    return network

def main():
    x, y, x_test, y_test, img_prep, img_aug = get_data()
#     with tf.device('/gpu:0'):
#         with tf.contrib.framework.arg_scope([tflearn.variables.variable], device='/cpu:0'):
    model = my_model(img_prep, img_aug)
    network = DNN(model)
    network.fit(x, y, n_epoch=100, shuffle=True, validation_set=(x_test, y_test), show_metric=True,
                batch_size=100, run_id='aa2')
    return

main()
