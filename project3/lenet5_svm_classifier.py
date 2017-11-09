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

from sklearn import svm
from itertools import compress
import glob

def get_proper_images(raw):
    raw_float = np.array(raw, dtype=float)
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])
    return images

def onehot_labels(labels):
    return np.eye(10)[labels]

def unpickle(file):
    # import cPickle
    import pickle as cPickle
    fo = open(file, 'rb')
    # dict = cPickle.load(fo)
    dict = cPickle.load(fo, encoding='bytes')
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
    # print(list(data1.keys()))
    # X = np.concatenate((get_proper_images(data1['data']),
    #                     get_proper_images(data2['data']),
    #                     get_proper_images(data3['data']),
    #                     get_proper_images(data4['data']),
    #                     get_proper_images(data5['data'])))
    X = np.concatenate((get_proper_images(data1[b'data']),
                        get_proper_images(data2[b'data']),
                        get_proper_images(data3[b'data']),
                        get_proper_images(data4[b'data']),
                        get_proper_images(data5[b'data'])))
    # Y = np.concatenate((onehot_labels(data1['labels']),
    #                     onehot_labels(data2['labels']),
    #                     onehot_labels(data3['labels']),
    #                     onehot_labels(data4['labels']),
    #                     onehot_labels(data5['labels'])))
    Y = np.concatenate((onehot_labels(data1[b'labels']),
                        onehot_labels(data2[b'labels']),
                        onehot_labels(data3[b'labels']),
                        onehot_labels(data4[b'labels']),
                        onehot_labels(data5[b'labels'])))

    # X_test = get_proper_images(unpickle('../cifar-10-batches-py/test_batch')['data'])
    # Y_test = onehot_labels(unpickle('../cifar-10-batches-py/test_batch')['labels'])

    X_test = get_proper_images(unpickle('../cifar-10-batches-py/test_batch')[b'data'])
    Y_test = onehot_labels(unpickle('../cifar-10-batches-py/test_batch')[b'labels'])

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

    dropout_probability = 0.5
    # dropout_probability = 1.0
    initial_learning_rate = 0.0001
    # learning_decay = 1E-5
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
    features = fully_connected(network, 512, activation=activation_function, name='derived_features')
    network = dropout(features, dropout_probability)
    network = fully_connected(network, 10, activation='softmax',
                              name='softmax_classification')
    network = regression(network, optimizer='adam',
                         loss=objective_function,
                         learning_rate=initial_learning_rate)
    # sgd = SGD(learning_rate=initial_learning_rate, lr_decay=learning_decay, decay_step=90)
    # network = regression(network, optimizer=sgd,
    #                      loss='categorical_crossentropy')
    return network, features


def main():
    x, y, x_test, y_test, img_prep, img_aug = get_data()
#     with tf.device('/gpu:0'):
#         with tf.contrib.framework.arg_scope([tflearn.variables.variable], device='/cpu:0'):
    model, features = my_model(img_prep, img_aug)
    network = DNN(model)
    # network.fit(x, y, n_epoch=100, shuffle=True, validation_set=(x_test, y_test), show_metric=True,
    #             batch_size=100, run_id='aa2')
    # task 3 stuff
    network.load('./lenet5_run.tflearn')
    feature_generator = DNN(features, session=network.session)
    if len(glob.glob('./lenet5_svm_features.npy')) != 1:
        svm_features = np.zeros((0, 512))
        for i in range(x.shape[0]):
            if i % 1000 == 0:
                print(i, svm_features.shape)
            chuckmein = x[i, :, :].reshape((1, x.shape[1], x.shape[2], x.shape[3]))
            svm_features = np.vstack((svm_features, feature_generator.predict(chuckmein)))
        np.save('./lenet5_svm_features.npy', svm_features)
    else:
        svm_features = np.load('./lenet5_svm_features.npy')

    if len(glob.glob('./lenet5_svm_features_test.npy')) != 1:
        svm_features_test = np.zeros((0, 512))
        for i in range(x_test.shape[0]):
            chuckmein = x_test[i, :, :].reshape((1, x.shape[1], x.shape[2], x.shape[3]))
            svm_features_test = np.vstack((svm_features_test, feature_generator.predict(chuckmein)))
        np.save('./lenet5_svm_features_test.npy', svm_features_test)
    else:
        svm_features_test = np.load('./lenet5_svm_features_test.npy')
    #  from here it's y vs. y_predict
    svm_y = np.zeros((y.shape[0], ))
    svm_y_test = np.zeros((y_test.shape[0]))
    for i in range(y.shape[0]):
        # print(y[i, :] == 1)
        mask =  y[i, :] == 1
        meh = list(compress(range(len(mask)), mask))
        svm_y[i] = meh[0]
    for i in range(y_test.shape[0]):
        mask = y_test[i, :] == 1
        meh = list(compress(range(len(mask)), mask))
        svm_y_test[i] = meh[0]

    clf = svm.SVC()
    clf.fit(svm_features, svm_y)
    predicted_y = clf.predict(svm_features_test)
    accuracy_mask = svm_y_test == predicted_y
    accuracy = float(len(list(compress(range(len(accuracy_mask)), accuracy_mask)))) / float(len(accuracy_mask))
    print(accuracy)

    # y_test vs. predicted_y metric

    return

main()
