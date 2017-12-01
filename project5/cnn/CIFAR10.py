import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.models.dnn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.optimizers import SGD
from tflearn.metrics import Accuracy, R2
import numpy as np
import tensorflow as tf
import urllib

import time
import h5py
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def get_proper_images(raw):
    raw_float = np.array(raw, dtype=float)
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])
    return images

def onehot_labels(labels):
    return np.eye(7)[labels]

def onenothot_labels(labels):
    out = np.zeros((labels.shape[0],))
    for i in range(labels.shape[0]):
        out[i] = np.argmax(labels[i, :])
    return out

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def get_data():
    data_norm = True
    data_augmentation = False

    # f = h5py.File('naive_dataset_large.h5', 'r')
    f = h5py.File('naive_dataset_small.h5', 'r')
    g = f['training']
    X = np.zeros((0, 1024))
    Y = []
    for item in g:
        X = np.vstack((X, np.array(g[item]['spectra'])))
        Y += [onehot_labels(np.array(g[item]['label']))]

    Y = np.array(Y)

    g = f['testing']
    X_test = np.zeros((0, 1024))
    Y_test = []
    for item in g:
        X_test = np.vstack((X_test, np.array(g[item]['spectra'])))
        Y_test += [onehot_labels(np.array(g[item]['label']))]

    Y_test = np.array(Y_test)

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
    # objective_function = 'mean_square'
    chooseme = R2()

    network = input_data(shape=[None, 1024, 1],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    # network = conv_2d(network, 32, 3, strides=1, padding='same', activation=activation_function,
    #                  bias=True, bias_init='zeros', weights_init=initializer,
    #                  regularizer='L2')
    network = conv_1d(network, 32, 4, strides=1, padding='same', 
                      activation=activation_function,
                      bias=True,
                      bias_init='zeros',
                      weights_init=initializer)
    # network = max_pool_2d(network, 2, strides=None, padding='same')
    # temporarily removing this puppy
    network = max_pool_1d(network, 2, strides=None, padding='same')
    # network = conv_2d(network, 64, 3, strides=1, padding='same', activation=activation_function,
    #                   bias=True, bias_init='zeros', weights_init=initializer)
    network = conv_1d(network, 64, 4, strides=1, padding='same',
                      activation=activation_function,
                      bias=True,
                      bias_init='zeros',
                      weights_init=initializer)
    # network = conv_2d(network, 64, 3, strides=1, padding='same', activation=activation_function,
    #                   bias=True, bias_init='zeros', weights_init=initializer)
    network = conv_1d(network, 64, 4, strides=1, padding='same',
                      activation=activation_function,
                      bias=True,
                      bias_init='zeros',
                      weights_init=initializer)
    # network = max_pool_2d(network, 2, strides=None, padding='same')
    network = max_pool_1d(network, 2, strides=None, padding='same')
    network = fully_connected(network, 512, activation=activation_function)
    network = dropout(network, dropout_probability)
    network = fully_connected(network, 7, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss=objective_function,
                         learning_rate=initial_learning_rate,
                         metric=chooseme)
    # sgd = SGD(learning_rate=initial_learning_rate, lr_decay=learning_decay, decay_step=90)
    # network = regression(network, optimizer=sgd,
    #                      loss='categorical_crossentropy')
    return network

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    a = time.time()
    x, y, x_test, y_test, img_prep, img_aug = get_data()  # modified for spectra
    b = time.time()

    x = x.reshape((x.shape[0], x.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
    print('data time: {}'.format(b-a))
#     with tf.device('/gpu:0'):
#         with tf.contrib.framework.arg_scope([tflearn.variables.variable], device='/cpu:0'):
    model = my_model(img_prep, img_aug)
    network = DNN(model)
    a = time.time()
    network.fit(x, y, n_epoch=1, shuffle=True, validation_set=(x_test, y_test), show_metric=True,
                batch_size=32, run_id='aa2')
    print(network.evaluate(x_test[0:32, :], y_test[0:32, :]))
    b = time.time()
    print('total time: {}'.format(b-a))

    # evali= model.evaluate(x_test, y_test)
    # print("Accuracy of the model is :", evali)
    divideby = 100
    dindex = int(x_test.shape[0] / divideby)
    labels = np.zeros((x_test.shape[0], 7))
    for i in range(divideby):
        start = i * dindex
        end = start + dindex
        prob_y = network.predict(x_test[start:end, :])
        y = network.predict_label(x_test[start:end, :])
        predictions = np.argmax(y, axis=1)
        for j in range(len(predictions)):
            labels[start + j, predictions[j]] = 1

    appendme = np.array([0,1,2,3,4,5,6])
    appendme = appendme.reshape((7,))
    y_test_decode = onenothot_labels(y_test)
    y_test_decode = np.concatenate((y_test_decode, appendme))
    labels_decode = onenothot_labels(labels)
    labels_decode = np.concatenate((labels_decode, appendme))

    accuracy = float(np.sum(labels_decode == y_test_decode)) / float(y_test_decode.shape[0])
    

    class_names = ['Background',
                   'HEU',
                   'WGPu',
                   'I131',
                   'Co60',
                   'Tc99',
                   'HEUandTc99']

    cnf_matrix = confusion_matrix(y_test_decode, labels_decode)

    # print("The predicted labels are :", lables[f])
    # prediction = model.predict(testImages)
    # print("The predicted probabilities are :", prediction[f])
    fig = plt.figure()
    class_names = ['Background',
                   'HEU',
                   'WGPu',
                   'I131',
                   'Co60',
                   'Tc99',
                   'HEUandTc99']
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    fig.savefig('classification_confusion_matrix.png')
    return

main()
