

from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
# import glob
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from itertools import compress
import numpy as np
import os.path

from linear_classifier import SVM

import time

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


def get_proper_images(raw):
    raw_float = np.array(raw, dtype=float)
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])
    return images


def get_data():
    data_norm = True
    data_augmentation = True

    data1 = unpickle('../cifar-10-batches-py/data_batch_1')
    data2 = unpickle('../cifar-10-batches-py/data_batch_2')
    data3 = unpickle('../cifar-10-batches-py/data_batch_3')
    data4 = unpickle('../cifar-10-batches-py/data_batch_4')
    data5 = unpickle('../cifar-10-batches-py/data_batch_5')
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


def main():
    a = time.time()
    x, y, x_test, y_test, img_prep, img_aug = get_data()

    svm_y = np.zeros((y.shape[0], ), dtype=int)
    svm_y_test = np.zeros((y_test.shape[0]), dtype=int)
    for i in range(y.shape[0]):
        # print(y[i, :] == 1)
        mask =  y[i, :] == 1
        meh = list(compress(range(len(mask)), mask))
        svm_y[i] = int(meh[0])
    for i in range(y_test.shape[0]):
        mask = y_test[i, :] == 1
        meh = list(compress(range(len(mask)), mask))
        svm_y_test[i] = int(meh[0])

    # runs = ['sigmoid_sigmoid_256',
    #         'sigmoid_sigmoid_crossentropy_256',
    #         'sigmoid_sigmoid_gaussiannoise_256',
    #         'relu_relu_256']

    runs = ['sigmoid_tanh_256']

    print('time required to fix the answers {}'.format(time.time() - a))

    # feature_generator = DNN(features, session=network.session)

    # if len(glob.glob('./data/dae/*train.npy')) != 1:
    #     svm_features = np.zeros((0, 512))
    #     for i in range(x.shape[0]):
    #         if i % 1000 == 0:
    #             print(i, svm_features.shape)
    #         chuckmein = x[i, :, :].reshape((1, x.shape[1], x.shape[2], x.shape[3]))
    #         svm_features = np.vstack((svm_features, feature_generator.predict(chuckmein)))
    #     np.save('./dae_svm_features.npy', svm_features)
    # else:
    #     svm_features = np.load('./dae_svm_features.npy')

    model_directory = './data/dae/'
    train_suffix = '-train.npy'
    test_suffix = '-test.npy'
    # validation_suffix = '-validate.npy'

    for item in runs:
        svm_features = np.load(os.path.join(model_directory, item + train_suffix))
        svm_features_test = np.load(os.path.join(model_directory, item + test_suffix))

    # if len(glob.glob('./data/dae/*test.npy')) != 1:
    #     svm_features_test = np.zeros((0, 512))
    #     for i in range(x_test.shape[0]):
    #         chuckmein = x_test[i, :, :].reshape((1, x.shape[1], x.shape[2], x.shape[3]))
    #         svm_features_test = np.vstack((svm_features_test, feature_generator.predict(chuckmein)))
    #     np.save('./dae_svm_features_test.npy', svm_features_test)
    # else:
    #     svm_features_test = np.load('./dae_svm_features_test.npy')
    #  from here it's y vs. y_predict
        # start = time.time()
        # clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='auto'))
        # clf.fit(X, y)
        # end = time.time()
        # print "Single SVC", end - start, clf.score(X,y)
        # proba = clf.predict_proba(X)
        #
        # n_estimators = 10
        # start = time.time()
        # clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
        # clf.fit(X, y)
        # end = time.time()
        # print "Bagging SVC", end - start, clf.score(X,y)
        # proba = clf.predict_proba(X)

        # start = time.time()
        # clf = RandomForestClassifier(min_samples_leaf=20)
        # clf.fit(X, y)
        # end = time.time()
        # print "Random Forest", end - start, clf.score(X,y)
        # proba = clf.predict_proba(X)
        #
        # n_estimators = 10
        # n_jobs = 4
        # start = time.time()
        # clf = OneVsRestClassifier(BaggingClassifier(
        #     SVC(kernel='linear', probability=True, class_weight=None),
        #     max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=n_jobs))
        # clf.fit(svm_features, svm_y)
        # end = time.time()
        # print("Bagging SVC", end - start, clf.score(svm_features_test, svm_y_test))
        # proba = clf.predict_proba(X)

        # Training svm regression classifier using SGD and BGD


        # # using SGD algorithm
        SVM_sgd = SVM()
        tic = time.time()
        # print(svm_y)
        svm_features = svm_features.reshape((svm_features.shape[1], svm_features.shape[0]))
        # losses_sgd = SVM_sgd.train(svm_features, svm_y, method='sgd', batch_size=200, learning_rate=1e-6, reg = 1e5, num_iters=10000, verbose=True, vectorized=True)
        losses_sgd = SVM_sgd.train(svm_features, svm_y, method='sgd', batch_size=200, learning_rate=1e-5, num_iters=1000, verbose=True, vectorized=True)
        toc = time.time()
        print('Traning time for SGD with vectorized version is %f \n' % (toc - tic))

        # y_train_pred_sgd = SVM_sgd.predict(X_train)[0]
        # print 'Training accuracy: %f' % (np.mean(y_train == y_train_pred_sgd))
        svm_features_test = svm_features_test.reshape((svm_features_test.shape[1], svm_features_test.shape[0]))
        y_val_pred_sgd = SVM_sgd.predict(svm_features_test)[0]
        print(y_val_pred_sgd)
        print('Validation accuracy: %f' % (np.mean(svm_y_test == y_val_pred_sgd)))

        # predicted_y = clf.predict(svm_features_test)
        # accuracy_mask = svm_y_test == predicted_y
        # accuracy = float(len(list(compress(range(len(accuracy_mask)), accuracy_mask)))) / float(len(accuracy_mask))
        # print('{} accuracy: {}'.format(item, accuracy))
        # print('time taken to complete run {}'.format(time.time() - a))



    # y_test vs. predicted_y metric
    # print('total time taken: {}'.format(b - a))
    return

main()