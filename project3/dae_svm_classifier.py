
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
# import glob
# from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
# from itertools import compress
import numpy as np
import os.path

import datasets

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
    # x, y, x_test, y_test, img_prep, img_aug = get_data()
    #
    # svm_y = np.zeros((y.shape[0], ), dtype=int)
    # svm_y_test = np.zeros((y_test.shape[0]), dtype=int)
    # for i in range(y.shape[0]):
    #     # print(y[i, :] == 1)
    #     mask = y[i, :] == 1
    #     meh = list(compress(range(len(mask)), mask))
    #     svm_y[i] = int(meh[0])
    # for i in range(y_test.shape[0]):
    #     mask = y_test[i, :] == 1
    #     meh = list(compress(range(len(mask)), mask))
    #     svm_y_test[i] = int(meh[0])

    # runs = ['sigmoid_sigmoid_256',
    #         'sigmoid_sigmoid_crossentropy_256',
    #         'sigmoid_sigmoid_gaussiannoise_256',
    #         'sigmoid_tanh_512',
    #         'relu_relu_256']

    # runs = ['sigmoid_sigmoid_snp_0.1_512',
    #         'sigmoid_sigmoid_snp_0.2_512',
    #         'sigmoid_sigmoid_snp_0.3_512',
    #         'sigmoid_sigmoid_snp_0.4_512',
    #         'sigmoid_sigmoid_snp_0.5_512']

    # runs = ['sigmoid_sigmoid_mask_0.1_512',
    #         'sigmoid_sigmoid_mask_0.2_512',
    #         'sigmoid_sigmoid_mask_0.3_512',
    #         'sigmoid_sigmoid_mask_0.4_512',
    #         'sigmoid_sigmoid_mask_0.5_512',
    #         'relu_relu_snp_0.4_512']

    # runs = ['sigmoid_sigmoid_gaussian_0.4_512']

    runs = ['forcnn_sigmoid_sigmoid_snp_0.4_675']

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
    encode_w_suffix = '-encw.npy'
    encode_b_suffix = '-encbh.npy'
    # decode_w = '-decw.npy'
    # decode_b = '-decb.npy'
    # train_suffix = '-forcnn_sigmoid_sigmoid_snp_0.4_675.npy'
    train_suffix_answer = '-train-answers.npy'
    # test_suffix = '-test.npy'
    test_suffix_answer = '-test-answers.npy'
    # validation_suffix = '-validate.npy'

    x, y, x_test, y_test = datasets.load_cifar10_dataset('./cifar-10-batches-py', mode='supervised')
    # y = onehot_labels(y)
    # y_test = onehot_labels(y_test)

    for item in runs:
        # svm_features = np.load(os.path.join(model_directory, item + train_suffix))
        # svm_features_test = np.load(os.path.join(model_directory, item + test_suffix))
        encode_w = np.load(os.path.join(model_directory, item + encode_w_suffix))
        encode_b = np.load(os.path.join(model_directory, item + encode_b_suffix))
        encode = np.add(np.dot(x, encode_w), encode_b)
        # svm_features = encode.reshape(x.shape[0], 3, 15, 15).transpose(0, 2, 3, 1)
        svm_features = encode
        encode = np.add(np.dot(x_test, encode_w), encode_b)
        svm_features_test = encode
        # svm_features_test = encode.reshape(x_test.shape[0], 3, 15, 15).transpose(0, 2, 3, 1)
        # print(svm_features.shape, svm_features_test.shape, y.shape, y_test.shape)
        # stop

        n_estimators = 10
        n_jobs = 4
        print('training svm')
        start = time.time()
        clf = OneVsRestClassifier(BaggingClassifier(
            SVC(kernel='linear', probability=True, class_weight=None),
            max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=n_jobs))
        clf.fit(svm_features, y)
        end = time.time()
        print("Bagging SVC", end - start, clf.score(svm_features_test, y_test))



    return

main()