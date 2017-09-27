
import numpy as np
from glob import glob
import os
# import cPickle
import pickle as cPickle  # for python3


class CIFAR10:
    def __init__(self, batch_path=os.path.join(os.getcwd(), './cifar-10-batches-py')):
        self.batch_path = batch_path
        self.get_batch_filehandles()
        self.test_batch, self.test_labels = self.get_test_data()
        self.filehandle_index = -1

    def get_batch_filehandles(self):
        self.batch_filehandles = []
        print(os.path.join(self.batch_path, 'data_batch_*'))
        filelist = glob(os.path.join(self.batch_path, 'data_batch_*'))
        print(filelist)
        print(filelist)
        for item in filelist:
            self.batch_filehandles += [open(item, 'rb')]

    def get_batch_data(self):
        # outputs dictionary.
        self.filehandle_index += 1
        if self.filehandle_index >= len(self.batch_filehandles):
            self.filehandle_index = 0
        # This is for python3, cPickle is really pickle
        dictionary = cPickle.load(self.batch_filehandles[self.filehandle_index],
                                  encoding='bytes')
        # # this is for python 2
        # dictionary = cPickle.load(self.batch_filehandles[self.filehandle_index])
        return dictionary

    def get_batch(self, samples=100):
        data = self.get_batch_data()
        # Python 2
        # index = range(int(data['data'].shape[0] / samples) - 1)
        # python 3
        index = list(range(int(data[b'data'].shape[0] / samples) - 1))
        for i in index:
            start = index[i] * samples
            end = ((index[i] + 1) * samples)
            # Python 2
            # payload = (data['data'][start:end, :], data['labels'][start:end, :])
            # batch = np.array(data['data'][start:end, :])
            # raw_labels = np.array(data['labels'][start:end, :])
            # Python 3
            batch = np.array(data[b'data'][start:end, :])
            raw_labels = np.array(data[b'labels'][start:end])
            labels = np.zeros((len(raw_labels), len(self.possible_labels)))
            for i in range(len(raw_labels)):
                labels[i, raw_labels[i]] = 1
            yield batch
            yield labels
        return

    def get_test_data(self):
        filepath = os.path.join(self.batch_path, 'test_batch')
        f = open(filepath, 'rb')
        data = cPickle.load(f, encoding='bytes')  # This is for python 3, cPickle is really pickle
        # data = cPickle.load(f)  # this is for python 2
        # This is for python3
        # payload = (data[b'data'], data[b'labels'])
        batch = np.array(data[b'data'])
        raw_labels = np.array(data[b'labels'])
        self.possible_labels = np.unique(raw_labels)
        labels = np.zeros((len(raw_labels), len(self.possible_labels)))
        for i in range(len(raw_labels)):
            labels[i, raw_labels[i]] = 1
        self.input_size = batch.shape[1]
        self.output_size = len(self.possible_labels)
        return batch, labels
