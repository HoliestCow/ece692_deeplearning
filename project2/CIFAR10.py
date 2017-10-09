
import numpy as np
from glob import glob
import os
# import cPickle
import pickle as cPickle  # for python3
from PIL import Image


class CIFAR10:
    def __init__(self, batch_path=os.path.join(os.getcwd(), './cifar-10-batches-py'),
                 test_data_batchsize=100):
        # self.num_test_samples = 5000
        self.pixel_width = 32
        self.pixel_height = 32
        self.channel_number = 3
        self.category_number = 10
        self.input_size = self.pixel_width * self.pixel_height * self.channel_number
        self.output_size = self.category_number
        self.num_test_samples = 10000
        
        self.filehandle_index = -1

        self.batch_path = batch_path
        self.get_batch_filehandles()
        # self.test_batch, self.test_labels = self.get_test_data()
        self.toggle = 0
#         self.prep_train_data()
        self.prep_test_data(test_data_batchsize)
        self.prep_train_data()
        # self.print_image()

    def prep_train_data(self):
        data_container = np.zeros((0, self.input_size))
        label_container = np.zeros((0, self.category_number))
        for i in range(len(self.batch_filehandles)):
            temp = self.get_batch_data()
            data_container = np.vstack((data_container, temp['data']))
            label_container = np.vstack((label_container, temp['labels']))
        print(data_container.shape)
        data_container = np.reshape(data_container, (data_container.shape[0], self.channel_number, self.pixel_height, self.pixel_width))
        print(data_container.shape)
        self.train_mean = data_container.mean(axis=(0, 2, 3), keepdims=False).astype(np.float32)
        self.train_mean = np.reshape(self.train_mean, (3, 1, 1))
        self.train_mean = np.tile(self.train_mean, [1, 32, 32])
        self.train_std = data_container.std(axis=(0, 2, 3), keepdims=False).astype(np.float32)
        self.train_std = np.reshape(self.train_std, (3, 1, 1))
        self.train_std = np.tile(self.train_std, [1, 32, 32])
        return

    def prep_data_4manipulation(self, batch):
        temp_batch = np.reshape(batch, (batch.shape[0], self.channel_number, self.pixel_height, self.pixel_width))
        temp_batch = np.divide(np.subtract(temp_batch, np.tile(self.train_mean, (temp_batch.shape[0], 1, 1, 1))), np.tile(self.train_std, (temp_batch.shape[0], 1, 1, 1)))
        # NOTE: This should be (nsamples, 3, 32, 32)
        return temp_batch
    
    def prep_manipulation2vanilla(self, batch):
        out = np.reshape(batch, (batch.shape[0], self.input_size))
        return out

    def get_batch_filehandles(self):
        self.batch_filehandles = []
        # print(os.path.join(self.batch_path, 'data_batch_*'))
        filelist = glob(os.path.join(self.batch_path, 'data_batch_*'))
        for item in filelist:
            self.batch_filehandles += [open(item, 'rb')]

    def close_batch_filehandles(self):
        for handle in self.batch_filehandles:
            handle.close()
        return

    def get_batch_data(self):
        # outputs dictionary.
        self.filehandle_index += 1
        if self.filehandle_index >= len(self.batch_filehandles):
            self.filehandle_index = 0
            self.close_batch_filehandles()
            self.get_batch_filehandles()
        # This is for python3, cPickle is really pickle
        # dictionary = cPickle.load(self.batch_filehandles[self.filehandle_index],
        #                          encoding='bytes')
        # # this is for python 2
        dictionary = cPickle.load(self.batch_filehandles[self.filehandle_index])
        num_samples = int(dictionary['data'].shape[0])
        new_index = np.arange(num_samples)
        np.random.shuffle(new_index)

        dictionary['data'] = dictionary['data'][new_index, :]
        # dictionary['labels'] = dictionary['labels'][new_index]
        # dictionary['labels'] = [dictionary['labels'][i] for i in new_index]
        temp = [dictionary['labels'][i] for i in new_index]
        labels = np.zeros((len(temp), len(self.possible_labels)))
        for i in range(len(temp)):
            labels[i, temp[i]] = 1
        dictionary['labels'] = labels
        return dictionary

    def coin_flip(self):
        if np.random.uniform() >= 0.5:
            return True
        else:
            return False

    def get_batch(self, samples=100, isHorizontalFlip=False, isHorizontalShift=False,
                  isVerticalShift=False):
        data = self.get_batch_data()
        # Python 2
        index = range(int(data['data'].shape[0] / samples) - 1)
        # python 3
        # index = list(range(int(data[b'data'].shape[0] / samples) - 1))
        for i in index:
            start = index[i] * samples
            end = ((index[i] + 1) * samples)
            # Python 2
            # payload = (data['data'][start:end, :], data['labels'][start:end, :])
            batch = data['data'][start:end, :]
            raw_labels = data['labels'][start:end, :]
            
            # convert data into z-scores and do data augmentation
            temp_batch = self.prep_data_4manipulation(batch)
            for j in range(batch.shape[0]):
                if isHorizontalFlip and self.coin_flip():
                    temp_batch[j, :, : , :] = temp_batch[j, :, :, ::-1] 
            batch = self.prep_manipulation2vanilla(temp_batch)
            # Python 3
            # batch = np.array(data[b'data'][start:end, :])
            # raw_labels = np.array(data[b'labels'][start:end])
            # labels = np.zeros((len(raw_labels), len(self.possible_labels)))
            # for i in range(len(raw_labels)):
                # labels[i, raw_labels[i]] = 1
            labels = raw_labels 
            yield batch
            yield labels
        return

    def get_train_data(self):
        out_data = np.zeros(shape=(0, self.pixel_height * self.pixel_width * self.channel_number))
        out_labels = np.zeros(shape=(0, self.category_number))
        for i in range(len(self.batch_filehandles)):
            dictionary = self.get_batch_data()
            out_data = np.concatenate((out_data, np.array(dictionary['data'])), axis=0)
            out_labels = np.concatenate((out_labels, np.array(dictionary['labels'])), axis=0)
        # print(out_data.shape, out_labels.shape)
        return out_data, out_labels


    def prep_test_data(self, nsamples=100):
        if self.toggle==1:
            self.test_f.close()
        self.toggle = 1
        self.nsamples = nsamples
        filepath = os.path.join(self.batch_path, 'test_batch')
        self.test_f = open(filepath, 'rb')
        # data = cPickle.load(f, encoding='bytes')  # This is for python 3, cPickle is really pickle
        data = cPickle.load(self.test_f)  # this is for python 2
        self.test_data = np.array(data['data'])
        raw_labels = np.array(data['labels'])
        # This is for python3
        # payload = (data[b'data'], data[b'labels'])
        # batch = np.array(data[b'data'])
        # raw_labels = np.array(data[b'labels'])
        self.possible_labels = np.unique(raw_labels)
        labels = np.zeros((len(raw_labels), len(self.possible_labels)))
        for i in range(len(raw_labels)):
            labels[i, raw_labels[i]] = 1

        self.test_labels = labels
        self.test_batch_samples = nsamples

        # fixed the sets such that TF can understand what's going on with this dataset
        self.num_test_epochs = int(labels.shape[0] / nsamples)
        self.input_size = self.test_data.shape[1]
        self.output_size = len(self.possible_labels)
        return
    
    def get_test_data(self):
        cursor = 0
        for i in range(self.num_test_epochs):
            # print(self.test_data[cursor:cursor + self.test_batch_samples, :])
            # print(self.test_labels[cursor:cursor + self.test_batch_samples, :]
            
            temp = self.prep_data_4manipulation(self.test_data[cursor:cursor + self.test_batch_samples, :])
            data = self.prep_manipulation2vanilla(temp)
            yield data
            yield self.test_labels[cursor:cursor + self.test_batch_samples, :]
            cursor += self.test_batch_samples
        self.prep_test_data(nsamples=self.nsamples)
        return

    def print_image(self):
        test_batch = self.get_test_data()
        x = next(test_batch)
        y = next(test_batch)
        single_img = x[0, :]
        single_img_reshaped = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))
        img = Image.fromarray(single_img_reshaped, 'RGB')
        img.save('test.png')
        return

