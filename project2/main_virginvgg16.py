

import sys
sys.path.append('/home/cbritt2/keras/build/lib')

# from keras.preprocessing import image
from keras import optimizers
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import TruncatedNormal
from virgin_convert_vgg16 import VGG16
from CIFAR10 import CIFAR10
import numpy as np


def main():
    data = CIFAR10()
    data.prep_test_data(nsamples=10000)
    test_data, test_labels = data.get_test_data()
    train_data, train_labels = data.get_train_data()
    test_data = np.transpose(np.reshape(test_data, (test_data.shape[0], 3, 32, 32)), (0, 2, 3, 1))
    train_data = np.transpose(np.reshape(train_data, (train_data.shape[0], 3, 32, 32)), (0, 2, 3, 1))

    # augment the training_data
    train_data_generator = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        data_format='channels_last')

    batch_size = 100

    # actual_train_data = train_data_generator.flow(train_data, train_labels, batch_size=batch_size)

    initializer = TruncatedNormal(mean=0.0, stddev=0.001, seed=None)
    
    lr = 1E-5
    epochs = 2
    dropout = 0.0
    model = VGG16(input_shape=(32, 32, 3), weights='imagenet', classes=10,
                  include_top=False)
    # model = VGG16(input_shape=(32, 32, 3), weights=None, classes=10)
    x = model.get_layer('block3_pool').output
    x = Flatten(name='Flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(dropout)(x)
    predictions = Dense(10, activation='softmax')(x)
    converted_model = Model(input=model.input, output=predictions)
    sgd = optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
    converted_model.compile(loss='categorical_crossentropy',
                            optimizer=sgd,
                            metrics=['accuracy'])
    converted_model.fit(x=train_data, y=train_labels, batch_size=batch_size, validation_split=0.2, epochs=epochs)
    # NOTE: only use below when using data augmentation.
    # converted_model.fit_generator(actual_train_data, steps_per_epoch=10000/batch_size, epochs=100, verbose=1, validation_data=(test_data, test_labels), use_multiprocessing=False)
    score = converted_model.evaluate(x=test_data, y=test_labels, batch_size=100)
    print('Testing accuracy: {}'.format(score))

    return

main()
