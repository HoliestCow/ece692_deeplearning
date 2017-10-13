

import sys
sys.path.append('/home/cbritt2/keras/build/lib')

# from keras.preprocessing import image
from keras import optimizers
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import TruncatedNormal
from virgin_convert_vgg16 import VGG16
# from CIFAR10 import CIFAR10
from keras.datasets.cifar10 import load_data
from keras import utils
import numpy as np


def main():
    # test_data = np.transpose(np.reshape(test_data, (test_data.shape[0], 3, 32, 32)), (0, 2, 3, 1))
    # train_data = np.transpose(np.reshape(train_data, (train_data.shape[0], 3, 32, 32)), (0, 2, 3, 1))
    (x_train, y_train), (x_test, y_test) = load_data()

    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)

    
    # augment the training_data
    train_data_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        data_format='channels_last')

    # actual_train_data = train_data_generator.flow(train_data, train_labels, batch_size=batch_size)

    # initializer = TruncatedNormal(mean=0.0, stddev=0.001, seed=None)
    batch_size = 100    
    lr = 0.01
    lr_decay = 1E-6
    epochs = 10
    dropout = 0.5
    # model = VGG16(input_shape=(32, 32, 3), weights='imagenet', classes=10,
                  # include_top=False, input_tensor=None)
    model = VGG16(input_shape=(32, 32, 3), weights=None, classes=10)
    x = model.get_layer('block3_pool').output
    x = Flatten(name='Flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(dropout)(x)
    predictions = Dense(10, activation='softmax')(x)
    converted_model = Model(input=model.input, output=predictions)
    sgd = optimizers.SGD(lr=lr, decay=lr_decay, momentum=0.9, nesterov=True)
    converted_model.compile(loss='categorical_crossentropy',
                            optimizer=sgd,
                            metrics=['accuracy'])
    # converted_model.fit(x=x_train, y=y_train, batch_size=batch_size, validation_split=0.2, epochs=epochs)
    # NOTE: only use below when using data augmentation.
    converted_model.fit_generator(train_data_generator.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=10000/batch_size, epochs=epochs, validation_data=(x_test, y_test))

    return

main()
