

import sys
sys.path.append('/home/cbritt2/keras/build/lib')

# from keras.preprocessing import image
from virgin_convert_vgg16 import VGG16
from CIFAR10 import CIFAR10

def main():
    data = CIFAR10()
    model = VGG16(input_shape=(32, 32, 3))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return

main()
