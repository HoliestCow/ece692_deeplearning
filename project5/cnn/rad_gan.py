""" GAN Example
Use a generative adversarial network (GAN) to generate digit images from a
noise distribution.
References:
    - Generative adversarial nets. I Goodfellow, J Pouget-Abadie, M Mirza,
    B Xu, D Warde-Farley, S Ozair, Y. Bengio. Advances in neural information
    processing systems, 2672-2680.
Links:
    - [GAN Paper](https://arxiv.org/pdf/1406.2661.pdf).
"""

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tflearn
import h5py

# Data loading and preprocessing
# import tflearn.datasets.mnist as mnist
# X, Y, testX, testY = mnist.load_data()
# bg = np.load('./integrations/bg_spectra_only.npy')
dataset = h5py.File('./spectral_data.h5', 'r')
x = np.array(dataset['training_data'], dtype=float)
x_test = np.array(dataset['testing_data'], dtype=float)
# validation_dataset = np.array(dataset['validation_data'])

# image_dim = 784 # 28*28 pixels
image_dim = 1024
z_dim = 200 # Noise data points
total_samples = x.shape[0]

def samplewise_mean(x):
    for i in range(x.shape[0]):
        mean = np.mean(x[i, :])
        std = np.std(x[i, :])
        z = np.divide(np.subtract(x[i, :], mean), std)
        x[i, :] = z

    return x

x = samplewise_mean(x)
x_test = samplewise_mean(x)


# Generator
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tflearn.fully_connected(x, 512, activation='relu')
        x = tflearn.fully_connected(x, image_dim, activation='sigmoid')
        return x


# Discriminator
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tflearn.fully_connected(x, 512, activation='relu')
        x = tflearn.fully_connected(x, 1, activation='sigmoid')
        return x

# Build Networks
gen_input = tflearn.input_data(shape=[None, z_dim], name='input_noise')
disc_input = tflearn.input_data(shape=[None, image_dim], name='disc_input')

gen_sample = generator(gen_input)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample, reuse=True)

# Define Loss
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
gen_loss = -tf.reduce_mean(tf.log(disc_fake))

# Build Training Ops for both Generator and Discriminator.
# Each network optimization should only update its own variable, thus we need
# to retrieve each network variables (with get_layer_variables_by_scope) and set
# 'placeholder=None' because we do not need to feed any target.
gen_vars = tflearn.get_layer_variables_by_scope('Generator')
gen_model = tflearn.regression(gen_sample, placeholder=None, optimizer='adam',
                               loss=gen_loss, trainable_vars=gen_vars,
                               batch_size=64, name='target_gen', op_name='GEN')
disc_vars = tflearn.get_layer_variables_by_scope('Discriminator')
disc_model = tflearn.regression(disc_real, placeholder=None, optimizer='adam',
                                loss=disc_loss, trainable_vars=disc_vars,
                                batch_size=64, name='target_disc', op_name='DISC')
# Define GAN model, that output the generated images.
gan = tflearn.DNN(gen_model)

# Training
# Generate noise to feed to the generator
z = np.random.uniform(-1., 1., size=[total_samples, z_dim])
# z = np.random.poisson(lam=1, size=z_dim)
# Start training, feed both noise and real images.
gan.fit(X_inputs={gen_input: z, disc_input: x},
        Y_targets=None,
        n_epoch=20)

# Generate images from noise, using the generator network.
f, a = plt.subplots(2, 2, figsize=(10, 4))
for i in range(2):
    for j in range(2):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[1, z_dim])
        # z = np.random.poisson(lam=1, size=z_dim)
        # Generate image from noise. Extend to 3 channels for matplot figure.
        # temp = [[ii, ii, ii] for ii in list(gan.predict([z])[0])]
        # a[j][i].imshow(np.reshape(temp, (28, 28, 3)))
        temp = list(gan.predict([z])[0])
        a[j][i].plot(temp)

f.savefig('test_gan.png')
