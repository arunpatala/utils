'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

from keras.callbacks import ModelCheckpoint, LambdaCallback

batch_size = 100
original_dim = 784
latent_dim = 784
intermediate_dim = 784
epochs = 50
epsilon_std = 1.0


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def get_vae(original_dim, latent_dim):
    inp = Input(shape=(original_dim,))
    x = Dense(intermediate_dim, activation='relu')(inp)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    enc = Model(inp,z_mean)
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    def dec(y):
        h_decoded = decoder_h(y)
        x_decoded_mean = decoder_mean(h_decoded)
        return x_decoded_mean
    y = Input(shape=(latent_dim,))
    x_decoded_mean = dec(z)
    vae = Model(inp, x_decoded_mean)   
    vae.summary() 
    # Compute VAE loss
    xent_loss = original_dim * metrics.binary_crossentropy(inp, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    vae.add_loss(vae_loss)
    return vae, enc, Model(y,dec(y))

vae, enc, dec = get_vae(original_dim, latent_dim)
vae.compile(optimizer='rmsprop')
vae.summary()

#vae.load_weights('models/vae.ckpt.tgz')

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


def plot_interpolation():
    encoder = enc

    x_test_encoded = encoder.predict(x_test[:11], batch_size=batch_size)
    digit_size = 28
    n = 10
    figure = np.zeros((digit_size * n, digit_size * n))
    for  i in range(10):
        xi = x_test_encoded[i]
        xxi = x_test[i].reshape(digit_size, digit_size)
        yi = x_test_encoded[i+1]
        yyi = x_test[i+1].reshape(digit_size, digit_size)
        for j in range(10):
            xyi = ((xi*(9-j)+yi*(j))/9)
            z_sample = np.array([xyi])
            x_decoded = dec.predict(z_sample)

            digit = x_decoded[0].reshape(digit_size, digit_size)
            if j==0: digit = xxi
            if j==9: digit = yyi
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    #plt.imshow(figure, cmap='Greys_r')
    plt.imsave('models/interpolation.png', figure, cmap='Greys_r')
    plt.close()

plot_interpolation()

lambda_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: plot_interpolation()

)

ckpt = ModelCheckpoint('models/vae.ckpt.tgz')

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None), 
        callbacks=[ckpt, lambda_callback])

