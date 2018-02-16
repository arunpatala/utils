from mixup_generator import *
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from model import *
from mnist import *


(x_train, y_train), (x_test, y_test), input_shape = mnist_data()

vae, enc, dec = get_vae(load='vae.ckpt.tgz')

tgen = VAEMixGenerator(x_train, y_train, enc, dec, batch_size=128, alpha=1.0, shuffle=False)

def vae_save(filepath):
    X,Y = [],[]
    for x,y in tgen():
        #print(x.shape, y.shape)
        X.append(x)
        Y.append(y)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    print(X.shape, Y.shape)
    np.save(filepath+".{}.npz".format('X'), X)
    np.save(filepath+".{}.npz".format('Y'), Y)


for i in range(40):
    print(i)
    vae_save('models/vae.{}'.format(i))