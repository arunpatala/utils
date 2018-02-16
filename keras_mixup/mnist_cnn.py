'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from mixup_generator import *
from mnist import *
from cifar import *
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from model import *
from irctc import *

import argparse
parser = argparse.ArgumentParser(description='alpha mixup')
parser.add_argument('--alpha','-a', type=float, required=True, action="store")
parser.add_argument('--dataset','-d', type=str, default='mnist', action="store", choices=['mnist', 'cifar10', 'irctc'])
parser.add_argument('--epochs','-e', type=int, default=40, action="store")
parser.add_argument('--batch_size','-b', type=int, default=128, action="store")
args = parser.parse_args()
alpha = args.alpha
print('alpha is', alpha)
print('dataset is', args.dataset)
ds = args.dataset
batch_size = args.batch_size

epochs = args.epochs

if ds=='mnist':
  (x_train, y_train), (x_test, y_test), input_shape = mnist_data()
elif ds=='cifar10':
  (x_train, y_train), (x_test, y_test), input_shape = cifar10_data()
elif ds=='irctc':
  (x_train, y_train), (x_test, y_test), input_shape = irctc_data("IRCTC")
else: print("DATASET not found")
num_classes = y_test.shape[1]
model = mnist_cnn(input_shape, num_classes)
#vae, enc, dec = get_vae(load='vae.ckpt.tgz')

print(model.summary())
csv_logger = CSVLogger('models/{}.training.{}.log'.format(ds, alpha))
ckpt = ModelCheckpoint('models/{}.ckpt.{}.tar.gz'.format(ds, alpha), verbose=1, monitor='val_loss')

model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'],
                  )
#tgen = VAEMixGenerator(x_train, y_train, enc, dec, batch_size=batch_size, alpha=alpha)
tgen = MixTensorGenerator(x_train, y_train, batch_size=batch_size, alpha=alpha)
vgen = TensorGenerator(x_test, y_test, batch_size=batch_size)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, min_lr=0.0001, verbose=1)

model.fit_generator(tgen(),
          epochs=epochs,
          steps_per_epoch= len(tgen),
          verbose=1,
          validation_data=vgen(),
          validation_steps=len(vgen),
          callbacks=[csv_logger, ckpt, reduce_lr])

score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
