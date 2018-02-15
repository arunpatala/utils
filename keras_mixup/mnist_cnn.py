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
from keras.callbacks import CSVLogger, ModelCheckpoint


import argparse
parser = argparse.ArgumentParser(description='alpha mixup')
parser.add_argument('--alpha','-a', type=float, required=True, action="store")
args = parser.parse_args()
alpha = args.alpha
print('alpha is', alpha)

batch_size = 128
num_classes = 10
epochs = 20
(x_train, y_train), (x_test, y_test), input_shape = mnist_data()

model = mnist_cnn(input_shape)

csv_logger = CSVLogger('models/training.{}.log'.format(alpha))
ckpt = ModelCheckpoint('models/ckpt{}.tar.gz'.format(alpha))

model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'],
                  )

tgen = MixTensorGenerator(x_train, y_train, batch_size=batch_size, alpha=0.2)
vgen = TensorGenerator(x_test, y_test, batch_size=batch_size)
model.fit_generator(tgen(),
          epochs=epochs,
          steps_per_epoch= len(tgen),
          verbose=1,
          validation_data=vgen(),
          validation_steps=len(vgen),
          callbacks=[csv_logger, ckpt])

score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
