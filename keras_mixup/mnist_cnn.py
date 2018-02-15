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

batch_size = 128
num_classes = 10
epochs = 10
(x_train, y_train), (x_test, y_test) = mnist_data()

model = mnist_cnn()

tgen = MixTensorGenerator(x_train, y_train, batch_size=batch_size, alpha=0.2)
vgen = TensorGenerator(x_test, y_test, batch_size=batch_size)
model.fit_generator(tgen(),
          epochs=epochs,
          steps_per_epoch= len(tgen),
          verbose=1,
          validation_data=vgen(),
          validation_steps=len(vgen))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
