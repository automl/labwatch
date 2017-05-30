from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from sacred import Experiment
from labwatch.assistant import LabAssistant
from labwatch.hyperparameters import UniformNumber, UniformFloat
from labwatch.optimizers import BayesianOptimization

ex = Experiment()
a = LabAssistant(ex, "labwatch_demo_keras", optimizer=BayesianOptimization)


@ex.config
def cfg():
    batch_size = 128
    num_units_first_layer = 512
    num_units_second_layer = 512
    dropout_first_layer = 0.2
    dropout_second_layer = 0.2
    learning_rate = 0.001



@a.search_space
def small_search_space():
    batch_size = UniformNumber(lower=32, upper=64, default=32, type=int, log_scale=True)
    learning_rate = UniformFloat(lower=10e-3, upper=10e-2, default=10e-2, log_scale=True)


@a.search_space
def large_search_space():
    batch_size = UniformNumber(lower=8, upper=64, default=32, type=int, log_scale=True)
    num_units_first_layer = UniformNumber(lower=16, upper=1024, default=32, type=int, log_scale=True)
    num_units_second_layer = UniformNumber(lower=16, upper=1024, default=32, type=int, log_scale=True)
    dropout_first_layer = UniformFloat(lower=0, upper=.99, default=.2)
    dropout_second_layer = UniformFloat(lower=0, upper=.99, default=.2)
    learning_rate = UniformFloat(lower=10e-6, upper=10e-1, default=10e-2, log_scale=True)


@ex.automain
def run(batch_size,
        num_units_first_layer,
        num_units_second_layer,
        dropout_first_layer,
        dropout_second_layer,
        learning_rate):

    num_classes = 10
    epochs = 20

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(num_units_first_layer, activation='relu', input_shape=(784,)))
    model.add(Dropout(dropout_first_layer))
    model.add(Dense(num_units_second_layer, activation='relu'))
    model.add(Dropout(dropout_second_layer))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=learning_rate),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    results = dict()
    results["optimization_target"] = 1 - score[1]

    return results
