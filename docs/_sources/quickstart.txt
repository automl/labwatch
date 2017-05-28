Quickstart
**********

The following tutorial assumes that you have a basic understanding of how Sacred works (Experiments, Observers).
In case you are unsure, have a look on the `Sacred Quickstart Guide <http://sacred.readthedocs.io/en/latest/>`_

In this tutorial we will see how we can optimize the hyperparameters of a feed forward network trained on MNIST
together with Labwatch and Sacred.
We will use the MNIST example of `keras <https://keras.io/>`_ to implement the neural network but note that
both Labwatch and Sacred are completely independent of which framework you use.
You can find the whole source code as well as more examples
`here <https://github.com/automl/labwatch/blob/master/examples/keras_mnist_mlp.py>`_

Automatically Tuning of Hyperparameters
=======================================


Sacred is a useful tool to keep track of all relevant information of your experiments such as hyperparameters,
results, dependencies and so on.

The following python code adapts the
`keras mnist example <https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py>`_ to work with Sacred:


.. code:: python

    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import RMSprop

    from sacred import Experiment

    ex = Experiment()

    @ex.config
    def cfg():
        batch_size = 128
        num_units_first_layer = 512
        num_units_second_layer = 512
        dropout_first_layer = 0.2
        dropout_second_layer = 0.2
        learning_rate = 0.001


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

In the dark old days you would probably now spend a lot of time to find the right setting for you hyperparameters by
iteratively trying out different setting.

Well, you're are not alone with this problem and it is actually a common problem in machine learning.
Recently a new subfield in machine learning (`AutoML <http://www.ml4aad.org/automl/>`_)
has emerged that tries to automated this procedure by casting it as an optimization problem. By now there exist several
optimization methods that tackle the hyperparameter optimization problem.

To make use of these methods in Labwatch, we fist have to instantiate a LabAssistant which will
connect our Sacred experiment with the hyperparameter optimizer through a MongoDB:

.. code:: python

    from labwatch.assistant import LabAssistant
    from labwatch.optimizers import BayesianOptimization

    a = LabAssistant(ex, "labwatch_demo_keras", optimizer=BayesianOptimization)

After that we have to define our configuration search space with the hyperparameter that we want to optimize.

.. code:: python

    @a.searchspace
    def search_space():
        batch_size = UniformNumber(lower=8,
                                   upper=64,
                                   default=32,
                                   type=int,
                                   log_scale=True)
        num_units_first_layer = UniformNumber(lower=16,
                                              upper=1024,
                                              default=32,
                                              type=int,
                                              log_scale=True)
        num_units_second_layer = UniformNumber(lower=16,
                                               upper=1024,
                                               default=32,
                                               type=int,
                                               log_scale=True)
        dropout_first_layer = UniformFloat(lower=0,
                                           upper=.99,
                                           default=.2)
        dropout_second_layer = UniformFloat(lower=0,
                                            upper=.99,
                                            default=.2)
        learning_rate = UniformFloat(lower=10e-6,
                                     upper=10e-1,
                                     default=10e-2,
                                     log_scale=True)


Here, for each hyperparameter we define a prior distribution, default value, its type and if we want
to adapt it on a log scale or not.

We can use now Sacred's command line interface to get a new configuration

.. code:: bash

    python experiment.py with search_space

Labwatch will now pass all already completed configurations that are stored in the database
to the hyperparameter optimizer, let it suggest a new configuration and then run the experiment with this configuration.


Configuration Search Spaces
===========================


At this point is probably a good idea to talk a little bit more about configuration spaces.
In general we distinguish between:

    - *categorical* hyperparameters that can take only discrete choices (e.g. {'a', 'b', 'c'})
    - *numerical* hyperparameters that can have either integer or continuous values.

Furthermore, Labwatch also allows you to define prior distributions (Gaussian, Uniform) for your hyperparameters.
Some hyperparameter optimizers such as for instance random search can exploit this prior knowledge to
suggest better configurations from early on.
In the case that you do not have a prior about you hyperparameter, just use a uniform distribution which is
simply defined by an upper and lower bound.


Hyperparameter Optimizers
=========================


Labwatch offers a simple interface to a variety of state-of-the-art hyperparameter optimization methods.
Note that every optimizer has its own properties and might not work for all use cases.
The following list will give you a brief overview of the optimizer that can be used with labwatch and in which
setting they would work. For more details we refer to the corresponding papers:

    - **Random search** is probably the simplest hyperparameter optimization method. It just samples hyperparameter
      configurations from the prior. The nice thing with random search is that it works in all search
      spaces and is easy to parallelize.

    - **Bayesian optimization** fits a probabilistic model to capture the current believe of the objective function.
      To select a new configuration, it use an utility function that only depend on the
      probabilistic model to trade off exploration and exploitation. Here we use Gaussian process to model our objective
      function, which work well in low (<10) dimensional continuous input spaces but do not work with categorical
      hyperparameters.

    - **SMAC** is also a Bayesian optimization method but uses random forest instead of Gaussian processes to model
      the objective function. It works in high dimensional mixed continuous and discret input space but will be
      be probably outperformed by GP-based Bayesian optimization in the low dimensional continuous space.


Multiple Search Spaces
======================


Sometimes it is quite convenient to have multiple different search space, for instance if you want to optimize
first only a subset of your hyperparameters and keep the others fixed.

Labwatch allows to have different search space as long as they have different names. For instance in our running
example if we want to optimize only the learning rate and batch size we can define a second search space:


.. code:: python

    @a.searchspace
    def small_search_space():
        batch_size = UniformNumber(lower=32, upper=64, default=32, type=int, log_scale=True)
        learning_rate = UniformFloat(lower=10e-3, upper=10e-2, default=10e-2, log_scale=True)


If we now call our experiment via:


.. code:: bash

    python experiment.py with small_search_space

we get a new configuration for the learning rate and the batch where as all other hyperparameter are set to the values
defined in the config cfg().

Note: To prevent inconsistencies and to not fool the optimizer, Labwatch passes only completed configurations that were
drawn from this search space to the optimizer. This means that our optimizer will not use the information from previous
experiment with the other search space.
