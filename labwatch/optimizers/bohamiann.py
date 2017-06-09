#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np


from ConfigSpace import Configuration
try:
    from robo.initial_design.init_random_uniform import init_random_uniform
    from robo.models.bnn import BayesianNeuralNetwork
    from robo.maximizers.direct import Direct
    from robo.acquisition_functions.log_ei import LogEI
except:
    pass
from labwatch.optimizers.base import Optimizer
from labwatch.converters.convert_to_configspace import (
    sacred_space_to_configspace, configspace_config_to_sacred)


class Bohamiann(Optimizer):

    def __init__(self, config_space, burnin=3000, n_iters=10000):

        super(Bohamiann, self).__init__(sacred_space_to_configspace(config_space))
        self.rng = np.random.RandomState(np.random.seed())
        self.n_dims = len(self.config_space.get_hyperparameters())

        # All inputs are mapped to be in [0, 1]^D
        self.lower = np.zeros([self.n_dims])
        self.upper = np.ones([self.n_dims])
        self.incumbents = []
        self.X = None
        self.y = None

        self.model = BayesianNeuralNetwork(sampling_method="sghmc",
                                           l_rate=np.sqrt(1e-4),
                                           mdecay=0.05,
                                           burn_in=burnin,
                                           n_iters=n_iters,
                                           precondition=True,
                                           normalize_input=True,
                                           normalize_output=True)

        self.acquisition_func = LogEI(self.model)

        self.maximizer = Direct(self.acquisition_func, self.lower, self.upper, verbose=False)

    def suggest_configuration(self):

        if self.X is None and self.y is None:
            # No data points yet to train a model, just return a random configuration instead
            new_x = init_random_uniform(self.lower, self.upper,
                                        n_points=1, rng=self.rng)[0, :]

        else:
            # Train the model on all finished runs
            self.model.train(self.X, self.y)
            self.acquisition_func.update(self.model)

            # Maximize the acquisition function
            new_x = self.maximizer.maximize()

        # Maps from [0, 1]^D space back to original space
        next_config = Configuration(self.config_space, vector=new_x)

        # Transform to sacred configuration
        result = configspace_config_to_sacred(next_config)

        return result
