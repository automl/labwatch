#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np

from ConfigSpace import Configuration
try:
    from robo.initial_design.init_random_uniform import init_random_uniform
    from robo.acquisition.log_ei import LogEI
    from robo.acquisition.integrated_acquisition import IntegratedAcquisition
    from robo.maximizers.direct import Direct
    from robo.priors.dngo_priors import DNGOPrior
    from robo.models.dngo import DNGO
except:
    pass
from labwatch.optimizers.base import Optimizer
from labwatch.converters.convert_to_configspace import (
    sacred_space_to_configspace, sacred_config_to_configspace,
    configspace_config_to_sacred)


class DNGOWrapper(Optimizer):

    def __init__(self, config_space, burnin=1000, chain_length=200,
                 n_hypers=20):

        super(DNGOWrapper, self).__init__(config_space)
        self.rng = np.random.RandomState(np.random.seed())
        self.config_space = sacred_space_to_configspace(config_space)
        self.n_dims = len(self.config_space.get_hyperparameters())

        # All inputs are mapped to be in [0, 1]^D
        self.X_lower = np.zeros([self.n_dims])
        self.X_upper = np.ones([self.n_dims])
        self.incumbents = []
        self.X = None
        self.Y = None


    def suggest_configuration(self):
        if self.X is None and self.Y is None:
            new_x = init_random_uniform(self.X_lower, self.X_upper,
                                        N=1, rng=self.rng)

        elif self.X.shape[0] == 1:
            # We need at least 2 data points to train a GP
            Xopt = init_random_uniform(self.X_lower, self.X_upper,
                                        N=1, rng=self.rng)

        else:
            prior = DNGOPrior()
            model = DNGO(batch_size=100, num_epochs=20000,
                         learning_rate=0.1, momentum=0.9,
                         l2=1e-16, adapt_epoch=5000,
                         n_hypers=20, prior=prior,
                         do_optimize=True, do_mcmc=True)
                 

            #acquisition_func = EI(model, task.X_lower, task.X_upper)
            lo = np.ones([model.n_units_3]) * -1
            up = np.ones([model.n_units_3])
            ei = LogEI(model, lo, up)

            acquisition_func = IntegratedAcquisition(
                model, ei, self.X_lower, self.X_upper)

            maximizer = Direct(acquisition_func, self.X_lower, self.X_upper)

            model.train(self.X, self.Y)

            acquisition_func.update(model)

            new_x = maximizer.maximize()


        # Map from [0, 1]^D space back to original space
        next_config = Configuration(self.config_space, vector=new_x[0, :])

        # Transform to sacred configuration
        result = configspace_config_to_sacred(next_config)

        return result


    def update(self, configs, costs, runs):
        converted_configs = [
            sacred_config_to_configspace(self.config_space, config)
            for config in configs]

        for (config, cost) in zip(converted_configs, costs):
            # Maps configuration to [0, 1]^D space
            x = config.get_array()

            if self.X is None and self.Y is None:
                self.X = np.array([x])
                self.Y = np.array([[cost]])
            else:
                self.X = np.append(self.X, x[np.newaxis, :], axis=0)
                self.Y = np.append(self.Y, np.array([[cost]]), axis=0)

    def needs_updates(self):
        return True
