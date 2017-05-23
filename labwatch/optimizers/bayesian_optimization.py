#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np
import george

from ConfigSpace import Configuration

from robo.priors.default_priors import DefaultPrior
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.maximizers.direct import Direct
from robo.acquisition_functions.log_ei import LogEI
from robo.acquisition_functions.marginalization import MarginalizationGPMCMC
from robo.initial_design.init_random_uniform import init_random_uniform

from labwatch.optimizers.base import Optimizer
from labwatch.converters.convert_to_configspace import (
    sacred_space_to_configspace, sacred_config_to_configspace,
    configspace_config_to_sacred)


class BayesianOptimization(Optimizer):

    def __init__(self, config_space, burnin=100, chain_length=200,
                 n_hypers=20):

        super(BayesianOptimization, self).__init__(config_space)
        self.rng = np.random.RandomState(np.random.seed())

        self.burnin = burnin
        self.chain_length = chain_length
        self.n_hypers = n_hypers

        self.config_space = sacred_space_to_configspace(config_space)

        n_inputs = len(self.config_space.get_hyperparameters())

        self.lower = np.zeros([n_inputs])
        self.upper = np.ones([n_inputs])

        self.X = None
        self.y = None

    def suggest_configuration(self):
        if self.X is None and self.y is None:
            new_x = init_random_uniform(self.lower, self.upper,
                                        n_points=1, rng=self.rng)[0, :]

        elif self.X.shape[0] == 1:
            # We need at least 2 data points to train a GP
            new_x = init_random_uniform(self.lower, self.upper,
                                        n_points=1, rng=self.rng)[0, :]

        else:
            cov_amp = 1
            n_dims = self.lower.shape[0]

            initial_ls = np.ones([n_dims])
            exp_kernel = george.kernels.Matern52Kernel(initial_ls,
                                                       ndim=n_dims)
            kernel = cov_amp * exp_kernel

            prior = DefaultPrior(len(kernel) + 1)

            model = GaussianProcessMCMC(kernel, prior=prior,
                                        n_hypers=self.n_hypers,
                                        chain_length=self.chain_length,
                                        burnin_steps=self.burnin,
                                        normalize_input=False,
                                        normalize_output=True,
                                        rng=self.rng,
                                        lower=self.lower,
                                        upper=self.upper)

            a = LogEI(model)

            acquisition_func = MarginalizationGPMCMC(a)

            max_func = Direct(acquisition_func, self.lower, self.upper, verbose=False)

            model.train(self.X, self.y)

            acquisition_func.update(model)

            new_x = max_func.maximize()

        next_config = Configuration(self.config_space, vector=new_x)

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

            if self.X is None and self.y is None:
                self.X = np.array([x])
                self.y = np.array([cost])
            elif x not in self.X:
                self.X = np.append(self.X, x[np.newaxis, :], axis=0)
                self.y = np.append(self.y, np.array([cost]), axis=0)

    def needs_updates(self):
        return True
