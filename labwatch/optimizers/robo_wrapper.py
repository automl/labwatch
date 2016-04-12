#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np
import george

from ConfigSpace import Configuration

from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.acquisition.ei import EI
from robo.acquisition.integrated_acquisition import IntegratedAcquisition
from robo.maximizers.direct import Direct
from robo.priors.default_priors import DefaultPrior
from robo.initial_design.init_random_uniform import init_random_uniform

from labwatch.optimizers.base import Optimizer
from labwatch.converters.convert_to_configspace import (
    sacred_space_to_configspace, sacred_config_to_configspace,
    configspace_config_to_sacred)


class RoBO(Optimizer):

    def __init__(self, config_space, burnin=1000, chain_length=200,
                 n_hypers=20):

        super(RoBO, self).__init__(config_space)
        self.rng = np.random.RandomState(np.random.seed())

        self.burnin = burnin
        self.chain_length = chain_length
        self.n_hypers = n_hypers

        self.config_space = sacred_space_to_configspace(config_space)

        n_hypers = len(self.config_space.get_hyperparameters())

        # All hyperparameter are mapped to be in [0, 1]^D
        self.X_lower = np.zeros([n_hypers])
        self.X_upper = np.ones([n_hypers])

        self.X = None
        self.Y = None

    def suggest_configuration(self):
        if self.X is None and self.Y is None:
            new_x = init_random_uniform(self.X_lower, self.X_upper,
                                        N=1, rng=self.rng)

        elif self.X.shape[0] == 1:
            # We need at least 2 data points to train a GP
            new_x = init_random_uniform(self.X_lower, self.X_upper,
                                        N=1, rng=self.rng)

        else:
            cov_amp = 1
            n_dims = self.X_lower.shape[0]
            config_kernel = george.kernels.Matern52Kernel(
                np.ones([n_dims]) * 0.01, ndim=n_dims)
            noise_kernel = george.kernels.WhiteKernel(1e-9, ndim=n_dims)
            kernel = cov_amp * config_kernel + noise_kernel
            prior = DefaultPrior(len(kernel))

            model = GaussianProcessMCMC(kernel, prior=prior,
                                        burnin=self.burnin,
                                        chain_length=self.chain_length,
                                        n_hypers=self.n_hypers)

            acq = EI(model, self.X_lower, self.X_upper)

            acquisition_func = IntegratedAcquisition(
                model, acq, self.X_lower, self.X_upper)

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
