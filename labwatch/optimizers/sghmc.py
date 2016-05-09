#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np

import theano
import theano.tensor as T
from functools import partial

from ConfigSpace import Configuration

from robo.initial_design.init_random_uniform import init_random_uniform

from labwatch.optimizers.base import Optimizer
from labwatch.converters.convert_to_configspace import (
    sacred_space_to_configspace, sacred_config_to_configspace,
    configspace_config_to_sacred)


import lasagne
from lasagne.layers import InputLayer, DenseLayer

from hmc_bnn.normalized import weight_norm
from hmc_bnn.bnn import *
from hmc_bnn.acquisition import EI as bnn_ei


class SGHMC(Optimizer):

    def __init__(self, config_space, burnin=1000, chain_length=200,
                 n_hypers=20):

        super(SGHMC, self).__init__(config_space)
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
        print("suggest_config")
        if self.X is None and self.Y is None:
            new_x = init_random_uniform(self.X_lower, self.X_upper,
                                        N=1, rng=self.rng)

        elif self.X.shape[0] == 1:
            # We need at least 2 data points to train a GP
            Xopt = init_random_uniform(self.X_lower, self.X_upper,
                                        N=1, rng=self.rng)

        else:
            my = np.mean(self.Y)
            stdy = np.std(self.Y)
            Y_norm = (self.Y - my) / stdy

            mx = np.mean(self.X)
            stdx = np.std(self.X)
            
            model = self.get_model(n_inputs=self.X.shape[1])            
            model.train(self.X, Y_norm, 9000, init_eta=1e-4, mdecay=0.1)

            #"""
            acquisition = bnn_ei(model)
            inc_idx_n = np.argmin(Y_norm)
            # a bunch of random points
            Xinit_rand = init_random_uniform(self.X_lower, self.X_upper, 512)
            # a bunch of pertubations around the incumbent
            Xinc_rand = np.repeat(np.atleast_2d(self.X[inc_idx_n]), 20, axis=0) \
                            + np.random.randn(20, self.X.shape[1]) * 0.1


            Xinit = floatX(np.concatenate([Xinc_rand, Xinit_rand], axis=0))
            Xres,Acq_opt,best = acquisition.optimize(Xinit,
                                            Y_norm[inc_idx_n].reshape(-1, 1),
                                            zip(self.X_lower, self.X_upper), 500)        
            Xopt = Xres[best].reshape(1, -1)

            # update the incumbent array
            inc_idx = np.argmin(self.Y)
            self.incumbents.append(self.X[inc_idx, None, :])


        # Map from [0, 1]^D space back to original space
        next_config = Configuration(self.config_space, vector=Xopt[0, :])

        # Transform to sacred configuration
        result = configspace_config_to_sacred(next_config)

        return result
        
    def target(self, x):
        return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)
    
    def get_net(self, n_inputs):
        l_in = InputLayer(shape=(None, n_inputs))
    
        l_hid1 = DenseLayer(
            l_in, num_units=100,
            W = lasagne.init.HeUniform(),
            b = lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.tanh)
        l_hid2 = DenseLayer(
            l_hid1, num_units=100,
            W = lasagne.init.HeUniform(),
            b = lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.tanh
        )
        """
        l_hid3 = DenseLayer(
            l_hid2, num_units=100,
            W = lasagne.init.HeNormal(),
            b = lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.tanh
        )
        #"""
        l_out = DenseLayer(
            l_hid2, num_units=2,
            W = lasagne.init.HeUniform(),
            b = lasagne.init.Constant(0.),
            nonlinearity=None)
        #l_out = weight_norm(l_out)
        return l_out
    
    def get_model(self, burn_in=4000, n_inputs=2):
        get_net_ = partial(self.get_net, n_inputs=n_inputs)
        variance_prior = LogVariancePrior(1e-4, prior_out_std_prec=0.001)
        weight_prior = WeightPrior(alpha=1., beta=1.)
        model = HMCBNN(get_net_,
                       updater=NoiseScaledAdaptSGHMCUpdater(),
                       weight_prior=weight_prior,
                       variance_prior=variance_prior,
                       burn_in=burn_in, capture_every=50,
                       n_target_nets=50)
        return model        

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
