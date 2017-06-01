#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import numpy as np

from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.facade import smac_facade

from labwatch.optimizers.base import Optimizer
from labwatch.converters.convert_to_configspace import (
    sacred_space_to_configspace, sacred_config_to_configspace,
    configspace_config_to_sacred)


class LabwatchScenario(Scenario):
    """
    Specialize the smac3 scenario here since we want to create
    everything within code without reading a smac scenario file.
    """

    def __init__(self, config_space, logger):
        self.logger = logger
        # we don't actually have a target algorithm here
        # we will implement algorithm calling and the SMBO loop ourselves
        self.ta = None
        self.execdir = None
        self.pcs_fn = None
        self.run_obj = 'quality'
        self.overall_obj = self.run_obj

        # Time limits for smac
        # these will never be used since we call
        # smac.choose_next() manually
        self.cutoff = None
        self.algo_runs_timelimit = None
        self.wallclock_limit = None

        # no instances
        self.train_inst_fn = None
        self.test_inst_fn = None
        self.feature_fn = None
        self.train_insts = []
        self.test_inst = []
        self.feature_dict = {}
        self.feature_array = None
        self.instance_specific = None
        self.n_features = 0

        # save reference to config_space
        self.cs = config_space
        
        # We do not need a TAE Runner as this is done by the Sacred Experiment
        self.tae_runner = None
        self.deterministic = False


class SMAC(Optimizer):
    def __init__(self, config_space, seed=None):

        if seed is None:
            self.seed = np.random.randint(0, 10000)
        else:
            self.seed = seed

        self.rng = np.random.RandomState(self.seed)

        super(SMAC, self).__init__(sacred_space_to_configspace(config_space))

        self.scenario = Scenario({"run_obj": "quality",
                                  "cs": self.config_space,
                                  "deterministic": "true"})
        self.solver = smac_facade.SMAC(scenario=self.scenario,
                                       rng=self.rng)

    def suggest_configuration(self):
        if self.X is None and self.y is None:
            next_config = self.config_space.sample_configuration()

        else:
            l = list(self.solver.solver.choose_next(self.X, self.y[:, None], incumbent_value=np.min(self.y)))
            next_config = l[0]

        result = configspace_config_to_sacred(next_config)

        return result

    def needs_updates(self):
        return True
