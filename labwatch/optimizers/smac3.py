import logging
import numpy as np

from smac.smbo.smbo import SMBO
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM

from labwatch.converters.convert_to_configspace import sacred_space_to_configspace
from labwatch.converters.convert_to_configspace import sacred_config_to_configspace
from labwatch.converters.convert_to_configspace import configspace_config_to_sacred


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
        self.run_obj = 'QUALITY'
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

        # save reference to config_space
        self.cs = config_space


class SMAC3(object):

    def __init__(self, config_space):

        self.sacred_space = config_space
        self.config_space = sacred_space_to_configspace(config_space)
        self.scenario = LabwatchScenario(self.config_space, None)
        self.run_history = RunHistory()
        self.smac = SMBO(self.scenario, np.random.get_state())
        self.num_params = len(self.config_space.get_hyperparameters())
        self.rh2EPM = RunHistory2EPM(num_params=self.num_params,
                                     cutoff_time=1e7,
                                     success_states=None,
                                     impute_censored_data=False,
                                     impute_state=None)

    def suggest_configuration(self):
        if self.run_history.empty():
            # if there is nothing to use for SMAC
            # just sample from the sacred search space.
            # Alternatively we could evaluate the default
            # but that might be dangerous if we run multiple
            # workers in parallel
            return self.sacred_space.sample()
        X_cfg, Y_cfg = self.rh2EPM.transform(self.run_history)

        next_config = self.smac.choose_next(X_cfg, Y_cfg)

        result = configspace_config_to_sacred(next_config)

        return result

    def update(self, config, cost, run_info):

        # JTS TODO: also fetch duration, status and seed
        # JTS TODO: ponder cost vs performance
        duration = 1
        status = StatusType.SUCCESS
        seed = 1
        converted_config = sacred_config_to_configspace(self.config_space,
                                                        config)
        self.run_history.add(config=converted_config, cost=cost,
                             time=duration, status=status,
                             instance_id=0, seed=seed)

    def needs_updates(self):
        return True
