#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import datetime
import time
import numbers
import functools
import gridfs
import pymongo

import sacred.optional as opt

from sacred.commandline_options import QueueOption
from sacred.observers.mongo import MongoObserver, MongoDbOption
from sacred.utils import create_basic_stream_logger

from labwatch.optimizers.random_search import RandomSearch
from labwatch.searchspace import SearchSpace, build_searchspace, fill_in_values, \
    get_values_from_config
from labwatch.utils.types import InconsistentSpace

from labwatch.utils.version_checks import (check_dependencies, check_sources,
                                           check_names)

if not opt.has_pymongo:
    raise RuntimeError("pymongo not found but needed by LabAssistant")

# SON Manipulators for saving and retrieving search spaces
SON_MANIPULATORS = []


class SearchSpaceManipulator(pymongo.son_manipulator.SONManipulator):

    def transform_incoming(self, son, collection):
        return son

    def transform_outgoing(self, son, collection):
        if "_class" in son.keys():
            if son["_class"] == "SearchSpace":
                return SearchSpace.from_json(son)
        else:
            for (key, value) in son.items():
                if isinstance(value, dict):
                    if "_class" in value:
                        if value["_class"] == "SearchSpace":
                            son[key] = SearchSpace(value)
                    else:  # Again, make sure to recurse into sub-docs
                        son[key] = self.transform_outgoing(value, collection)
        return son

SON_MANIPULATORS.append(SearchSpaceManipulator())


class LabAssistant(object):

    """
    The main class for Labwatch. It runs an experiment with a configuration suggested by
    and hyperparameter optimizer.

    The hyperparameter optimizer uses the information about the experiment that are stored in the
    database to suggest a new configuration.
    """

    def __init__(self,
                 experiment,
                 database_name=None,
                 url="localhost",
                 optimizer=None,
                 prefix='default',
                 always_inject_observer=False):

        """
        Create a new LabAssistant and connects it with a database.

        Parameters
        ----------
        experiment : object
            The (sacred) experiment that is going to be optimized.
        database_name : str
            The name of the database where all information about the dataset are saved.
        optimizer: object, optional
            Specifies which optimizer is used to suggest a new hyperparameter configuration
        prefix: str, optional
            Additional prefix for the database
        always_inject_observer: bool, optional
            If true an MongoObserver is added to the experiment.
        """

        self.ex = experiment

        self.db_name = database_name
        self.url = url
        self.db = None

        self.ex.logger = create_basic_stream_logger()
        self.logger = self.ex.logger.getChild('LabAssistant')
        self.prefix = prefix
        self.version_policy = 'newer'
        self.always_inject_observer = always_inject_observer
        self.optimizer_class = optimizer
        self.block_time = 1000  # TODO: what value should this be?
        # if self.db is not None:
        #     self._init_db()
        # else:
        #     self.runs = None
        #     self.db_searchspace = None
        #     self.optimizer = None
        # remember for which experiments we have config hooks setup
        self.observer_mapping = dict()
        # mark that we have newer looked for finished runs
        self.known_jobs = set()
        self.last_checked = None
        self.search_space = None
        # then inject an observer if it is required
        #if self.db is not None and self.always_inject_observer:
        #self._inject_observer(self.ex)

    def _init_db(self):
        client = pymongo.MongoClient(self.url)

        self.db = client[self.db_name]
        self.runs = self.db[self.prefix].runs
        self.db_searchspace = self.db[self.prefix].search_space
        for manipulator in SON_MANIPULATORS:
            self.db.add_son_manipulator(manipulator)

    def _parse_db_from_args(self, args, logger):
        db_flag = "--" + MongoDbOption.get_flag()[1]
        if args.get(db_flag) is not None:
            logger.info("Using db provided via {} flag".format(db_flag))
            db_name = args[db_flag]
            c = pymongo.MongoClient()
            self.db = c[db_name]
            # init database
            self._init_db()
            # verify searchspace against database
            try:
                self._verify_and_init_searchspace(self.search_space)
                return True
            except:
                logger.warn("Tried to use db provided via args but caught an "
                            "exception", exc_info=True)
                self.db = None
                self.runs = None
                self.db_searchspace = None
                self.optimizer = None
        return False

    def _verify_and_init_searchspace(self, space_from_ex):
        # Get searchspace from the database or from the experiment

        # Check if search space is already in the database
        # (Note: We don't have any id yet that's why we have to loop over all entries)
        in_db = False
        if self.db_searchspace.count() > 0:
            for sp in self.db_searchspace.find():
                if sp == space_from_ex:
                    self.search_space = sp
                    in_db = True
        if not in_db:
            sp_id = self.db_searchspace.insert(space_from_ex.to_json())
            self.search_space = self.db_searchspace.find_one({"_id": sp_id})

        return self.search_space

    def _clean_config(self, config):
        values = get_values_from_config(config, self.search_space.parameters)
        return values

    def _search_space_wrapper(self, space, space_name, fixed=None, fallback=None, preset=None):
        # This function pretends to be a ConfigScope for a named_config
        # but under the hood it is getting a suggestion from the optimizer

        self.search_space_name = space_name
        sp = build_searchspace(space)

        # Establish connection to database
        if self.db is None:
            self._init_db()

        # TODO: Get MongoObserver from experiment
        self._inject_observer(self.ex)

        # Check the validity of this search space
        self._verify_and_init_searchspace(sp)

        # Create the optimizer
        if self.optimizer_class is not None:
            if not self.db:
                import warnings
                warnings.warn('No database. Falling back to random search')
                self.optimizer = RandomSearch(self.search_space)
            self.optimizer = self.optimizer_class(self.search_space)
        else:
            self.optimizer = RandomSearch(self.search_space)

        fixed = fixed or {}
        final_config = dict(preset or {})
        # the fallback parameter is needed to fit the interface of a
        # ConfigScope, but here it is not supported.
        assert not fallback, "{}".format(fallback)
        # ensure we have a search space definition
        if self.search_space is None:
            raise ValueError("LabAssistant search_space_wrapper called but "
                             "there is no search space definition")

        # Get a hyperparameter configuration from the optimizer
        values = self.get_suggestion()

        # Create configuration object
        config = fill_in_values(self.search_space.search_space, values,
                                fill_by='uid')
        final_config.update(config)
        final_config.update(fixed)

        return final_config

    def _inject_observer(self, ex):
        if self.db is None:
            raise ValueError("LabAssistant has no database "
                             "but you called inject_observer")
        if ex in self.observer_mapping:
            # we already have an observer for this experiment
            return
        # otherwise create a new one
        fs = gridfs.GridFS(self.db, collection=self.prefix)
        observer = MongoObserver(self.runs, fs)
        ex.observers.append(observer)
        self.observer_mapping[ex] = observer
        # warn if we have two observers writing to the same database
        collection_names = set()
        for observer in ex.observers:
            name = observer.runs.name
            if name in collection_names:
                self.logger.warn("Multiple MongoObservers with the same "
                                 "collection name, make sure they are writing "
                                 "into different databases!")
            collection_names.add(name)

    def _dequeue_run(self, remaining_time, sleep_time):
        criterion = {'status': 'QUEUED'}
        ex_info = self.ex.get_experiment_info()
        run = None
        start_time = time.time()
        while remaining_time > 0.:
            run = self.runs.find_one(criterion)
            if run is None:
                self.logger.warn('Could not find run from queue waiting for '
                                 'max another {} s'.format(remaining_time))
                time.sleep(sleep_time)
                expired_time = (time.time() - start_time)
                remaining_time = self.block_time - expired_time
            else:
                # verify the run
                check_names(ex_info['name'], run['experiment']['name'])
                check_sources(ex_info['sources'], run['experiment']['sources'])
                check_dependencies(ex_info['dependencies'],
                                   run['experiment']['dependencies'],
                                   self.version_policy)
                
                # set status to INITIALIZING to prevent others from
                # running the same Run.
                old_status = run['status']
                run['status'] = 'INITIALIZING'
                replace_summary = self.runs.replace_one(
                    {'_id': run['_id'], 'status': old_status},
                    replacement=run)
                if replace_summary.modified_count == 1 or \
                   replace_summary.raw_result['updatedExisting']:
                    # the second part above is necessary in case we are
                    # working with an older mongodb server (version < 2.6)
                    # which will not return the modified_count flag
                    break  # we've successfully acquired a run
        return run
        
    # ########################## exported functions ###########################

    def set_database(self, database):
        self.db = database
        self._init_db()
        # we need to verify the searchspace again
        self._verify_and_init_searchspace(self.search_space)
    
    def update_optimizer(self):
        if self.db is None:
            self.logger.warn("Cannot update optimizer, reason: no database!")
            return
        # First check database for all configurations
        if self.last_checked is None:
            # if we never checked the database we have to check
            # everything that happened since the definition of time ;)
            self.last_checked = datetime.datetime.min
        # oldest_still_running = None
        # running_jobs = self.runs.find(
        #     {
        #         'heartbeat': {'$gte': self.last_checked},
        #         'status': 'RUNNING'
        #     },
        #     sort=[("start_time", 1)]
        # )
        #

        # Take all jobs that are finished and were run with a config from this searchspace
        completed_jobs = self.runs.find(
            {
                'heartbeat': {'$gte': self.last_checked},
                'status': 'COMPLETED',
                'meta.options.UPDATE': self.search_space_name
            }
        )
        # update the last checked to the oldest one that is still running
        self.last_checked = datetime.datetime.now()
        # collect all configs and their results
        info = [(self._clean_config(job["config"]), convert_result(job["result"]), job)
                for job in completed_jobs if job["_id"] not in self.known_jobs]        
        if len(info) > 0:
            configs, results, jobs = (list(x) for x in zip(*info))
            self.known_jobs |= {job['_id'] for job in jobs}
            modifications = self.optimizer.update(configs, results, jobs)
            # the optimizer might modify the additional info of jobs
            if modifications is not None:
                for job in modifications:
                    new_info = job.info
                    self.runs.update_one(
                        {'_id': job["_id"]},
                        {'$set': {'info': new_info}},
                        upsert=False)

    def get_suggestion(self):
        if self.search_space is None:
            raise ValueError("LabAssistant sample_suggestion called "
                             "without a defined search space")
        #if self.optimizer.needs_updates():
        self.update_optimizer()

        suggestion = self.optimizer.suggest_configuration()
        values = {self.search_space.parameters[k]['uid']: v for k, v in suggestion.items() if k in self.search_space.parameters}
        return values

    def get_current_best(self, return_job_info=False):
        if self.db is None:
            self.logger.warn("cannot update optimizer, reason: no database!")
            return
        # ("status", 1) sorts according to status in ascending order
        best_job = self.runs.find_one({'status': 'COMPLETED'},
                                      sort=[("result", 1)])
        if best_job is None:
            best_result = None
            best_config = None
        else:
            best_result = best_job["result"]
            best_config = self._clean_config(best_job["config"])
        if return_job_info:
            return best_config, best_result, best_job
        else:
            return best_config, best_result

    def run_suggestion(self, command=None):
        # get config from optimizer
        #return self.run_config(self.get_suggestion(), command)
        values = self.get_suggestion()
        config = fill_in_values(self.search_space.search_space, values, fill_by='uid')

        return self.run_config(config, command)

    def run_random(self, command=None):
        return self.run_config(self.optimizer.get_random_config(), command)

    def run_default(self, command=None):
        return self.run_config(self.optimizer.get_default_config(), command)

    def run_config(self, config, command=None):
        if config is None:
            raise RuntimeError("None is not an acceptable config!")
        #config = self._clean_config(config)
        self._inject_observer(self.ex)
        if command is None:
            res = self.ex.run(config_updates=config)
        else:
            res = self.ex.run_command(command, config_updates=config)
        return res

    def enqueue_suggestion(self, command='main'):
        # Next get config from optimizer
        config = self._clean_config(self.get_suggestion())
        if config is None:
            raise RuntimeError("Optimizer did not return a config!")
        self._inject_observer(self.ex)
        res = self.ex.run_command(command,
                                  config_updates=config,
                                  args={"--queue": QueueOption()})

    def run_from_queue(self, wait_time_in_s=10 * 60, sleep_time=5):
        run = self._dequeue_run(wait_time_in_s, sleep_time)
        if run is None:
            self.logger.warn("No run found in queue for {} s -> terminating"
                             .format(wait_time_in_s))
            return None
        else:
            # remove MongoObserver if we have one for that experiment
            had_matching_observer = False
            if self.ex in self.observer_mapping:
                had_matching_observer = True
                matching = None
                for i, observer in enumerate(self.ex.observers):
                    if observer == self.observer_mapping[self.ex]:
                        matching = i
                if matching is None:
                    self.logger.warn("Could not remove observer in run_from_queue")
                    pass
                else:
                    del self.ex.observers[matching]
                    del self.observer_mapping[self.ex]
            # add a matching MongoObserver to the experiment and tell it to
            # overwrite the run
            fs = gridfs.GridFS(self.db, collection=self.prefix)
            self.ex.observers.append(MongoObserver(self.runs, fs,
                                                   overwrite=run))

            # run the experiment
            res = self.ex.run_command(run['command'],
                                      config_updates=run['config'])

            # remove the extra observer
            self.ex.observers.pop()
            # and inject the default one
            if had_matching_observer:
                self._inject_observer(self.ex)
            return res

    # ############################## Decorators ###############################

    def searchspace(self, function):
        """Decorator for creating a searchspace definition from a function."""
        #if self.search_space is not None:
        #    raise RuntimeError('Only one searchspace allowed per Assistant')

        # space = build_searchspace(function)
        #
        # #TODO: Get MongoObserver from experiment
        #
        # # Establish connection to database
        # self._init_db()
        #
        # # Check the validity of this search space
        # self._verify_and_init_searchspace(space)

        # Get a configuration from the optimizer and add it as a named config
        searchspace_wrapper = functools.partial(self._search_space_wrapper,
                                                space=function,
                                                space_name=function.__name__)
        self.ex._add_named_config(function.__name__, searchspace_wrapper)


def convert_result(result):
    if isinstance(result, dict):
        if "optimization_target" not in result:
            raise ValueError("The result of your experiment is a dict "
                             "without the key optimization_target, which "
                             "is required by labwatch")
        else:
            assert isinstance(result["optimization_target"], numbers.Number)
            return result["optimization_target"]
    elif not isinstance(result, numbers.Number):
        raise ValueError("The result of your experiment is a {} "
                         "but labwatch expects either a number "
                         "or a dict".format(type(result)))
    else:
        return result
