from __future__ import division, print_function, unicode_literals
import sys
import time
from datetime import datetime
from datetime import timedelta

import pymongo
import gridfs
from pymongo.son_manipulator import SONManipulator

import sacred.optional as opt
if not opt.has_pymongo:
    raise RuntimeError("pymongo not found but needed by LabAssistant")
from sacred.observers import MongoObserver
from sacred.observers.mongo import MongoDbOption
from sacred.utils import create_basic_stream_logger
from sacred.commandline_options import QueueOption
from sacred.arg_parser import parse_args

from labwatch.utils.types import InconsistentSpace
from labwatch.hyperparameters import decode_param_or_op
from labwatch.searchspace import SearchSpace, build_searchspace
from labwatch.optimizers import RandomSearch
from labwatch.commandline_options import AssistedOption


# SON Manipulators for saving and retrieving search spaces
SON_MANIPULATORS = []

class SearchSpaceManipulator(SONManipulator):

    def transform_incoming(self, son, collection):
        return son

    def transform_outgoing(self, son, collection):
        if "_class" in son.keys():
            if son["_class"] == "SearchSpace":
                return SearchSpace(son)
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
    We will record experiments using the following scheme:
    Each experiment has its own database. This database has
    a search_space document that stores the definition of the
    space TODO how is it defined ? :
    db.search_space - entries structured like {
       ...
    }

    db.runs - standard run entries as defined through sacred
    """

    def __init__(self,
                 database,
                 experiment,
                 optimizer=None,
                 prefix='default',
                 always_inject_observer=False):
        self.db = database
        self.ex = experiment
        self.ex.logger = create_basic_stream_logger()
        self.logger = self.ex.logger.getChild('LabAssistant')
        self.prefix = prefix
        self.version_policy = 'newer'
        self.always_inject_observer = always_inject_observer
        self.optimizer = optimizer
        if self.db is not None:
            self._init_db()
        else:
            self.runs = None
            self.db_searchspace = None
            self.optimizer = None
        # remember for which experiments we have config hooks setup
        self.have_hooks_for = set()
        self.observer_mapping = dict()
        # mark that we have newer looked for finished runs
        self.known_jobs = set()
        self.last_checked = None
        self.space_initialized = False
        self.search_space = None
        ##### Experiment modifications #####
        # first add a hook to the experiment
        self._add_hook(self.ex)
        # then inject an observer if it is required
        if self.db is not None and self.always_inject_observer:
            self._inject_observer(self.ex)

    def _init_db(self):
        self.runs = self.db[self.prefix].runs
        self.db_searchspace = self.db[self.prefix].search_space
        for manipulator in SON_MANIPULATORS:
            self.db.add_son_manipulator(manipulator)
        
    def _verify_and_init_searchspace(self, space_from_ex, new_space=False):
        # try to figure out if a searchspace is defined in the database
        if self.db is None:
            self.space_initialized = True
            self.search_space = space_from_ex
            self.optimizer = RandomSearch(self.search_space)
            return self.search_space
        space_from_db = self.db_searchspace.find_one()
        if space_from_db is not None:
            if space_from_ex is not None:
                # if this search space has no id we set it to
                # the one from the database before comparing
                if not space_from_ex.has_key("_id"):
                    space_from_ex["_id"] = space_from_db["_id"]
                if (space_from_db != space_from_ex):
                    raise InconsistentSpace("The search space of your experiment " \
                                            "is incompatible with the space " \
                                            "stored in the database! Use "\
                                            "new_space=True if it changed.")
        else:
            if space_from_ex is None:
                raise RuntimeError("You provided no search space and no " \
                                   "space is saved in the database!")
            sp_id = self.db_searchspace.insert(space_from_ex)
            space_from_db = self.db_searchspace.find_one({"_id" : sp_id})
        self.space_id = space_from_db["_id"]
        if self.optimizer:
            self.optimizer = self.optimizer.__class__(space_from_db)
        else:
            self.optimizer = RandomSearch(space_from_db)
        # update the optimizer 
        if self.optimizer.needs_updates():
            self.update_optimizer()
        self.space_initialized = True
        self.search_space = space_from_db
        return self.search_space

    def _clean_config(self, config):
        res = {}
        for key in config.keys():
            if isinstance(key, basestring) and len(key) > 0 and key[0] == '_':
                # ignore this key
                continue
            else:
                res[key] = config[key]
        return res
    
    def _config_hook(self, orig_cfg, command_name, logger, args):
        # ensure we have a search space definition
        if (not self.space_initialized) or (self.search_space is None):
            raise ValueError("LabAssistant config_hook called but " \
                             "there is no search space definition")
        # if there was no database passed to the assistant
        # try to fetch one from the args
        logger = logger.getChild('LabAssistant')
        if self.db is None:
            db_flag = "--" + MongoDbOption.get_flag()[1]
            if args.has_key(db_flag) and args[db_flag] is not None:
                logger.info("Using db provided via {} flag".format(db_flag))
                db_name = args[db_flag]
                c = pymongo.MongoClient()
                self.db = c[db_name]
                # init database
                self._init_db()
                # verify searchspace against database
                try:
                    self._verify_and_init_searchspace(self.search_space)
                except:
                    logger.warn("Tried to use db provided via flag but caught an exception")
                    self.db = None
                    self.runs = None
                    self.db_searchspace = None
                    self.optimizer = None
            else:
                logger.warn("have no database! Using random search!")        
        # check for assisted flag in args
        flag_string = "--" + AssistedOption.get_flag()[1]
        if args.has_key(flag_string):
            assist_command = args[flag_string]
        else:
            assist_command = False
        cfg = self.get_suggestion()
        result_cfg = {}
        if assist_command:
            for key in cfg.keys():
                if orig_cfg.has_key(key):
                    result_cfg[key] = cfg[key]
                else:
                    # check if the key starts with an underscore
                    # in which case it is safe for the default
                    # not to contain it, otherwise we warn the user
                    err = "config value for {}, which LabAssistant wants to fill "  \
                          "is not in the original config! " + \
                          "This happens because your default config " + \
                          "does not contain it"
                    if isinstance(key, basestring) and len(key) > 0 and key[0] != '_':
                        logger.warn(err.format(key))
                        result_cfg[key] = cfg[key] 
        return result_cfg

    def _add_hook(self, ex):
        def hook_closure(config, command_name, logger, args):
            return self._config_hook(config, command_name, logger, args)
        ex.config_hook(hook_closure)
        self.have_hooks_for.add(ex)
        
    def _inject_observer(self, ex):
        if self.db is None:
            raise ValueError("LabAssistant has no database " \
                             "but you called inject_observer")
        if self.observer_mapping.has_key(ex):
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
                self.logger.warn("Multiple MongoObservers with the same " \
                                 "collection name, make sure they are writing " \
                                 "into different databases!")
            collection_names.add(name)

    def _dequeue_run(self, remaining_time):
        criterion = {'status':'QUEUED'}
        ex_info = self.ex.get_experiment_info()
        run = None
        while remaining_time > 0.:
            run = self.runs.find_one(criterion)
            if run is None:
                self.logger.warn('Could not find run from queue ' \
                                 ' waiting for max another {} s'.format(remaining_time))
                time.sleep(sleep_time)
                expired_time = (time.time() - start_time)
                remaining_time = self.block_time - expired_time
            else:
                # verify the run
                _check_names(ex_info['name'], run['experiment']['name'])
                _check_sources(ex_info['sources'], run['experiment']['sources'])
                _check_dependencies(ex_info['dependencies'],
                                    run['experiment']['dependencies'],
                                    version_policy)
                
                # set status to INITIALIZING to prevent others from
                # running the same Run.
                old_status = run['status']
                run['status'] = 'INITIALIZING'
                replace_summary = self.runs.replace_one(
                    {'_id': run['_id'], 'status': old_status},
                    replacement=run)
                if replace_summary.modified_count == 1:
                    break  # we've successfully acquired a run
        return run
        
    ###################### exported functions ######################
    def update_optimizer(self):
        if self.db is None:
            self.logger.warn("cannot update optimizer, reason: no database!")
            return
        # First check database for all configurations
        if self.last_checked is None:
            # if we never checked the database we have to check
            # everything that happened since the definition of time ;)
            self.last_checked = datetime.min
        oldest_still_running = None
        running_jobs = self.runs.find(
            {'heartbeat': {'$gte': self.last_checked}}
        )
        for job in running_jobs:
            if job["_id"] not in self.known_jobs:
                # update optimizer with all finished results
                if job['status'] == 'COMPLETED':
                    self.optimizer.update(job["config"],
                                          job["final_outcome"])
                    # mark it as known
                    self.known_jobs.add(job["_id"])
                elif job["status"] == "RUNNING":
                    # TODO this is not correct
                    if oldest_still_running is None:
                        oldest_still_running = job["start_time"]
                    else:
                        oldest_still_running = min(job["start_time"], oldest_still_running)
        self.last_checked = oldest_still_running or datetime.now()

    def get_suggestion(self, clean=False):
        if (not self.space_initialized) or (self.search_space is None):
            raise ValueError("LabAssistant sample_suggestion called " \
                             "without a defined search space")
        config = self.optimizer.suggest_configuration()
        # clean underscored variables
        if clean:
           config = self._clean_config(config)
                        
        return config
                
    def run_suggestion(self, command='main'):
        if self.optimizer.needs_updates():
            self.update_optimizer()
        # Next get config from optimizer
        config = self.get_suggestion(clean=True)
        if config is None:
            raise RuntimeError("Optimizer did not return a config!")
        self._inject_observer(ex)
        res = self.ex.run_command(command, config_updates=config)
        return res

    def run_random(self, command='main'):
        config = self._clean_config(self.optimizer.get_random_config())
        self._inject_observer(ex)
        res = self.ex.run_command(command, config_updates=config)
        return res

    def run_default(self, command='main'):
        config = self._clean_config(self.optimizer.get_default_config())
        self._inject_observer(ex)
        res = self.ex.run_command(command, config_updates=config)
        return res

    def run_config(self, config, command='main'):
        config = self._clean_config(config)
        self._inject_observer(ex)
        res = self.ex.run_command(command, config_updates=config)
        return res

    def enqueue_suggestion(self, command='main'):
        if self.optimizer.needs_updates():
            self.update_optimizer()
        # Next get config from optimizer
        config = self.get_suggestion(clean=True)
        if config is None:
            raise RuntimeError("Optimizer did not return a config!")
        res = self.ex.run_command(command,
                                  config_updates=config,
                                  args={ "--queue" : QueueOption()})                                  
        
    
    def run_from_queue(self, wait_time_in_s=10 * 60, sleep_time=30):
        criterion = {'status':'QUEUED'}
        self._dequeue_run(wait_time_in_s)
        if run is None:
            self.logger.warn("No run found in queue for {} s -> terminating".format(wait_time_in_s))
            return None
        else:
            # remove MongoObserver if we have one for that experiment
            had_matching_observer = False
            if self.observer_mapping.has_key(self.ex):
                had_matching_observer = True
                matching = None
                for i,observer in self.ex.observers:
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
            res = self.ex.run_command(run['command'], config_updates=run['config'])
            
            # remove the extra observer
            self.ex.observers.pop()
            # and inject the default one
            if had_matching_observer:
                self._inject_observer(self.ex)
            return res
    
    ########## Decorators ##########

    def searchspace(self, function):
        """
        Decorator for creating a searchspace definition
        from a function.
        """
        space = build_searchspace(function)
        # validate the search space
        return self._verify_and_init_searchspace(space)
