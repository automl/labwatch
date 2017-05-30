#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import re

from sacred.config import ConfigScope
from sacred.utils import join_paths
from labwatch.hyperparameters import Parameter, ConditionResult
from labwatch.hyperparameters import decode_param_or_op
from labwatch.utils.types import InconsistentSpace, ParamValueExcept


class SearchSpace(object):

    def __init__(self, search_space):
        super(SearchSpace, self).__init__()
        # extract the _id from the searchspace definition
        self._id = search_space.get('_id', None)
        if '_id' in search_space:
            del search_space['_id']

        self.search_space = search_space
        parameters = collect_hyperparameters(search_space)
        params = sorted(parameters.values(), key=lambda x: x['uid'])
        self.uids_to_names = {param["uid"]: param['name'] for param in params}
        self.conditions = []
        self.non_conditions = []
        self.parameters = {}
        # first simply insert all
        for param in params:
            assert(isinstance(param, Parameter))
            self.parameters[param['name']] = param
            if isinstance(param, ConditionResult):
                self.conditions.append(param["name"])
            else:
                self.non_conditions.append(param["name"])

        self.contains_conditions = len(self.conditions) > 0
        self.validate_conditions()

    def to_json(self):
        son = dict(self.search_space)
        son['_class'] = 'SearchSpace'
        if self._id is not None:
            son['_id'] = self._id
        return son

    @classmethod
    def from_json(cls, son):
        assert son['_class'] == 'SearchSpace'
        del son['_class']
        return SearchSpace(son)

    def validate_conditions(self):
        for pname in self.conditions:
            cparam = self.parameters[pname]
            conditioned_on = cparam["condition"]["uid"]
            if not (conditioned_on in self.uids_to_names):
                err = "Conditional parameter: {} depends on: {} " \
                      "which does not exist in the defined SearchSpace!"
                raise InconsistentSpace(err.format(pname, conditioned_on))

    def is_valid_name(self, name):
        return name in self.uids_to_names.values()

    def valid(self, config):
        # TODO check this again for consistency
        valid = True
        for pname in self.non_conditions:
            valid &= self.parameters[pname].valid(config[pname])
        if not valid:
            return False
        for pname in self.conditions:
            cparam = self.parameters[pname]
            conditioned_on = cparam["condition"]["uid"]
            if cparam in config.keys():
                if conditioned_on in self.uids_to_names.keys():
                    valid &= conditioned_on.sample(config[self.uids_to_names[conditioned_on]])
                    valid &= cparam.valid(config[pname])
                else:
                    valid = False
        return valid

    def sample(self, max_iters_till_cycle=50, strategy="random"):
        if strategy not in ["random", "default"]:
            raise ParamValueExcept("Unknown sampling strategy {}".format(strategy))
        # allocate result dict
        res = {}
        # first add all fixed parameters
        # for pname in self.fixed:
        #    res[pname] = self[pname]
        # second sample all non conditions
        considered_params = set()
        for pname in self.non_conditions:
            if strategy == "random":
                res[pname] = self.parameters[pname].sample()
            else:
                res[pname] = self.parameters[pname].default()
            considered_params.add(pname)
        # then the conditional parameters
        remaining_params = set(self.conditions)
        i = 0
        while remaining_params:
            for pname in self.conditions:
                if pname in remaining_params:
                    cparam = self.parameters[pname]
                    conditioned_on = self.uids_to_names[cparam["condition"]["uid"]]
                    if conditioned_on in res.keys():
                        if strategy == "random":
                            cres = self.parameters[pname].sample(res[conditioned_on])
                        else:
                            cres = self.parameters[pname].default(res[conditioned_on])
                        if cres:
                            res[pname] = cres
                        considered_params.add(pname)
                        remaining_params.remove(pname)
                    elif conditioned_on in considered_params:
                        remaining_params.remove(pname)
                    else:
                        continue
            i += 1
            if i > max_iters_till_cycle:
                err = "Cannot satisfy conditionals involving " \
                      "parameters {} probably a loop! If you are sure " \
                      "no loop exists increase max_iters_till_cycle"
                raise InconsistentSpace(err.format(remaining_params))
        return res

    def default(self, max_iters_till_cycle=50):
        return self.sample(max_iters_till_cycle, strategy="default")

    def __eq__(self, other):
        if not isinstance(other, SearchSpace):
            return False
        else:
            return self.search_space == other.search_space


# decorator
def build_search_space(function):
    # abuse configscope to parse search space definitions
    scope = ConfigScope(function)
    space_dict = dict(scope())

    # parse generic dict to a search space
    space = SearchSpace(space_dict)
    return space


def set_name(hparam, name):
    if ('name' not in hparam or
            len(hparam['name']) > len(name) or
            hparam['name'] > name):
        hparam['name'] = name


def merge_parameters(params, new_params):
    for k, v in new_params.items():
        if k not in params:
            params[k] = v
        else:
            set_name(params[k], v['name'])
    return params


def collect_hyperparameters(search_space, path=''):
    """
    Recursively collect all the hyperparameters from a search space.

    Parameters
    ----------
    search_space : dict
        A JSON-like structure that describes the search space.
    path : str
        The path to the current entry. Used to determine the name of the
        detected hyperparameters. Optional: Only used for the recursion.

    Returns
    -------
    parameters : dict
        A dictionary that to all the collected hyperparameters from their uids.
    """
    # First try to decode to hyperparameter
    if isinstance(search_space, dict):
        try:
            hparam = decode_param_or_op(search_space)
            set_name(hparam, path)
            return {hparam['uid']: hparam}
        except ValueError:
            pass

    parameters = {}
    # if the space is a dict (but not a hyperparameter) we parse it recursively
    if isinstance(search_space, dict):
        for k, v in search_space.items():
            # add the current key and a '.' as prefix when recursing
            sub_params = collect_hyperparameters(v, join_paths(path, k))
            parameters = merge_parameters(parameters, sub_params)
        return parameters

    # if the space is a list we iterate it recursively
    elif isinstance(search_space, (tuple, list)):
        for i, v in enumerate(search_space):
            # add '[N]' to the name when recursing
            sub_params = collect_hyperparameters(v, path + '[{}]'.format(i))
            parameters = merge_parameters(parameters, sub_params)
        return parameters
    else:
        # if the space is anything else do nothing
        return parameters


def fill_in_values(search_space, values, fill_by='uid'):
    """
    Recursively insert given values into a search space to receive a config.

    Parameters
    ----------
    search_space : dict
        A JSON-like structure that describes the search space.
    values : dict
        A dictionary mapping uids to values.

    Returns
    -------
    dict
        A configuration that results from replacing hyperparameters by the
        corresponding values.

    """
    if isinstance(search_space, dict):
        if '_class' in search_space and fill_by in search_space:
            return values[search_space[fill_by]]
        else:
            return {k: fill_in_values(v, values, fill_by)
                    for k, v in search_space.items()}
    elif isinstance(search_space, (list, tuple)):
        config = [fill_in_values(v, values, fill_by) for v in search_space]
        return type(search_space)(config)
    else:
        return search_space


def get_by_path(config, path):
    """
    Get a config-entry by its dotted and indexed name.

    Parameters
    ----------
    config : dict
        The configuration dictionary to get the values from.
    path : str
        The config entry corresponding to the given path.
    Returns
    -------
    object
        The configuration entry that corresponds to the given path.

    """
    current = config
    for p in filter(None, re.split('[.\[\]]', path)):
        try:
            p = int(p)
        except ValueError:
            pass
        current = current[p]
    return current


def get_values_from_config(config, hyperparams):
    """
    Infer the values of hyperparameters from a given configuration.
    Parameters
    ----------
    config : dict
        A JSON configuration that has to correspond to the search space.
    hyperparams : dict
        A dictionary that maps uids to hyperparameters.

    Returns
    -------
    dict
        A dictionary mapping names to values.
    """
    return {hparam['name']: get_by_path(config, hparam['name'])
            for uid, hparam in hyperparams.items()}
