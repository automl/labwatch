#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from sacred.config import ConfigScope

from labwatch.utils import FixedDict
from labwatch.hyperparameters import Parameter, ConditionResult
from labwatch.hyperparameters import decode_param_or_op
from labwatch.utils.types import InconsistentSpace, ParamValueExcept


class SearchSpace(FixedDict):

    def __init__(self, storage):
        params = []
        fixed = {}
        self.uids_to_names = {}
        for key, value in storage.items():
            if isinstance(value, dict) and ("_class" in value):
                param = decode_param_or_op(value)
                params.append(param)
                self.uids_to_names[param["uid"]] = key
            else:
                fixed[key] = value

        super(SearchSpace, self).__init__(fixed=fixed)
        self.contains_conditions = False
        self.conditions = []
        self.non_conditions = []
        # first simply insert all and kick non unique names
        already_seen = set()
        for param in params:
            assert(isinstance(param, Parameter))
            self.contains_conditions &= isinstance(param,
                                                   ConditionResult)
            if param["uid"] in already_seen:
                err = "multiple definitions of {} " \
                      "in search-space, using first definition!"
                print(err.format(param["uid"]))
            else:
                self[self.uids_to_names[param["uid"]]] = param
                if isinstance(param, ConditionResult):
                    self.conditions.append(self.uids_to_names[param["uid"]])
                else:
                    self.non_conditions.append(self.uids_to_names[param["uid"]])
                already_seen.add(param["uid"])
        for pname in self.conditions:
            cparam = self[pname]
            conditioned_on = cparam["condition"]["uid"]
            if not (conditioned_on in already_seen):
                err = "Conditional parameter: {} depends on: {} " \
                      "which does not exist in the defined SearchSpace!"
                raise InconsistentSpace(err.format(pname, conditioned_on))

    def is_valid_name(self, name):
        return name in self.uids_to_names.values()
        
    def valid(self, config):
        # TODO check this again for consistency
        valid = True
        for pname in self.non_conditions:
            valid &= self[pname].valid(config[pname])
        if not valid:
            return False
        for pname in self.conditions:
            cparam = self[pname]
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
                res[pname] = self[pname].sample()
            else:
                res[pname] = self[pname].default()
            considered_params.add(pname)
        # then the conditional parameters
        remaining_params = set(self.conditions)
        i = 0
        while remaining_params:
            for pname in self.conditions:
                if pname in remaining_params:
                    cparam = self[pname]
                    conditioned_on = self.uids_to_names[cparam["condition"]["uid"]]
                    if conditioned_on in res.keys():
                        if strategy == "random":
                            cres = self[pname].sample(res[conditioned_on])
                        else:
                            cres = self[pname].default(res[conditioned_on])
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


# decorator
def build_searchspace(function):
    # abuse configscope to parse searchspace definitions
    scope = ConfigScope(function)
    space_dict = scope()
    # parse generic dict to a search space
    space = SearchSpace(space_dict)
    return space
