from labwatch.searchspace import SearchSpace
from labwatch.utils.types import str_to_class, \
    basic_types, types_to_str, str_to_types
from labwatch.utils.types import ParamValueExcept
import numpy as np

# TODO: guard ConfigSpace import
from ConfigSpace import ConfigurationSpace, Configuration
import ConfigSpace.hyperparameters as csh
from ConfigSpace.conditions import InCondition


def convert_simple_param(name, param):
    if param["_class"] == 'Constant':
        return csh.Constant(name, param["value"])
    elif param["_class"] == 'Categorical':
        # convert the choices to only contain
        # basic types (they might contain Constant parameters
        basic_choices = []
        for choice in param["choices"]:
            if isinstance(choice, dict):
                basic_choices.append(choice["default"])
            elif not isinstance(choice, basic_types):
                err = "Choice parameter {} is not " \
                      "a base type or Constant!"
                raise ParamValueExcept(err.format(choice))
            else:
                basic_choices.append(choice)        
        return csh.CategoricalHyperparameter(name=name,
                                            choices=basic_choices,
                                            default=basic_choices[0])
    elif param["_class"] == 'UniformFloat':
        return csh.UniformFloatHyperparameter(name=name,
                                             lower=param["lower"],
                                             upper=param["upper"],
                                             default=param["default"],
                                             log=param["log_scale"])
    elif param["_class"] == 'UniformInt':
        return csh.UniformIntegerHyperparameter(name=name,
                                               lower=param["lower"],
                                               upper=param["upper"],
                                               default=param["default"],
                                               log=param["log_scale"])
    elif param["_class"] == 'UniformNumber':
        ptype = str_to_types[param["type"]]
        if  ptype == float:
            return csh.UniformFloatHyperparameter(name=name,
                                                 lower=param["lower"],
                                                 upper=param["upper"],
                                                 default=param["default"],
                                                 log=param["log_scale"])
        elif ptype == int:
            return csh.UniformIntegerHyperparameter(name=name,
                                                   lower=param["lower"],
                                                   upper=param["upper"],
                                                   default=param["default"],
                                                   log=param["log_scale"])
        else:
            raise ValueError("Don't know how to represent UniformNumber "
                             "with type: {} in ConfigSpace notation".format(param["type"]))    
    elif param["_class"] == 'Gaussian':
        return csh.NormalFloatHyperparameter(name=name,
                                            mu=param["mu"],
                                            sigma=param["sigma"],
                                            log=param["log_scale"])
                                         
    else:
        raise ValueError("Don't know how to represent {} in ConfigSpace notation.".format(param))


def sacred_space_to_configspace(space):
    # first convert all non conditionals
    non_conditions = {}
    conditions = []
    for name in space.non_conditions:
        param = space[name]
        converted_param = convert_simple_param(name, param)
        non_conditions[name] = converted_param
    for name in space.conditions:
        param = space[name]
        converted_result = convert_simple_param(name, param["result"])
        # next build the condition as required by the ConfigSpace
        condition = param["condition"]
        condition_name = space.uids_to_names[condition["uid"]]
        converted_condition = None
        if non_conditions.has_key(condition_name):            
            converted_condition = non_conditions[condition_name]
        else:
            raise ValueError("Unknown parameter in Condition")
        converted_choices = []
        for choice in condition["choices"]:
            if isinstance(choice, dict):
                if choice["_class"] != "Constant":
                    raise ValueError("Invalid choice encountered in Condition")

                converted_choices.append(choice["value"])
            else:
                converted_choices.append(choice)
        cond = InCondition(converted_result,
                           converted_condition,
                           values=converted_choices)
        non_conditions[name] = converted_result
        conditions.append(cond)
    # finally build the ConfigSpace
    cs = ConfigurationSpace(seed=np.random.seed())
    for _name, param in non_conditions.items():
        cs.add_hyperparameter(param)
    for cond in conditions:
        cs.add_condition(cond)
    return cs


def sacred_config_to_configspace(cspace, config):
    if isinstance(cspace, SearchSpace):
        raise ValueError("You called sacred_config_to_configspace " \
                         "with an instance of labwatch.SearchSpace " \
                         "but an instance of ConfigSpace.ConfigurationSpace " \
                         "is required. Use sacred_space_to_configspace().")
    return Configuration(cspace, values=config)


def configspace_config_to_sacred(config):
    result = {}
    values = config.get_dictionary()
    for name,value in values.items():
        if value is None:
            # we remove all None entries as our
            # (sacred searchspace) convention is
            # to not include them
            pass
        else:
            result[name] = value
    return result

        
