import sys
import numpy as np
import importlib
import copy
import pprint

from sacred.config import ConfigScope
from six import integer_types

from labwatch.utils.types import str_to_class, \
    basic_types, types_to_str, str_to_types
from labwatch.utils.types import ParamValueExcept
from labwatch.utils import FixedDict

# global parameter counting
parameter_counter = 0
def get_parameter_counter():
    global parameter_counter
    res = parameter_counter
    parameter_counter += 1
    return res


# decode wrappers
def decode_param_or_op(storage):
    """ decode method for converting BSON like dicts
        to parameter values.
    """
    assert("_class" in storage)
    cname = str_to_class(storage["_class"])
    res = cname.decode(storage)
    return res

# parameters
# a parameter is a dict that can have blocked/fixed values

class Parameter(FixedDict):
    
    def __init__(self, uid=None, fixed=None):
        if uid is None:
            uid = get_parameter_counter()
        if fixed is None:
            fixed = {"uid" : uid}
        else:
            fixed["uid"] = uid
        super(Parameter, self).__init__(fixed = fixed)
        
    def __or__(self, other):
        if not isinstance(other, Condition):
            err = "or operator | requires Condition() instance on right side"
            raise ParamValueExcept(err)
        if other["uid"] == self["uid"]:
            err = "Parameter {} cannot be conditioned on itself"
            raise ParamInconsistent(err.format(self["uid"]))
        if isinstance(self, ConditionResult):
            err = "Parameter {} appears to be a nested condition " \
                  "specified via the | operator which is not supported"
            raise ParamInconsistent(err.format(self["uid"]))
        return ConditionResult(self, other)

    def default(self):
        raise NotImplementedError("default() not implemented")

    def valid(self):
        raise NotImplementedError("valid() not implemented")

    def sample(self):
        raise NotImplementedError("sample() not implemented")

    @classmethod
    def decode(cls, storage):
        raise NotImplementedError("decode() not implemented for class {}".format(cls))

class Constant(Parameter):
    def __init__(self, value, uid=None):
        super(Constant, self).__init__(uid=uid, fixed = {"value" : value})

    def default(self):
        return self["value"]
        
    def sample(self):
        return self.default()

    def valid(self, value):
        return self["value"] == value
        
    @classmethod
    def decode(cls, storage):
        uid = storage["uid"]
        value = storage["value"]
        return cls(value, uid=uid)

class Categorical(Parameter):
    def __init__(self, choices_in, uid=None):
        choices = []
        for choice in choices_in:
            if isinstance(choice, Constant):
                self.choices.append(choice)
            else:
                if not isinstance(choice, basic_types):
                    err = "Choice parameter {} is not " \
                          "a base type or Constant!"
                    raise ParamValueExcept(err.format(choice))
                choices.append(choice)
        fixed = {
            "choices" : choices
        }
        super(Categorical, self).__init__(uid=uid, fixed = fixed)

    def default(self):
        res = self["choices"][0]
        if isinstance(res, Constant):
            res = res.sample()
        return res
        
    def sample(self):
        res = np.random.choice(self["choices"])
        if isinstance(res, Constant):
            res = res.sample()
        return res

    def valid(self, value):
        any_valid = False
        for choice in self["choices"]:
            if isinstance(choice, Constant):            
                any_valid |= choice.valid(value)
            else:
                any_valid |= (value == choice)
        return any_valid
        
    @classmethod
    def decode(cls, storage):
        uid = storage["uid"]
        choices = []
        for choice in storage["choices"]:
            if isinstance(choice, dict):
                p = decode_param_or_op(choice)
            else:
                if not isinstance(choice, basic_types):
                    err = "Choice parameter {} is not " \
                          "a base type or Constant!"
                    raise ParamValueExcept(err.format(choice))
                choices.append(choice)
        return cls(choices, uid=uid)

class UniformNumber(Parameter):

    def __init__(self, 
                 lower,
                 upper,
                 type,
                 default=None,
                 log_scale=False,
                 uid=None):
        if default is None:
            if log_scale:
                default = np.exp((np.log(lower) + np.log(upper)) / 2.)
            else:
                default = (lower + upper) / 2.
        fixed = {
            "lower" : type(lower),
            "upper" : type(upper),
            "type" : types_to_str[type],
            "default" : type(default),
            "log_scale" : log_scale
        }
        super(UniformNumber, self).__init__(uid = uid, fixed = fixed)
        if not (self["lower"] <= self["default"] <= self["upper"]):
            err = "Default for {} is not between min and max".format(self["uid"])
            raise ParamValueExcept(err)
        if self["upper"] <= self["lower"]:
            err = "Upper bound {} is larger than lower bound {} for {} ".format(self["upper"], self["lower"], self["uid"])
            raise ParamValueExcept(err)

    def default(self):
        return self["default"]
        
    def sample(self):
        mtype = str_to_types[self["type"]]
        mmin = mtype(self["lower"])
        mmax = mtype(self["upper"])
        if self["log_scale"]:
            if mmin < 0. or mmax < 0.:
                raise ParamValueExcept("log_scale only allowed for positive ranges")
            mmin = mtype(np.log(np.maximum(mmin, 1e-7)))
            mmax = mtype(np.log(mmax))
        if mtype in integer_types:
            if self["log_scale"]:
                return mtype(np.exp(np.random.randint(mmin, mmax)))
            else:
                return np.random.randint(mmin, mmax)
        elif mtype == float:
            if self["log_scale"]:
                return mtype(np.exp(np.random.uniform(mmin, mmax)))
            else:
                return np.random.uniform(mmin, mmax)
        else:
            err = "Invalid type: {} for UniformNumber"
            raise ParamValueExcept(err.format(mtype))

    def valid(self, value):
        return self["lower"] <= value <= self["upper"]
    
    @classmethod
    def decode(cls, storage):
        uid = storage["uid"]
        type = str_to_types[storage["type"]]
        lower = type(storage["lower"])
        upper = type(storage["upper"])
        default = type(storage["default"])
        log_scale = bool(storage["log_scale"])
        return cls(lower, upper, type,
                   default, log_scale, uid)

class UniformFloat(UniformNumber):
    def __init__(self, 
                 lower,
                 upper,
                 default=None,
                 log_scale=False,
                 uid=None):
        super(UniformFloat, self).__init__(lower, upper, float,
                                           default=default,
                                           log_scale=log_scale,
                                           uid=uid)

    @classmethod
    def decode(cls, storage):
        uid = storage["uid"]
        type = str_to_types[storage["type"]]
        lower = type(storage["lower"])
        upper = type(storage["upper"])
        default = type(storage["default"])
        log_scale = bool(storage["log_scale"])
        return cls(lower, upper, 
                   default, log_scale, uid)
    
class UniformInt(UniformNumber):
    def __init__(self, 
                 lower,
                 upper,
                 default=None,
                 log_scale=False,
                 uid=None):
        super(UniformFloat, self).__init__(lower, upper, int,
                                           default=default,
                                           log_scale=log_scale,
                                           uid=uid)

    @classmethod
    def decode(cls, storage):
        uid = storage["uid"]
        type = str_to_types[storage["type"]]
        lower = type(storage["lower"])
        upper = type(storage["upper"])
        default = type(storage["default"])
        log_scale = bool(storage["log_scale"])
        return cls(lower, upper, 
                   default, log_scale, uid)


class Gaussian(Parameter):
    """ A Gaussian just has a different distribution 
    """
    def __init__(self,
                 mu,
                 sigma,
                 log_scale=False,
                 uid=None):
        type = float
        fixed = {
            "mu" : type(mu),
            "sigma" : type(sigma),
            "type" : types_to_str[type],
            "log_scale" : log_scale
        }
        super(Gaussian, self).__init__(uid=uid, fixed = fixed)

    def default(self):
        return self["mu"]
        
    def sample(self):
        mtype = str_to_types[self["type"]]
        mu = self["mu"]
        sigma = self["sigma"]
        if not (type == float):
            raise ParamValueExcept("Parameter with normal distribution" \
                                   " must be float!")
        if self["log_scale"]:
            return np.random.lognormal(mtype(mu), mtype(sigma))
        else:
            return np.random.normal(mtype(mu), mtype(sigma))

    def valid(self, value):
        return isinstance(value, (float, int, long))
    
    @classmethod
    def decode(cls, storage):
        uid = storage["uid"]
        type = str_to_types[storage["type"]]
        mu = type(storage["mu"])
        sigma = type(storage["sigma"])
        log_scale = bool(storage["log_scale"])
        return cls(mu, sigma, log_scale, uid=uid)    

class ConditionResult(Parameter):

    def __init__(self, result, condition):
        assert(isinstance(condition, Condition))
        uid = result["uid"]
        fixed = { "condition" : condition,
                  "result" : result
        }
        super(ConditionResult, self).__init__(uid=uid, fixed=fixed)

    def default(self, condition_res):
        condition_true = self["condition"].sample(condition_res)
        if condition_true:
            return self["result"].default()
        else:
            return None
        
    def sample(self, condition_res):
        condition_true = self["condition"].sample(condition_res)
        if condition_true:
            return self["result"].sample()
        else:
            return None

    def valid(self, value):
        return self["result"].valid(value)

    @classmethod
    def decode(cls, storage):
        condition = Condition.decode(storage["condition"])
        result = decode_param_or_op(storage["result"])
        return cls(result, condition)

# parameter operations
class ParameterOperation(FixedDict):
    pass
    
class Condition(ParameterOperation):
    
    def __init__(self, param, choices):
        assert(isinstance(param, (Categorical, int)))
        assert(isinstance(choices, list))
        if isinstance(param, str):
            uid = param
        elif isinstance(param, int):
            uid = param
        else:
            uid = param["uid"]
        fixed = {
            "uid" : uid,
            "choices" : choices
        }
        super(Condition, self).__init__(fixed=fixed)
        
    def sample(self, cres):
        for choice in self["choices"]:
            if isinstance(choice, Constant):
                if choice["value"] == cres:
                    return True
            elif choice == cres:
                return True
        return False

    def valid(self, cres):
        return True

    @classmethod
    def decode(cls, storage):
        choices = []
        for choice in storage["choices"]:
            if isinstance(choice, basic_types):
                choices.append(choice)
            else:
                choices.append(Constant.decode(choice))
        return cls(storage["uid"], choices)
