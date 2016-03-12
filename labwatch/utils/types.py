import importlib


def str_to_class(cls_str):
    # module_name, class_name = cls_str.rsplit('.', 1)
    # somemod = importlib.import_module(module_name)
    somemod = importlib.import_module('labwatch.hyperparameters')
    class_name = cls_str
    return getattr(somemod, class_name)


def fullname(o):
    # return o.__module__ + "." + o.__class__.__name__
    return o.__class__.__name__

types_to_str = {
    float : "float",
    int   : "int",
    # long  : "long",  # FIXME: doesn't exist in python 3 anymore (remove?)
    str   : "str"
}

str_to_types = {
    "float" : float,
    "int"   : int,
    # "long"  : long, # FIXME: doesn't exist in python 3 anymore (remove?)
    "str"   : str
}

basic_types = tuple(types_to_str.keys())


class ParamValueExcept(Exception):
    pass


class InconsistentSpace(Exception):
    pass


class ParamInconsistent(Exception):
    pass
