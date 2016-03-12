#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from labwatch.utils.types import fullname


def warn_not_allowed(self, key):
    warning = "WARNING: you tried to set key {}" \
              " for class {} which is among the fixed keys!"
    print(warning.format(key, fullname(self)))


class FixedDict(dict):
    def __init__(self, fixed=None):
        super(FixedDict, self).__init__()
        for key in fixed.keys():
            dict.__setitem__(self, key, fixed[key])
        dict.__setitem__(self, "_class", fullname(self))
        self.fixed = set(fixed.keys()).union({"_class"})

    def __setitem__(self, key, value):
        """ __setitem__ for parameters only works 
            for non fixed values!
        """
        if key not in self.fixed:
            return dict.__setitem__(self, key, value)
        else:
            warn_not_allowed(self, key)

    def __delitem__(self, key):
        if key not in self.fixed:
            dict.__delitem__(self, key)
