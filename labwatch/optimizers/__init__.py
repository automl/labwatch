#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from .base import Optimizer
from .random_search import RandomSearch

try:
    from .smac import SMAC3
except ImportError:
    print('WARNING: SMAC not found')

try:
    from .bayesian_optimization import BayesianOptimization
except ImportError:
    print('WARNING: BayesianOptimization not found')
