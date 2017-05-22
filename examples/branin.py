#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pymongo

from sacred import Experiment
from labwatch.assistant import LabAssistant
from labwatch.optimizers import BayesianOptimization
from labwatch.hyperparameters import UniformFloat
import numpy as np


ex = Experiment()
a = LabAssistant(ex, "labwatch_demo",
                 hostname="132.230.166.220",
                 port=27018,
                 optimizer=BayesianOptimization,
                 always_inject_observer=True)


@ex.config
def cfg():
    x = (0., 5.)


@a.searchspace
def search_space():
    x = (UniformFloat(-5, 10), UniformFloat(0, 15))


@ex.automain
def branin_cost(x):
    x1, x2 = x
    print("{:.2f}, {:.2f}".format(x1, x2))
    y = (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10

    return y
