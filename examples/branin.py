#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pymongo

from sacred import Experiment
from labwatch.assistant import LabAssistant
from labwatch.optimizers import SMAC3
from labwatch.hyperparameters import UniformFloat
import numpy as np

c = pymongo.MongoClient()
db = c.labwatch_branin
ex = Experiment('labwatch_branin_test')
a = LabAssistant(db, ex, optimizer=SMAC3, always_inject_observer=True)

@ex.config
def cfg():
    x1 = 0.
    x2 = 5.

@a.searchspace
def search_space():
    x1 = UniformFloat(lower=-5, upper=10)
    x2 = UniformFloat(lower=0, upper=15)
  
@ex.automain
def branin_cost(x1, x2):
    y = (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return y
