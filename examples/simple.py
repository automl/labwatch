#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import pymongo

from sacred import Experiment
from labwatch.assistant import LabAssistant
from labwatch.hyperparameters import UniformFloat
import numpy as np

c = pymongo.MongoClient()
db = c.labwatch_demo
ex = Experiment('labwatch_simple_test')
a = LabAssistant(db, ex)

@ex.config
def cfg():
  C = 1.0
  gamma = 0.7


@a.searchspace
def search_space():
    C = UniformFloat(lower=0, upper=10, log_scale=True)
    gamma = UniformFloat(lower=0, upper=10)
  
@ex.automain
def main(C, gamma):
    print(C)
    print(gamma)

    return gamma * np.exp(C)
