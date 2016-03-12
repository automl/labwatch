#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from labwatch.hyperparameters import *
from labwatch.searchspace import build_searchspace

import pprint

pp = pprint.PrettyPrinter(indent=4)


def test_small_config_space():
    def simple_sp():
        batch_size = UniformNumber(lower=32, upper=64, default=32, type=int)
        num_units_first_conv = UniformNumber(lower=32, upper=64, default=32,
                                             type=int)
        num_units_second_conv = UniformNumber(lower=32, upper=64, default=32,
                                              type=int)
        dropout_rate = UniformNumber(lower=0.2, upper=0.9, default=0.5,
                                     type=float)

    space = build_searchspace(simple_sp)
    cfg = space.sample()
    assert space.valid(cfg) == True


def test_config_space_with_condition():
    def space_with_condition():
        batch_size = UniformNumber(lower=32, upper=64, default=32, type=int)
        n_layers = Categorical([1, 2])
        units_first = UniformNumber(lower=32,
                                    upper=64, default=32, type=int)
        two = Constant(2)
        units_second = UniformNumber(lower=32,
                                     upper=64, default=32,
                                     type=int) | Condition(n_layers, [two])
        dropout_second = UniformNumber(lower=0.2, upper=0.8,
                                       default=0.5, type=float) | Condition(
            n_layers, [2])

    space = build_searchspace(space_with_condition)
    cfg = space.sample()
    pp.pprint(space)
    pp.pprint(cfg)
    assert space.valid(cfg) == True
