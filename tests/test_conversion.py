#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

import pytest
from labwatch.hyperparameters import *
from labwatch.searchspace import build_searchspace
from labwatch.converters.convert_to_configspace import (
    sacred_space_to_configspace, sacred_config_to_configspace,
    configspace_config_to_sacred)


def simple_sp():
    batch_size = UniformNumber(lower=32, upper=64, default=32, type=int)
    num_units_first_conv = UniformNumber(lower=32, upper=64, default=32,
                                         type=int)
    num_units_second_conv = UniformNumber(lower=32, upper=64, default=32,
                                          type=int)
    dropout_rate = UniformNumber(lower=0.2, upper=0.9, default=0.5, type=float)


def space_with_condition():
    batch_size = UniformNumber(lower=32, upper=64, default=32, type=int)
    n_layers = Categorical([1, 2])
    units_first = UniformNumber(lower=32,
                                upper=64, default=32, type=int)
    two = Constant(2)
    units_second = UniformNumber(lower=32,
                                 upper=64, default=32, type=int) | Condition(
        n_layers, [two])
    dropout_second = UniformNumber(lower=0.2, upper=0.8,
                                   default=0.5, type=float) | Condition(
        n_layers, [2])


def test_convert_small_config_space():
    space = build_searchspace(simple_sp)
    cspace = sacred_space_to_configspace(space)

    cs_non_conditions = cspace.get_all_unconditional_hyperparameters()
    for name in space.non_conditions:
        assert name in cs_non_conditions


def test_convert_larger_config_space():
    space = build_searchspace(space_with_condition)
    cspace = sacred_space_to_configspace(space)

    cs_non_conditions = cspace.get_all_unconditional_hyperparameters()
    for name in space.non_conditions:
        assert name in cs_non_conditions
    cs_conditions = cspace.get_all_conditional_hyperparameters()
    for name in space.conditions:
        assert name in cs_conditions


def test_convert_config():
    space = build_searchspace(space_with_condition)
    cspace = sacred_space_to_configspace(space)

    config = space.sample()
    cs_config = sacred_config_to_configspace(cspace, config)
    assert config == cs_config.get_dictionary()
    config_convert_back = configspace_config_to_sacred(cs_config)
    assert config == config_convert_back


def test_config_config_wrong_space_raises():
    space = build_searchspace(space_with_condition)
    cspace = sacred_space_to_configspace(space)

    config = space.sample()
    # passing the wrong type of space to config_to_configspace raises
    with pytest.raises(ValueError):
        cs_config = sacred_config_to_configspace(space, config)
