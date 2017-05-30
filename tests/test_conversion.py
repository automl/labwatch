#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import json
import pytest

from labwatch.hyperparameters import *
from labwatch.searchspace import build_searchspace, collect_hyperparameters, \
    fill_in_values
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
    for name in space.conditions:        assert name in cs_conditions


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


@pytest.mark.parametrize("hparam, name", [
    (Constant(23), 'Constant'),
    (UniformFloat(0., 1.), 'UniformFloat'),
    (UniformInt(1, 10), 'UniformInt'),
    (UniformNumber(0, 1, float), 'UniformNumber'),
    (Categorical([2, 4, 6]), 'Categorical'),
    (Gaussian(0, 1.0), 'Gaussian')
])
def test_automatic_hyperparameters_to_dict_conversion(hparam, name):
    # use JSON serialization to force dict conversion
    d = json.loads(json.dumps(hparam))
    assert isinstance(d, dict)
    assert "uid" in d
    assert "_class" in d
    assert d['_class'] == name


@pytest.mark.parametrize("hparam", [
    Constant(23),
    UniformFloat(0., 1.),
    UniformInt(1, 10),
    UniformNumber(0, 1, float),
    Categorical([2, 4, 6]),
    Gaussian(0, 1.0),
])
def test_decoding_hyperparameters_from_dict(hparam):
    # use JSON serialization to force dict conversion
    d = json.loads(json.dumps(hparam))
    # decode
    h = decode_param_or_op(d)
    assert isinstance(h, Parameter)
    assert isinstance(h, type(hparam))
    assert h == hparam


def test_simple_searchspace_conversion():
    a = Constant(7)
    b = UniformFloat(0, 1)
    space = {
        'a': a,
        'b': b
    }
    sp_dict = json.loads(json.dumps(space))

    params = collect_hyperparameters(sp_dict, {})
    assert params == {
        a['uid']: a,
        b['uid']: b
    }


def test_searchspace_conversion_with_repetition():
    a = UniformFloat(0, 1)
    space = {
        'a': a,
        'b': a
    }
    sp_dict = json.loads(json.dumps(space))

    params = collect_hyperparameters(sp_dict, {})
    assert params == {
        a['uid']: a
    }


def test_searchspace_conversion_with_substructure():
    a = UniformFloat(0, 1)
    b = UniformInt(2, 12)
    c = Gaussian(0, 1)
    space = {
        'a': a,
        'foo': {
            'bar': b,
            'nested': {
                'a': a
            }
        },
        'using_list': [a, b, c]
    }
    sp_dict = json.loads(json.dumps(space))

    params = collect_hyperparameters(sp_dict, {})
    assert params == {
        a['uid']: a,
        b['uid']: b,
        c['uid']: c
    }

    assert params[a['uid']]['name'] == 'a'
    assert params[b['uid']]['name'] == 'foo.bar'
    assert params[c['uid']]['name'] == 'using_list[2]'


def test_fill_in_values():
    a = UniformFloat(0, 1)
    b = UniformInt(2, 12)
    c = Gaussian(0, 1)
    search_space = json.loads(json.dumps({
        'a': a,
        'foo': {
            'bar': b,
            'nested': {
                'a': a
            }
        },
        'using_list': [a, b, c]
    }))
    values = {
        a['uid']: 11,
        b['uid']: 2.2,
        c['uid']: 'c'
    }
    cfg = fill_in_values(search_space, values)
    assert cfg == {
        'a': 11,
        'foo': {
            'bar': 2.2,
            'nested': {
                'a': 11
            }
        },
        'using_list': [11, 2.2, 'c']
    }
