#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals

from pkg_resources import parse_version


def parse_name_ver(name_version):
    name, _, ver = name_version.partition('==')
    return name, parse_version(ver)


def check_dependencies(ex_dep, run_dep, version_policy):
    ex_dep = dict([parse_name_ver(name_version) for name_version in ex_dep])
    check_version = {
        'newer': lambda ex, name, b: name in ex and ex[name] >= b,
        'equal': lambda ex, name, b: name in ex and ex[name] == b,
        'exists': lambda ex, name, b: name in ex
    }[version_policy]
    for name_version in run_dep:
        name, ver = parse_name_ver(name_version)
        assert check_version(ex_dep, name, parse_version(ver)), \
            "{} mismatch: ex={}, run={}".format(name, ex_dep[name], ver)


def check_sources(ex_sources, run_sources):
    for ex_source, run_source in zip(ex_sources, run_sources):
        if not ex_source == tuple(run_source):
            raise KeyError('Source files did not match: experiment:'
                           ' {} [{}] != {} [{}] (run)'.format(
                            ex_source[0], ex_source[1],
                            run_source[0], run_source[1]))


def check_names(ex_name, run_name):
    if not ex_name == run_name:
        raise KeyError('experiment names did not match: experiment name '
                       '{} != {} (run name)'.format(ex_name, run_name))
