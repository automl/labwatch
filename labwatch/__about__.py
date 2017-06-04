#!/usr/bin/env python
# coding=utf-8
"""
This module contains meta-information about the labwatch package.
It is kept simple and separate from the main module, because this information
is also read by the setup.py. And during installation the labwatch module cannot
yet be imported.
"""

from __future__ import division, print_function, unicode_literals

__all__ = ["__version__", "__authors__", "__url__"]

__version__ = "0.1.0"

__authors__ = 'Aaron Klein, Klaus Greff'

__url__ = "https://github.com/automl/labwatch"
