#!/usr/bin/env python

"""Utility functions for lidtk."""

# core modules
import os


def make_path_absolute(path):
    """
    Get an absolute filepath.

    Parameters
    ----------
    path : str

    Returns
    -------
    absolute_path : str
    """
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path
