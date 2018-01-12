#!/usr/bin/env python

"""Utility functions for lidtk."""

# core modules
import logging
import os
import platform
import subprocess
import yaml


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


def get_software_info():
    """
    Get information about the used software.

    Returns
    -------
    sw_info : list
    """
    sw_info = {}
    sw_info['Platform'] = platform.platform()
    sw_info['Python Version'] = platform.python_version()
    return sw_info


def get_hardware_info():
    """
    Get important environment information that might influence experiments.

    Returns
    -------
    logstring : list
    """
    logstring = []
    cpuinfo_path = '/proc/cpuinfo'
    if os.path.isfile(cpuinfo_path):
        with open(cpuinfo_path) as f:
            cpuinfo = f.readlines()
        for line in cpuinfo:
            if "model name" in line:
                logstring.append("CPU: {}".format(line.strip()))
                break

    gpuinfo_path = '/proc/driver/nvidia/version'
    if os.path.isfile(gpuinfo_path):
        with open(gpuinfo_path) as f:
            version = f.read().strip()
        logstring.append("GPU driver: {}".format(version))
    try:
        logstring.append("VGA: {}".format(find_vga()))
    except:
        pass
    return logstring


def find_vga():
    """
    Get VGA identifier.

    Returns
    -------
    vga : str
    """
    vga = subprocess.check_output("lspci | grep -i 'vga\|3d\|2d'",
                                  shell=True,
                                  executable='/bin/bash')
    return vga


def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return cfg


def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : dict

    Returns
    -------
    cfg : dict
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            if not os.path.isfile(cfg[key]):
                logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg
