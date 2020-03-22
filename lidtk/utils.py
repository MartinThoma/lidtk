#!/usr/bin/env python

"""Utility functions for lidtk."""

# Core Library modules
import logging
import os
import platform
import subprocess

# Third party modules
import click
import pkg_resources
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
    sw_info : dict
    """
    sw_info = {}
    sw_info["Platform"] = platform.platform()
    sw_info["Python Version"] = platform.python_version()
    return sw_info


def get_hardware_info():
    """
    Get important environment information that might influence experiments.

    Returns
    -------
    hw_info : dict
    """
    hw_info = {}
    cpuinfo_path = "/proc/cpuinfo"
    if os.path.isfile(cpuinfo_path):
        with open(cpuinfo_path) as f:
            cpuinfo = f.readlines()
        for line in cpuinfo:
            if "model name" in line:
                hw_info["CPU"] = line.strip()
                break

    gpuinfo_path = "/proc/driver/nvidia/version"
    if os.path.isfile(gpuinfo_path):
        with open(gpuinfo_path) as f:
            version = f.read().strip()
        hw_info["GPU driver"] = version
    try:
        hw_info["VGA"] = find_vga()
    except:  # noqa
        pass
    return hw_info


def find_vga():
    """
    Get VGA identifier.

    Returns
    -------
    vga : str
    """
    vga = subprocess.check_output(
        r"lspci | grep -i 'vga\|3d\|2d'", shell=True, executable="/bin/bash"
    )
    return str(vga).strip()


def load_cfg(yaml_filepath=None):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str, optional (default: package config file)

    Returns
    -------
    cfg : dict
    """
    if yaml_filepath is None:
        yaml_filepath = pkg_resources.resource_filename("lidtk", "config.yaml")
    # Read YAML experiment definition file
    with open(yaml_filepath, "r") as stream:
        cfg = yaml.safe_load(stream)
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
        if hasattr(key, "endswith") and key.endswith("_path"):
            if cfg[key].startswith("~"):
                cfg[key] = os.path.expanduser(cfg[key])
            else:
                cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg


@click.command(name="map", help="Map predictions to something known by WiLI")
@click.option(
    "--config", type=click.Path(exists=True), help="Path to a YAML configuration file"
)
@click.option("--source", type=click.Path(exists=True), help="Path to a txt file")
@click.option("--dest", type=click.Path(exists=False), help="Path to a txt file")
def map_classification_result(config, source, dest):
    """
    Map the classification to something known by WiLI.

    Parameters
    ----------
    config : str
    source : str
    dest : str
    """
    cfg = load_cfg(config)

    # Read data
    with open(source, "r") as fp:
        read_lines = fp.readlines()
        read_lines = [line.rstrip("\n") for line in read_lines]

    # Create new data
    new_lines = []
    for line in read_lines:
        if line in cfg["mapping"]:
            new_lines.append(cfg["mapping"][line])
        else:
            new_lines.append("unk")
            logging.warning("Map '{}' to 'unk'".format(line))

    # Write text file
    with open(dest, "w") as fp:
        fp.write("\n".join(new_lines))
