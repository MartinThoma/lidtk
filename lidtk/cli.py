#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run the lidtk main script."""

# Core Library modules
import logging.config

# Third party modules
import click
import pkg_resources
import yaml

# First party modules
import lidtk
import lidtk.analysis.unicode_block
import lidtk.classifiers.char_distribution.char_dist_metric_train_test as cdm
import lidtk.classifiers.cld2_mod
import lidtk.classifiers.google_mod
import lidtk.classifiers.langdetect_mod
import lidtk.classifiers.langid_mod
import lidtk.classifiers.nn
import lidtk.classifiers.text_cat
import lidtk.classifiers.tfidf_nn
import lidtk.data.create_ml_dataset
import lidtk.data.download_documents
import lidtk.utils

filepath = pkg_resources.resource_filename("lidtk", "config.yaml")
with open(filepath, "r") as stream:
    config = yaml.safe_load(stream)
logging.config.dictConfig(config["LOGGING"])


@click.group()
@click.version_option(version=lidtk.__version__)
def entry_point():
    pass


entry_point.add_command(lidtk.data.download_documents.main)
entry_point.add_command(lidtk.data.language_utils.main)
entry_point.add_command(lidtk.classifiers.text_cat.entry_point)
entry_point.add_command(lidtk.classifiers.cld2_mod.entry_point)
entry_point.add_command(lidtk.data.create_ml_dataset.main)
entry_point.add_command(lidtk.classifiers.nn.entry_point)
entry_point.add_command(lidtk.classifiers.langdetect_mod.entry_point)
entry_point.add_command(lidtk.classifiers.langid_mod.entry_point)
entry_point.add_command(lidtk.analysis.unicode_block.main)
entry_point.add_command(cdm.entry_point)
entry_point.add_command(lidtk.utils.map_classification_result)
entry_point.add_command(lidtk.classifiers.google_mod.entry_point)
entry_point.add_command(lidtk.classifiers.tfidf_nn.entry_point)


if __name__ == "__main__":
    entry_point()
