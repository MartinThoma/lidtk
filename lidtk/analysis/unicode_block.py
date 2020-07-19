#!/usr/bin/env python

"""Analyze  how important a Unicode block is for the different languages."""

# Core Library modules
import logging
from typing import Any, Dict, List

# Third party modules
import click
import numpy as np

# First party modules
from lidtk.data import wili

logger = logging.getLogger(__name__)


@click.command(name="analyze-unicode-block", help=__doc__)
@click.option("--start", default=123, show_default=True)
@click.option("--end", default=456, show_default=True, help="End of Unicode range")
def main(start: int, end: int) -> None:
    """Run."""
    # Read data
    data = wili.load_data()
    logger.info("Finished loading data")

    lang_amounts = {}  # type: Dict[str, List[Any]]
    for paragraph, label in zip(data["x_train"], data["y_train"]):
        if label not in lang_amounts:
            lang_amounts[label] = []
        chars_in_range = 0
        for char in paragraph:
            if start <= ord(char) <= end:
                chars_in_range += 1
        amount = float(chars_in_range) / len(paragraph)
        lang_amounts[label].append(amount)

    for key in lang_amounts.keys():
        lang_amounts[key] = np.array(lang_amounts[key]).mean() * 100

    print(f"Label    Chars in range [{start} - {end}]")
    print("-" * 80)
    lang_a = sorted(lang_amounts.items(), key=lambda n: n[1], reverse=True)
    for i, (label, chars_in_range_t) in enumerate(lang_a, start=1):
        print(f"{i:>3}. {label:<10}  {chars_in_range_t:>5.2f}%")
