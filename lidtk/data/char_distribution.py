#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Visualize C_0.99 for all languages except the 10 with most characters."""

# core modules
import logging

# 3rd party modules
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def main(lang_stats):
    """
    Plot basic statisics of languages.

    Parameters
    ----------
    lang_stats : list of dict
        Each dict represents a language
    """
    to_analyze = [('theta_80_len', '$|C_{0.80}|$'),
                  ('theta_99_len', '$|C_{0.99}|$'),
                  ('theta_100_len', '$|C_{1.00}|$'),
                  ('paraphgrah_len_min', 'Minimum paragrpah length'),
                  ('paraphgrah_len_mean', 'Average paragrpah length'),
                  ('paraphgrah_len_max', 'Maximum paragrpah length')]
    for key, ylabel in to_analyze:
        threshold_max = 500
        l = sorted([d[key]
                    for iso_lang_code, d in lang_stats.items()
                    if d[key] < threshold_max])
        colors = []
        en_found = False
        eng_value = lang_stats['eng'][key]
        rus_value = lang_stats['rus'][key]
        if 'zh-yue' in lang_stats:
            zh_yue_value = lang_stats['zh-yue'][key]
        else:
            zh_yue_value = 0
            logging.warning("Didn't find zh-yue.")
        for value in l:
            if value == eng_value and not en_found:
                colors.append('red')
                en_found = True
            elif value == rus_value:
                colors.append('blue')
            elif value == zh_yue_value:
                colors.append('lime')
            else:
                colors.append('grey')
        plot_1d(l, colors, xlabel='Languages', ylabel=ylabel)


def plot_1d(l, colors=None, xlabel='', ylabel=''):
    """Plot a 1D list l of numbers."""
    ax = sns.barplot([i for i in range(len(l))],
                     l,
                     palette=colors)
    ax.set(xlabel=xlabel, ylabel=ylabel, label='big')
    ax.set_xticks([])

    plt.savefig('example.pdf')
    plt.savefig('example.png')
    plt.show()


if __name__ == '__main__':
    main()
