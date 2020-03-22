#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Visualize C_0.99 for all languages except the 10 with most characters."""

# Third party modules
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
    to_analyze = [
        ("theta_80_len", "$|C_{0.80}|$", 150),
        ("theta_99_len", "$|C_{0.99}|$", 150),
        ("theta_100_len", "$|C_{1.00}|$", 150),
        ("paraphgrah_len_min", "Minimum paragrpah length", 500),
        ("paraphgrah_len_mean", "Average paragrpah length", 600),
        ("paraphgrah_len_max", "Maximum paragrpah length", 15000),
    ]
    for key, ylabel, threshold_max in to_analyze:
        print("key: {}".format(key))
        languages = sorted(
            [
                d[key]
                for iso_lang_code, d in lang_stats.items()
                if d[key] < threshold_max
            ]
        )
        for iso_lang_code, d in lang_stats.items():
            if d[key] >= threshold_max:
                print("* {lang}: {value}".format(lang=iso_lang_code, value=d[key]))
        colors = []
        en_found = False
        rus_found = False
        zh_yue_found = False
        eng_value = lang_stats["eng"][key]
        rus_value = lang_stats["rus"][key]
        zh_yue_value = lang_stats["zh-yue"][key]
        for value in languages:
            if value == eng_value and not en_found:
                colors.append("red")
                en_found = True
            elif value == rus_value and not rus_found:
                colors.append("blue")
                rus_found = True
            elif value == zh_yue_value and not zh_yue_found:
                colors.append("lime")
                zh_yue_found = True
            else:
                colors.append("grey")
        plot_1d(languages, colors, xlabel="Languages", ylabel=ylabel, name=key)


def plot_1d(l, colors=None, xlabel="", ylabel="", name="example"):
    """Plot a 1D list l of numbers."""
    ax = sns.barplot([i for i in range(len(l))], l, palette=colors)
    ax.set(xlabel=xlabel, ylabel=ylabel, label="big")
    ax.set_xticks([])

    plt.savefig("{name}.pdf".format(name=name))
    plt.savefig("{name}.png".format(name=name))
    plt.show()


if __name__ == "__main__":
    main()
