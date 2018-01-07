#!/usr/bin/env python

# Copyright (c) 2006 Thomas Mangin

# This program is distributed under Gnu General Public License
# (cf. the file COPYING in distribution). Alternatively, you can use
# the program under the conditions of the Artistic License (as Perl).

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

# core moudles
import codecs
import glob
import logging
import os
import re
import sys
try:
    from exceptions import KeyboardInterrupt
except:
    pass

# 3rd party modules
import click

# internal modules
from lidtk.utils import make_path_absolute

nb_ngrams = 400
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


###############################################################################
# CLI                                                                         #
###############################################################################
@click.group(name='textcat')
def cli():
    """Utility function for click to group commands."""
    pass


@cli.command(name='predict')
@click.option('--path',
              default='~/.lidtk/text_cat/lang_lm',
              help='Path to a directory with *.lm files')
@click.option('--text')
def predict_cli(path, text):
    """Classify the language of a given input."""
    l = NGram(make_path_absolute(path))
    print(l.classify(text))


@cli.command()
@click.option('--txt_dir', default='~/.lidtk/text_cat/lang_txt/')
@click.option('--lm_dir', default='~/.lidtk/text_cat/lang_lm/')
def generate(txt_dir, lm_dir):
    """Generate your own .lm files."""
    conf = Generate(make_path_absolute(txt_dir))
    conf.save(make_path_absolute(lm_dir))


###############################################################################
# Logic                                                                       #
###############################################################################
class _NGram:
    def __init__(self, arg={}):
        t = type(arg)
        if t == type(""):
            self.add_text(arg)
            self.normalise()
        elif t == type({}):
            self.ngrams = arg
            self.normalise()
        else:
            self.ngrams = dict()

    def add_text(self, text):
        ngrams = dict()

        text = text.replace('\n', ' ')
        text = re.sub('\s+', ' ', text)
        words = text.split(' ')

        for word in words:
            word = '_' + word + '_'
            size = len(word)
            for i in range(size):
                for s in (1, 2, 3, 4):
                    sub = word[i:i + s]
                    # print "[",sub,"]"
                    if sub not in ngrams:
                        ngrams[sub] = 0
                    ngrams[sub] += 1

                    if i + s >= size:
                        break
        self.ngrams = ngrams
        return self

    def sorted(self):
        sorted = [(self.ngrams[k], k) for k in self.ngrams.keys()]
        sorted.sort()
        sorted.reverse()
        sorted = sorted[:nb_ngrams]
        return sorted

    def normalise(self):
        count = 0
        ngrams = dict()
        for v, k in self.sorted():
            ngrams[k] = count
            count += 1

        self.ngrams = ngrams
        return self

    def compare(self, ngram):
        d = 0
        ngrams = ngram.ngrams
        for k in self.ngrams.keys():
            if k in ngrams:
                d += abs(ngrams[k] - self.ngrams[k])
            else:
                d += nb_ngrams
        return d


class NGram:
    def __init__(self, folder, ext='.lm'):
        self.ngrams = dict()
        folder = os.path.join(folder, '*' + ext)
        size = len(ext)
        count = 0

        for fname in sorted(glob.glob(os.path.normcase(folder))):
            count += 1
            lang = os.path.split(fname)[-1][:-size]
            ngrams = dict()
            with codecs.open(fname, 'r', 'utf8') as file:
                for line in file.readlines():
                    parts = line[:-1].split('\t ')
                    if len(parts) != 2:
                        raise ValueError("invalid language file %s line : %s" %
                                         (fname, parts))
                    try:
                        ngrams[parts[0]] = int(parts[1])
                    except KeyboardInterrupt:
                        raise
                    except:
                        raise ValueError("invalid language file %s line : %s" %
                                         (fname, parts))

                if len(ngrams.keys()):
                    self.ngrams[lang] = _NGram(ngrams)

        if not count:
            raise ValueError("no language files found at folder={}"
                             .format(folder))

    def classify(self, text):
        ngram = _NGram(text)
        r = 'guess'

        langs = list(self.ngrams.keys())
        r = langs.pop()
        min = self.ngrams[r].compare(ngram)

        for lang in langs:
            d = self.ngrams[lang].compare(ngram)
            if d < min:
                min = d
                r = lang

        return r


class Generate:
    """
    Generate language files.

    Parameters
    ----------
    folder : str
    ext : str
    """

    def __init__(self, folder, ext='.txt'):
        self.ngrams = dict()
        folder = os.path.normcase(os.path.join(folder, '*' + ext))
        logging.info("Search files ending with {} in {}"
                     .format(ext, folder))
        size = len(ext)
        count = 0

        for fname in glob.glob(folder):
            count += 1
            lang = os.path.split(fname)[-1][:-size]
            n = _NGram()

            file = codecs.open(fname, 'r', 'utf8')
            for line in file.readlines():
                n.add_text(line)
            file.close()

            n.normalise()
            self.ngrams[lang] = n

    def save(self, folder, ext='.lm'):
        logging.info("Will save {} files to {}."
                     .format(len(self.ngrams), folder))
        for lang in self.ngrams.keys():
            fname = os.path.join(folder, lang + ext)
            file = codecs.open(fname, 'w', 'utf8')
            for v, k in self.ngrams[lang].sorted():
                file.write("%s\t %d\n" % (k, v))
            file.close()
            print('saved {}'.format(folder))
