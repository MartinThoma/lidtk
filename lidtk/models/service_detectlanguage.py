#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for the usage of the detectlanguage.com service.

Prerequesites
-------------
* API key from https://github.com/detectlanguage/detectlanguage-python
* pip install detectlanguage
"""

# Core Library modules
import os

# Third party modules
import detectlanguage

detectlanguage2label = {
    u"gv": "glv",
    u"gu": "guj",
    u"sco": "sco",
    u"zh-Hant": u"lzh",
    u"gd": "gla",
    u"ga": "gle",
    u"gn": "grn",
    u"gl": "glg",
    u"lb": "ltz",
    u"la": "lat",
    u"ln": "lin",
    u"lo": "lao",
    u"tt": "tat",
    u"tr": "tur",
    u"lv": "lav",
    u"lt": "lit",
    u"tk": "tuk",
    u"th": "tha",
    u"tg": "tgk",
    u"te": "tel",
    u"ta": "tam",
    u"yi": "yid",
    u"ceb": "ceb",
    u"yo": "yor",
    u"de": "deu",
    u"da": "dan",
    u"dv": "div",
    u"qu": "que",
    u"el": "ell",
    u"eo": "epo",
    u"en": "eng",
    u"zh": "zho",
    u"bo": "bod",
    u"uk": "ukr",
    u"eu": "eus",
    u"et": "est",
    u"es": "spa",
    u"ru": "rus",
    u"rm": "roh",
    u"ro": "ron",
    u"be": "bel",
    u"bg": "bul",
    u"ba": "bak",
    u"bn": "ben",
    u"jw": "jav",
    u"bh": "bho",
    u"br": "bre",
    u"bs": "bos",
    u"ja": "jpn",
    u"oc": "oci",
    u"or": "ori",
    u"co": "cos",
    u"nso": "nso",
    u"ca": "cat",
    u"cy": "cym",
    u"cs": "ces",
    u"ps": "pus",
    u"pt": "por",
    u"tl": "tgl",
    u"pa": "pan",
    u"vi": "vie",
    u"war": "war",
    u"pl": "pol",
    u"hy": "hye",
    u"hr": "hrv",
    u"ht": "hat",
    u"hu": "hun",
    u"hi": "hin",
    u"vo": "vol",
    u"mg": "mlg",
    u"uz": "uzb",
    u"ml": "mal",
    u"mn": "mon",
    u"mi": "mri",
    u"mk": "mkd",
    u"ur": "urd",
    u"mt": "mlt",
    u"ms": "msa",
    u"mr": "mar",
    u"ug": "uig",
    u"my": "mya",
    u"af": "afr",
    u"sw": "swa",
    u"is": "isl",
    u"am": "amh",
    u"it": "ita",
    u"iw": "heb",
    u"sv": "swe",
    u"ia": "ina",
    u"as": "asm",
    u"ar": "ara",
    u"su": "sun",
    u"ay": "aym",
    u"az": "aze",
    u"ie": "ile",
    u"id": "ind",
    u"nl": "nld",
    u"no": "nob",
    u"ne": "nep",
    u"fr": "fra",
    u"fy": "fry",
    u"fa": "fas",
    u"fi": "fin",
    u"sa": "san",
    u"fo": "fao",
    u"ka": "kat",
    u"kk": "kaz",
    u"sr": "srp",
    u"sq": "sqi",
    u"ko": "kor",
    u"kn": "kan",
    u"km": "khm",
    u"sk": "slk",
    u"si": "sin",
    u"so": "som",
    u"sn": "sna",
    u"ku": "kur",
    u"sl": "slv",
    u"ky": "kir",
    u"sd": "snd",
    u"fj": "hif",
    u"egy": "arz",
}


class ServiceClassifier(object):
    """Wrap services in a class to get the desired interface."""

    def __init__(self, api_key):
        """Constructor."""
        self.api_key = api_key
        detectlanguage.configuration.api_key = api_key

    def predict(self, text_or_list):
        """
        Predict the language.

        Paramters
        ---------
        text_or_list : str

        Returns
        -------
        languages : list
        """
        res = detectlanguage.detect(text_or_list)
        return [el[0]["language"] for el in res]


def create_model(nb_classes, input_shape):
    """Create a model for LID."""
    model = ServiceClassifier("c5b442a13bdc4522f2a0463581c5bcb0")
    return model


def find_missmatches():
    """Find which languages detectlanguage and WiLI support."""
    print(
        "detectlanguage supports {} languages.".format(len(detectlanguage.languages()))
    )
    import sys

    sys.path.append("..")
    import language_utils

    labels_file = os.path.abspath("../labels.csv")
    language_data = language_utils.get_language_data(labels_file)
    wiki_codes = [el["Wiki Code"] for el in language_data]
    english_names = [el["English"].lower().decode("UTF-8") for el in language_data]

    wili_doesnt_know = []
    for lang in detectlanguage.languages():
        lang["name"] = str(lang["name"])
        if lang["code"] in detectlanguage2label:
            continue
        elif lang["code"] in wiki_codes:
            i = wiki_codes.index(lang["code"])
            detectlanguage2label[lang["code"]] = language_data[i]["Label"]
        elif lang["name"].lower() in english_names:
            i = english_names.index(lang["name"].lower())
            detectlanguage2label[lang["code"]] = language_data[i]["Label"]
        else:
            wili_doesnt_know.append(lang)

    dl_doesnt_know = []
    for lang in language_data:
        if lang["Label"] not in detectlanguage2label.values():
            dl_doesnt_know.append(lang["Label"])
    print(
        "WiLI does not know: {} ({})".format(
            [el for el in wili_doesnt_know], len(wili_doesnt_know)
        )
    )
    print(
        "Detectlanguage does not know: {} ({})".format(
            dl_doesnt_know, len(dl_doesnt_know)
        )
    )


if __name__ == "__main__":
    # model = create_model(None, None)
    # print(model.predict("Ich habe ein Haus."))
    find_missmatches()
