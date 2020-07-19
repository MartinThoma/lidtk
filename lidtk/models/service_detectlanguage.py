#!/usr/bin/env python

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
    "gv": "glv",
    "gu": "guj",
    "sco": "sco",
    "zh-Hant": "lzh",
    "gd": "gla",
    "ga": "gle",
    "gn": "grn",
    "gl": "glg",
    "lb": "ltz",
    "la": "lat",
    "ln": "lin",
    "lo": "lao",
    "tt": "tat",
    "tr": "tur",
    "lv": "lav",
    "lt": "lit",
    "tk": "tuk",
    "th": "tha",
    "tg": "tgk",
    "te": "tel",
    "ta": "tam",
    "yi": "yid",
    "ceb": "ceb",
    "yo": "yor",
    "de": "deu",
    "da": "dan",
    "dv": "div",
    "qu": "que",
    "el": "ell",
    "eo": "epo",
    "en": "eng",
    "zh": "zho",
    "bo": "bod",
    "uk": "ukr",
    "eu": "eus",
    "et": "est",
    "es": "spa",
    "ru": "rus",
    "rm": "roh",
    "ro": "ron",
    "be": "bel",
    "bg": "bul",
    "ba": "bak",
    "bn": "ben",
    "jw": "jav",
    "bh": "bho",
    "br": "bre",
    "bs": "bos",
    "ja": "jpn",
    "oc": "oci",
    "or": "ori",
    "co": "cos",
    "nso": "nso",
    "ca": "cat",
    "cy": "cym",
    "cs": "ces",
    "ps": "pus",
    "pt": "por",
    "tl": "tgl",
    "pa": "pan",
    "vi": "vie",
    "war": "war",
    "pl": "pol",
    "hy": "hye",
    "hr": "hrv",
    "ht": "hat",
    "hu": "hun",
    "hi": "hin",
    "vo": "vol",
    "mg": "mlg",
    "uz": "uzb",
    "ml": "mal",
    "mn": "mon",
    "mi": "mri",
    "mk": "mkd",
    "ur": "urd",
    "mt": "mlt",
    "ms": "msa",
    "mr": "mar",
    "ug": "uig",
    "my": "mya",
    "af": "afr",
    "sw": "swa",
    "is": "isl",
    "am": "amh",
    "it": "ita",
    "iw": "heb",
    "sv": "swe",
    "ia": "ina",
    "as": "asm",
    "ar": "ara",
    "su": "sun",
    "ay": "aym",
    "az": "aze",
    "ie": "ile",
    "id": "ind",
    "nl": "nld",
    "no": "nob",
    "ne": "nep",
    "fr": "fra",
    "fy": "fry",
    "fa": "fas",
    "fi": "fin",
    "sa": "san",
    "fo": "fao",
    "ka": "kat",
    "kk": "kaz",
    "sr": "srp",
    "sq": "sqi",
    "ko": "kor",
    "kn": "kan",
    "km": "khm",
    "sk": "slk",
    "si": "sin",
    "so": "som",
    "sn": "sna",
    "ku": "kur",
    "sl": "slv",
    "ky": "kir",
    "sd": "snd",
    "fj": "hif",
    "egy": "arz",
}


class ServiceClassifier:
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
    print(f"detectlanguage supports {len(detectlanguage.languages())} languages.")
    # Core Library modules
    import sys

    sys.path.append("..")
    # Third party modules
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
    print(f"WiLI does not know: {list(wili_doesnt_know)} " f"({len(wili_doesnt_know)})")
    print(f"Detectlanguage does not know: {dl_doesnt_know} ({len(dl_doesnt_know)})")


if __name__ == "__main__":
    # model = create_model(None, None)
    # print(model.predict("Ich habe ein Haus."))
    find_missmatches()
