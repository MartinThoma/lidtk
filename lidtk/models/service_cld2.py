#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for the usage of the detectlanguage.com service.

Prerequesites
-------------
* API key from https://github.com/detectlanguage/detectlanguage-python
* pip install detectlanguage
"""

# Third party modules
import cld2
from fuzzywuzzy import process

servicecode2label = {
    "gv": "glv",
    "gu": "guj",
    "sco": "sco",
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
    "rm": "roh",
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
    "new": "new",
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
    "os": "oss",
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
    "nn": "nno",
    "no": "nob",
    "ne": "nep",
    "pam": "pam",
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
    "xx-Tibt": "bod",
    "xx-Orya": "ori",
    "xx-Java": "jav",
    "xx-Knda": "kan",
    "xx-Thai": "tha",
    "xx-Telu": "tel",
    "xx-Taml": "tam",
    "xx-Tglg": "tgl",
    "xx-Sund": "sun",
    "xx-Sinh": "sin",
    "xx-Mong": "mon",
    "xx-Mlym": "msa",
    "xx-Latn": "lat",
    "xx-Laoo": "lao",
    "xx-Hebr": "heb",
    "xx-Gujr": "guj",
    "xx-Geor": "kat",
    "xx-Beng": "ben",
    "xx-Armn": "hye",
    "xx-Arab": "ara",
    "lif": "lim",
    "xx-Grek": "ell",
    "xx-Khmr": "khm",
    "zzp": "",  # pig latin will not be supported
    "tlh": "",  # Klingon will not be supported
    "X_Gurmukhi": "pnb",  # will probably be supported
    "ab": "abk",
    "aa": "aar",
    "sm": "smo",
}


def find_missmatches():
    """
    Find which sets CLD-2 supports, but not WiLI and vice-versa.

    Print everything directly
    """
    print("cld2 supports {} languages.".format(len(cld2.LANGUAGES)))
    from lidtk.data import language_utils

    language_data = language_utils.get_language_data("labels.csv")
    wiki_codes = [el["Wiki Code"] for el in language_data]
    english_names = [el["English"].lower().decode("UTF-8") for el in language_data]

    wili_doesnt_know = []
    for lang in cld2.LANGUAGES:
        lang = {"name": lang[0], "code": lang[1]}
        lang["name"] = str(lang["name"])
        if lang["code"] in servicecode2label:
            continue
        elif lang["code"] in wiki_codes:
            i = wiki_codes.index(lang["code"])
            servicecode2label[lang["code"]] = language_data[i]["Label"]
        elif lang["name"].lower() in english_names:
            i = english_names.index(lang["name"].lower())
            servicecode2label[lang["code"]] = language_data[i]["Label"]
        else:
            norm = lang["name"]
            if "_" in norm:
                norm = norm.split("_")[1]
            extracted = process.extractOne(lang["name"], english_names)
            if extracted[1] > 60:
                i = english_names.index(extracted[0])
                print(
                    "service ({}): {} == {} (WiLI)? : '{}': '{}'".format(
                        lang["code"],
                        lang["name"],
                        extracted[0],
                        lang["code"],
                        language_data[i]["Label"],
                    )
                )
            wili_doesnt_know.append(lang)

    print(servicecode2label)

    dl_doesnt_know = []
    for lang in language_data:
        if lang["Label"] not in servicecode2label.values():
            dl_doesnt_know.append(lang["Label"])
    print(
        "WiLI does not know: {} ({})".format(
            [el for el in wili_doesnt_know], len(wili_doesnt_know)
        )
    )
    print("cld2 does not know: {} ({})".format(dl_doesnt_know, len(dl_doesnt_know)))


if __name__ == "__main__":
    # model = create_model(None, None)
    # print(model.predict("Ich habe ein Haus."))
    find_missmatches()
