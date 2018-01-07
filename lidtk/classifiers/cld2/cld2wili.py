#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Map CLD2 to wili.

The following classes from WiLI seem not to exist in CLD-2:

['ace', 'als', 'ang', 'arg', 'arz', 'asm', 'ast', 'ava', 'aym', 'azb', 'bak',
'bar', 'bcl', 'be-tarask', 'bjn', 'bpy', 'bre', 'bxr', 'cbk', 'cdo', 'che',
'chv', 'ckb', 'cor', 'cos', 'crh', 'csb', 'diq', 'dsb', 'dty', 'egl', 'epo',
'ext', 'fao', 'frp', 'fry', 'fur', 'gag', 'glk', 'glv', 'grn', 'hak', 'hau',
'hbs', 'hif', 'hsb', 'ibo', 'ido', 'ile', 'ilo', 'ina', 'jam', 'jbo', 'kaa',
'kab', 'kbd', 'koi', 'kok', 'kom', 'krc', 'ksh', 'lad', 'lat', 'lez', 'lij',
'lim', 'lin', 'lmo', 'lrc', 'ltg', 'ltz', 'mai', 'map-bms', 'mdf', 'mhr',
'min', 'mon', 'mri', 'mrj', 'mwl', 'myv', 'mzn', 'nan', 'nap', 'nav', 'nci',
'nds', 'nds-nl', 'new', 'nno', 'nrm', 'nso', 'oci', 'olo', 'orm', 'oss', 'pag',
'pam', 'pap', 'pcd', 'pdc', 'pfl', 'pnb', 'pus', 'que', 'roa-tara', 'roh',
'rue', 'rup', 'sah', 'san', 'scn', 'sco', 'sgs', 'sme', 'sna', 'snd', 'som',
'srd', 'srn', 'stq', 'szl', 'tat', 'tcy', 'tet', 'ton', 'tsn', 'tuk', 'tyv',
'udm', 'uig', 'vec', 'vep', 'vls', 'vol', 'vro', 'war', 'wln', 'wol', 'wuu',
'xho', 'xmf', 'yor', 'zea', 'zh-yue']
"""

import cld2


servicecode2label = {'gv': 'glv', 'gu': 'guj', 'sco': 'sco', 'gd': 'gla',
                     'ga': 'gle', 'gn': 'grn', 'gl': 'glg', 'lb': 'ltz',
                     'la': 'lat', 'ln': 'lin', 'lo': 'lao', 'tt': 'tat',
                     'tr': 'tur', 'lv': 'lav', 'lt': 'lit', 'tk': 'tuk',
                     'th': 'tha', 'tg': 'tgk', 'te': 'tel', 'ta': 'tam',
                     'yi': 'yid', 'ceb': 'ceb', 'yo': 'yor', 'de': 'deu',
                     'da': 'dan', 'rm': 'roh', 'dv': 'div', 'qu': 'que',
                     'el': 'ell', 'eo': 'epo', 'en': 'eng', 'zh': 'zho',
                     'bo': 'bod', 'uk': 'ukr', 'eu': 'eus', 'et': 'est',
                     'es': 'spa', 'ru': 'rus', 'new': 'new', 'ro': 'ron',
                     'be': 'bel', 'bg': 'bul', 'ba': 'bak', 'bn': 'ben',
                     'jw': 'jav', 'bh': 'bho', 'br': 'bre', 'bs': 'bos',
                     'ja': 'jpn', 'oc': 'oci', 'os': 'oss', 'or': 'ori',
                     'co': 'cos', 'nso': 'nso', 'ca': 'cat', 'cy': 'cym',
                     'cs': 'ces', 'ps': 'pus', 'pt': 'por', 'tl': 'tgl',
                     'pa': 'pan', 'vi': 'vie', 'war': 'war', 'pl': 'pol',
                     'hy': 'hye', 'hr': 'hrv', 'ht': 'hat', 'hu': 'hun',
                     'hi': 'hin', 'vo': 'vol', 'mg': 'mlg', 'uz': 'uzb',
                     'ml': 'mal', 'mn': 'mon', 'mi': 'mri', 'mk': 'mkd',
                     'ur': 'urd', 'mt': 'mlt', 'ms': 'msa', 'mr': 'mar',
                     'ug': 'uig', 'my': 'mya', 'af': 'afr', 'sw': 'swa',
                     'is': 'isl', 'am': 'amh', 'it': 'ita', 'iw': 'heb',
                     'sv': 'swe', 'ia': 'ina', 'as': 'asm', 'ar': 'ara',
                     'su': 'sun', 'ay': 'aym', 'az': 'aze', 'ie': 'ile',
                     'id': 'ind', 'nl': 'nld', 'nn': 'nno', 'no': 'nob',
                     'ne': 'nep', 'pam': 'pam', 'fr': 'fra', 'fy': 'fry',
                     'fa': 'fas', 'fi': 'fin', 'sa': 'san', 'fo': 'fao',
                     'ka': 'kat', 'kk': 'kaz', 'sr': 'srp', 'sq': 'sqi',
                     'ko': 'kor', 'kn': 'kan', 'km': 'khm', 'sk': 'slk',
                     'si': 'sin', 'so': 'som', 'sn': 'sna', 'ku': 'kur',
                     'sl': 'slv', 'ky': 'kir', 'sd': 'snd',
                     'to': 'ton',
                     'tn': 'tsn',
                     'wo': 'wol',
                     'xh': 'xho',
                     'chr': 'chr',
                     'rw': 'kin',
                     'lg': 'lug',
                     'st': 'UNK',  # Southern Sotho
                     'ny': 'UNK',  # Chichewa, Chewa, Nyanja
                     'syr': 'UNK',  # TODO
                     'hmn': 'UNK',  # TODO
                     'xx-Tfng': 'UNK',  # X_Tifinagh
                     'xx-Copt': 'UNK',  # X_Coptic
                     'xx-Goth': 'UNK',  # X_Gothic
                     'xx-Yiii': 'UNK',  # X_Yi
                     'iu': 'UNK',  # Eskimo–Aleut
                     'un': 'UNK',  # Unknown
                     'zh-Hant': 'lzh',
                     'xx-Tibt': 'bod',
                     'xx-Orya': 'ori',
                     'xx-Java': 'jav',
                     'xx-Knda': 'kan',
                     'xx-Thai': 'tha',
                     'xx-Telu': 'tel',
                     'xx-Taml': 'tam',
                     'xx-Tglg': 'tgl',
                     'xx-Sund': 'sun',
                     'xx-Sinh': 'sin',
                     'xx-Mong': 'mon',
                     'xx-Mlym': 'msa',
                     'xx-Latn': 'lat',
                     'xx-Laoo': 'lao',
                     'xx-Hebr': 'heb',
                     'xx-Gujr': 'guj',
                     'xx-Geor': 'kat',
                     'xx-Beng': 'ben',
                     'xx-Armn': 'hye',
                     'xx-Arab': 'ara',
                     'lif': 'lim',
                     'xx-Grek': 'ell',
                     'xx-Khmr': 'khm'}

langname2label = {'SWEDISH': 'swe', 'KURDISH': 'kur', 'SUNDANESE': 'sun',
                  'ORIYA': 'ori', 'DUTCH': 'nld', 'ASSAMESE': 'asm',
                  'NORWEGIAN_N': 'nno', 'NEPALI': 'nep', 'VIETNAMESE': 'vie',
                  'MANX': 'glv', 'PEDI': 'nso', 'ALBANIAN': 'sqi', 'KANNADA':
                  'kan', 'HAITIAN_CREOLE': 'hat', 'NEWARI': 'new', 'PERSIAN':
                  'fas', 'QUECHUA': 'que', 'X_Arabic': 'ara', 'ESPERANTO':
                  'epo', 'OCCITAN': 'oci', 'INDONESIAN': 'ind', 'LITHUANIAN':
                  'lit', 'X_Greek': 'ell', 'SHONA': 'sna', 'X_Tagalog': 'tgl',
                  'BURMESE': 'mya', 'POLISH': 'pol', 'LAOTHIAN': 'lao',
                  'IRISH': 'gle', 'BOSNIAN': 'bos', 'TATAR': 'tat', 'ROMANIAN':
                  'ron', 'X_Sinhala': 'sin', 'X_Thai': 'tha', 'SWAHILI': 'swa',
                  'DANISH': 'dan', 'ICELANDIC': 'isl', 'X_Bengali': 'ben',
                  'SLOVENIAN': 'slv', 'FRENCH': 'fra', 'GALICIAN': 'glg',
                  'SLOVAK': 'slk', 'ENGLISH': 'eng', 'THAI': 'tha', 'OSSETIAN':
                  'oss', 'KHMER': 'khm', 'ESTONIAN': 'est', 'TELUGU': 'tel',
                  'UKRAINIAN': 'ukr', 'CZECH': 'ces', 'MALAY': 'msa',
                  'X_Javanese': 'jav', 'SCOTS': 'sco', 'YIDDISH': 'yid',
                  'KYRGYZ': 'kir', 'PUNJABI': 'pan', 'CROATIAN': 'hrv',
                  'X_Lao': 'lao', 'GUARANI': 'grn', 'UZBEK': 'uzb', 'GEORGIAN':
                  'kat', 'MACEDONIAN': 'mkd', 'VOLAPUK': 'vol', 'X_Sundanese':
                  'sun', 'AZERBAIJANI': 'aze', 'Chinese': 'zho', 'DHIVEHI':
                  'div', 'INTERLINGUA': 'ina', 'PASHTO': 'pus', 'CORSICAN':
                  'cos', 'INTERLINGUE': 'ile', 'CATALAN': 'cat',
                  'LUXEMBOURGISH': 'ltz', 'SOMALI': 'som', 'SINDHI': 'snd',
                  'TURKMEN': 'tuk', 'X_Gujarati': 'guj', 'BASQUE': 'eus',
                  'URDU': 'urd', 'HINDI': 'hin', 'TAGALOG': 'tgl', 'MALAYALAM':
                  'mal', 'X_Khmer': 'khm', 'Korean': 'kor', 'BRETON': 'bre',
                  'LINGALA': 'lin', 'LATIN': 'lat', 'GUJARATI': 'guj',
                  'BASHKIR': 'bak', 'UIGHUR': 'uig', 'MONGOLIAN': 'mon',
                  'SPANISH': 'spa', 'AFRIKAANS': 'afr', 'PORTUGUESE': 'por',
                  'X_Tamil': 'tam', 'FINNISH': 'fin', 'GERMAN': 'deu',
                  'SINHALESE': 'sin', 'BELARUSIAN': 'bel', 'KAZAKH': 'kaz',
                  'RHAETO_ROMANCE': 'roh', 'HUNGARIAN': 'hun', 'AMHARIC':
                  'amh', 'NORWEGIAN': 'nob', 'PAMPANGA': 'pam', 'TURKISH':
                  'tur', 'RUSSIAN': 'rus', 'TIBETAN': 'bod', 'MALAGASY': 'mlg',
                  'BIHARI': 'bho', 'MARATHI': 'mar', 'GREEK': 'ell', 'HEBREW':
                  'heb', 'X_Georgian': 'kat', 'X_Armenian': 'hye', 'ITALIAN':
                  'ita', 'MAORI': 'mri', 'X_Latin': 'lat', 'BENGALI': 'ben',
                  'MALTESE': 'mlt', 'TAJIK': 'tgk', 'TAMIL': 'tam',
                  'X_Mongolian': 'mon', 'X_Hebrew': 'heb', 'LIMBU': 'lim',
                  'SCOTS_GAELIC': 'gla', 'YORUBA': 'yor', 'X_Tibetan': 'bod',
                  'ARMENIAN': 'hye', 'LATVIAN': 'lav', 'SERBIAN': 'srp',
                  'FAROESE': 'fao', 'AYMARA': 'aym', 'ARABIC': 'ara',
                  'X_Oriya': 'ori', 'FRISIAN': 'fry', 'BULGARIAN': 'bul',
                  'WARAY_PHILIPPINES': 'war', 'Japanese': 'jpn', 'WELSH':
                  'cym', 'X_Telugu': 'tel', 'JAVANESE': 'jav', 'SANSKRIT':
                  'san', 'X_Kannada': 'kan', 'CEBUANO': 'ceb', 'X_Malayalam':
                  'msa'}

langname2label = {}


def init():
    for lang in cld2.LANGUAGES:
        lang = {'name': lang[0], 'code': lang[1]}
        if lang['code'] in servicecode2label:
            langname2label[lang['name']] = servicecode2label[lang['code']]
        else:
            print("could not find '{}'".format(lang))


def main():
    init()
    print("In both: {}".format(len(servicecode2label)))
    print("Only CLD-2: {}".format(len(langname2label)))


if __name__ == '__main__':
    main()
