"""Download 1000 documents of each language."""

# Core Library modules
import json
import logging
import os
import pickle
import random
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Third party modules
import click
import pkg_resources
import progressbar
import requests
import wikipedia

# First party modules
from lidtk.utils import make_path_absolute

logger = logging.getLogger(__name__)
logging.getLogger("requests").setLevel(logging.WARNING)
INFINITY = float("inf")


@click.command(name="download", help=__doc__)
@click.option("--to_extract", default=1000, show_default=True)
@click.option("--target_dir", default="~/.data/langs/", show_default=True)
def main(to_extract: int, target_dir: str) -> None:
    """
    Extract language data from Wikipedia projects.

    Only projects listed in `languages.csv` are considered.

    Parameters
    ----------
    to_extract : int
    target_dir : str
        Path to a directory where the extracted content will be stored.
    """
    target_dir = make_path_absolute(target_dir)
    for lang in get_wiki_codes():
        logger.info("#" * 80)
        logger.info(lang)
        pickle_filename = os.path.join(target_dir, f"{lang}.pickle")
        if not os.path.exists(pickle_filename):
            paragraphs, used_pages = find_pages(lang, to_extract)
            if len(paragraphs) < to_extract:
                continue
            data = {"paragraphs": paragraphs[:to_extract], "used_pages": used_pages}
            with open(pickle_filename, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_wiki_codes(skip_langs: Optional[List[str]] = None) -> List[Any]:
    """
    Get wikipedia codes to fetch data from.

    Parameters
    ----------
    skip_langs : Optional[List[str]]
        A hack to test things faster.

    Returns
    -------
    wiki_codes : List[Any]
    """
    wiki_languages_filepath = pkg_resources.resource_filename("lidtk", "languages.csv")
    with open(wiki_languages_filepath) as f:
        content = f.read().strip().split("\n")
    content = [el.strip() for el in content]
    if skip_langs is None:
        skip_langs = []
        # skip_langs = ['haw', 'ab', 'pi', 'xal', 'nov', 'kl', 'arc', 'na',
        #               'ki', 'tpi']
        logger.info(f"Skip the following wikipedias: {skip_langs}")
    return [lang for lang in content if lang not in skip_langs]


def normalize_data(paragraph: str) -> str:
    """
    Bring unicode in one form.

    Some symbols can be written in multiple ways.

    Parameters
    ----------
    paragraph : str

    Returns
    -------
    paragraph : str
    """
    paragraph = unicodedata.normalize("NFC", paragraph)
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    return paragraph.strip()


def extract_paragraphs(section: str, min_paragraph_length: int = 140) -> List[str]:
    """
    Extract paragraphs from Wikipedia.

    Parameters
    ----------
    section : str
    min_paragraph_length : int, optional (default: 140)

    Returns
    -------
    paragraphs : list
    """
    paragraphs = []
    for paragraph in section.split("\n"):
        paragraph = normalize_data(paragraph)
        if len(paragraph) < min_paragraph_length:
            continue
        is_math = r"\displaystyle" in paragraph
        if is_literature(paragraph) or is_math:
            continue  # It is something that we want to ignore
        paragraphs.append(paragraph)
    return paragraphs


def is_literature(paragraph: str) -> bool:
    """
    Check if a paragraph is a literature entry.

    Parameters
    ----------
    paragraph : str

    Returns
    -------
    is_literature : bool
    """
    doi_regex = re.compile(r"""(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)""")
    issn_regex = re.compile(r"""ISSN \d+""", re.IGNORECASE)
    vol_regex = re.compile(r"""vol\. [IVCXL\d]+""", re.IGNORECASE)
    return (
        "ISBN" in paragraph
        or bool(doi_regex.search(paragraph))
        or bool(issn_regex.search(paragraph))
        or bool(vol_regex.search(paragraph))
        or "https://" in paragraph
        or "http://" in paragraph
    )


def find_pages(
    lang_wiki: str = "de",
    to_extract: int = 1000,
    max_time_s: int = 4 * 60 * 60,
    verbose: bool = False,
):
    """
    Extract paragraphs from random wikipedia pages of a given language.

    Parameters
    ----------
    lang_wiki : str, optional (default: de)
    to_extract : int, optional (default: 1000)
        Number of paragraphs to be extracted
    max_time_s  : int, optional (default: 4h)
        Maximum time in seconds to run the paragraph extraction.
    verbose : boolen, optional (default: False)

    Returns
    -------
    tuple : extracted_paragraphs, list of source pages
    """
    wikipedia.set_lang(lang_wiki)
    extracted_paragraphs = []  # type: List[str]
    used_pages = set()  # type: Set[str]
    queried = []  # type: List[str]
    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=to_extract)
    t0 = time.time()
    while len(extracted_paragraphs) < to_extract and (time.time() - t0) < max_time_s:
        random_pages = wikipedia.random(pages=10)
        for random_page in random_pages:
            if "/" in random_page:
                # see https://to.wikipedia.org/wiki/Tuʻi_Tonga_Fefine/en
                continue
            parse_page(
                random_page,
                extracted_paragraphs,
                queried,
                used_pages,
                bar,
                verbose,
                to_extract,
            )
    if False and len(extracted_paragraphs) < to_extract:
        apcontinue = ""
        max_reached = False
        while not max_reached and len(extracted_paragraphs) < to_extract:
            out = get_all_page_titles(lang_wiki, apcontinue=apcontinue, max_pages=10)
            page_titles_queue = out["page_titles"]
            max_reached = out["max_reached"]
            apcontinue = out["apcontinue"]
            random.shuffle(page_titles_queue)
            print(f"Loaded {len(page_titles_queue)} pages")
            while len(extracted_paragraphs) < to_extract and len(page_titles_queue) > 0:
                print(len(page_titles_queue))
                page_title, revision_id = page_titles_queue.pop()
                parse_page(
                    page_title, extracted_paragraphs, queried, used_pages, bar, verbose
                )

    bar.update(to_extract)
    bar.finish()
    return extracted_paragraphs, list(used_pages)


def parse_page(
    random_page, extracted_paragraphs, queried, used_pages, bar, verbose, to_extract
):
    """
    Parse a page and add its content to the corpus.

    Parameters
    ----------
    random_page : title
    extracted_paragraphs : list
    queried : list
    used_pages : set
        Prevent reading pages multiple times
    bar : ProgressBar object
    verbose : bool, optional
    to_extract : int
        Number of pages to be extracted.
    """
    try:
        page = wikipedia.page(random_page)
    except:  # noqa, I don't care about erros - just try the next!
        return
    if page.title in queried:
        return
    else:
        queried.append(page.title)
    content = page.content

    if verbose:
        t = page.title
        print(f"\t## {t}")
    content = re.sub("={2,}", "==", content)
    sections = content.split("==")
    for section in sections:
        section = section.strip()
        paragraphs = extract_paragraphs(section)
        extracted_paragraphs += paragraphs
        if len(paragraphs) > 0:
            used_pages.add(page.revision_id)
        bar.update(min(to_extract, len(extracted_paragraphs)))
        if verbose:
            for p in paragraphs:
                p = p
                print([p])
                print("###")


def get_all_page_titles(
    lang: str, apcontinue: Union[None, bool, str] = "", max_pages: float = INFINITY
) -> Dict[str, Any]:
    """
    Get all page titles.

    This recursive function fetches all page titles from a Wikipedia.

    Parameters
    ----------
    lang : str
        e.g. 'de'
    apcontinue : str
    max_pages : int

    Returns
    -------
    results : Dict[str, Any]
        'page_titles' : list
        'apcontinue' : str
        'max_reached' : bool
    """
    page_titles = []  # type: List[Tuple[str, str]]
    apcontinue = True
    q = [
        "list=allpages",
        "aplimit=2",
        "apfilterredir=nonredirects",
        f"apcontinue={apcontinue}",
    ]
    max_reached = False
    while apcontinue and len(page_titles) < max_pages:
        result = query(lang, q)
        page_titles += [(p["title"], p["pageid"]) for p in result["query"]["allpages"]]
        if "continue" not in result:
            print("continue not in result")
            apcontinue = None
            break
        apcontinue = result["continue"]["apcontinue"]
        q[2] = f"apcontinue={apcontinue}"
        if len(page_titles) > max_pages:
            print("max_pages reached")
            max_reached = True
            break
    return {
        "page_titles": page_titles,
        "apcontinue": apcontinue,
        "max_reached": max_reached,
    }


def query(lang, query):
    """
    Send a query to a wikipeda.

    Parameters
    ----------
    lang : str
    query: list

    Returns
    -------
    decoded_response : dict
    """
    query = "&".join(query)
    q = (
        "https://{lang}.wikipedia.org/w/api.php?action=query&{query}"
        "&format=json".format(lang=lang, query=query)
    )
    r = requests.get(q)
    return json.loads(r.text)
