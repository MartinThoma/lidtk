# First party modules
import lidtk.data.download_documents


def test_get_wiki_codes():
    wiki_codes = lidtk.data.download_documents.get_wiki_codes()
    assert isinstance(wiki_codes, list)


def test_normalize_data():
    normalized = lidtk.data.download_documents.normalize_data("foobar")
    assert normalized == "foobar"
    normalized = lidtk.data.download_documents.normalize_data("üäöüß")
    assert normalized == "üäöüß"


def test_extract_paragraphs():
    section = (
        "== Title ==\n"
        "This is a text. This is a text. This is a text."
        "This is a text. This is a text. This is a text."
        "This is a text. This is a text. This is a text."
        "This is a text. This is a text. This is a text.\n"
        "A new test. A new test. A new test. A new test."
        "A new test. A new test. A new test. A new test."
        "A new test. A new test. A new test. A new test.\n"
        r"\displaystyle \frac{123}{456} + 3 + 5 + \pi + "
        r" 3 + 5 + \pi + 3 + 5 + \pi + 3 + 5 + \pi + 3 + 5 + \pi + "
        r" 3 + 5 + \pi + 3 + 5 + \pi + 3 + 5 + \pi + 3 + 5 + \pi + "
    )
    paragraphs = lidtk.data.download_documents.extract_paragraphs(section)
    assert len(paragraphs) == 2


def test_find_pages():
    lidtk.data.download_documents.find_pages(
        lang_wiki="en", to_extract=10, max_time_s=30, verbose=False
    )
