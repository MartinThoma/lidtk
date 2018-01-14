# core modules
import unittest

# internal modules
import lidtk.data.download_documents


class UtilsTest(unittest.TestCase):

    def test_get_wiki_codes(self):
        wiki_codes = lidtk.data.download_documents.get_wiki_codes()
        self.assertIsInstance(wiki_codes, list)

    def test_normalize_data(self):
        normalized = lidtk.data.download_documents.normalize_data('foobar')
        self.assertEqual(normalized, 'foobar')
        normalized = lidtk.data.download_documents.normalize_data('üäöüß')
        self.assertEqual(normalized, 'üäöüß')

    def test_extract_paragraphs(self):
        section = ('== Title ==\n'
                   'This is a text. This is a text. This is a text.'
                   'This is a text. This is a text. This is a text.'
                   'This is a text. This is a text. This is a text.'
                   'This is a text. This is a text. This is a text.\n'
                   'A new test. A new test. A new test. A new test.'
                   'A new test. A new test. A new test. A new test.'
                   'A new test. A new test. A new test. A new test.\n'
                   r'\displaystyle \frac{123}{456} + 3 + 5 + \pi + '
                   ' 3 + 5 + \pi + 3 + 5 + \pi + 3 + 5 + \pi + 3 + 5 + \pi + '
                   ' 3 + 5 + \pi + 3 + 5 + \pi + 3 + 5 + \pi + 3 + 5 + \pi + '
                   )
        paragraphs = lidtk.data.download_documents.extract_paragraphs(section)
        self.assertEqual(len(paragraphs), 2)

    def test_find_pages(self):
        lidtk.data.download_documents.find_pages(lang_wiki='en',
                                                 to_extract=10,
                                                 max_time_s=30,
                                                 verbose=False)
