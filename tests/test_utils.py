# core modules
import unittest


class UtilsTest(unittest.TestCase):

    def test_make_path_absolute(self):
        from lidtk.utils import make_path_absolute
        make_path_absolute('.')
