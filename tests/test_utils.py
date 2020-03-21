# core modules
import pkg_resources
import unittest

# internal modules
import lidtk.utils


class UtilsTest(unittest.TestCase):
    def test_make_path_absolute(self):
        path = lidtk.utils.make_path_absolute(".")
        self.assertIsInstance(path, str)

    def test_get_software_info(self):
        software_info = lidtk.utils.get_software_info()
        self.assertIsInstance(software_info, dict)

    def test_get_hardware_info(self):
        hardware_info = lidtk.utils.get_hardware_info()
        self.assertIsInstance(hardware_info, dict)

    def test_load_cfg(self):
        filepath = pkg_resources.resource_filename("lidtk", "config.yaml")
        cfg = lidtk.utils.load_cfg(filepath)
        self.assertIsInstance(cfg, dict)
