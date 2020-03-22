# Third party modules
import pkg_resources

# First party modules
import lidtk.utils


def test_make_path_absolute():
    path = lidtk.utils.make_path_absolute(".")
    assert isinstance(path, str)


def test_get_software_info():
    software_info = lidtk.utils.get_software_info()
    assert isinstance(software_info, dict)


def test_get_hardware_info():
    hardware_info = lidtk.utils.get_hardware_info()
    assert isinstance(hardware_info, dict)


def test_load_cfg():
    filepath = pkg_resources.resource_filename("lidtk", "config.yaml")
    cfg = lidtk.utils.load_cfg(filepath)
    assert isinstance(cfg, dict)
