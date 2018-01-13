from setuptools import find_packages
from setuptools import setup

config = {
    'name': 'lidtk',
    'version': '0.1.0',
    'author': 'Martin Thoma',
    'author_email': 'info@martin-thoma.de',
    'maintainer': 'Martin Thoma',
    'maintainer_email': 'info@martin-thoma.de',
    'packages': find_packages(),
    'scripts': ['bin/lidtk'],
    # 'package_data': {'hwrt': ['templates/*', 'misc/*']},
    'platforms': ['Linux'],
    'url': 'https://github.com/MartinThoma/language-identification',
    'license': 'MIT',
    'description': 'Language identification Toolkit',
    'long_description': ("A tookit for language identification."),
    'install_requires': [
        'argparse',
        'PyYAML',
        'wikipedia',
        'cld2-cffi',
    ],
    'tests_require': [
        'pytest>=3.3.2',
        'pytest-cov>=2.5.1',
        'pytest-pep8>=1.0.6',
    ],
    'keywords': ['Machine Learning', 'Data Science'],
    'download_url': 'https://github.com/MartinThoma/language-identification',
    'classifiers': ['Development Status :: 1 - Planning',
                    'Environment :: Console',
                    'Intended Audience :: Developers',
                    'Intended Audience :: Science/Research',
                    'Intended Audience :: Information Technology',
                    'License :: OSI Approved :: MIT License',
                    'Natural Language :: English',
                    'Programming Language :: Python :: 3.5',
                    'Topic :: Scientific/Engineering :: Information Analysis',
                    'Topic :: Software Development',
                    'Topic :: Utilities'],
    'zip_safe': False,
}

setup(**config)
