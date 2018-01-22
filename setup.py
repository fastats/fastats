#!/usr/bin/env python3

import codecs
import importlib.util
from os import path

from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))


def read_utf8(filename):
    """
    Ensure consistent encoding
    """
    with codecs.open(path.join(here, filename), encoding='utf-8') as f:
        return f.read()


long_description = read_utf8('README.md')

# import just the _version module, don't pull in any fastats dependencies
spec = importlib.util.spec_from_file_location(
    '_version', path.join(here, 'fastats', '_version.py')
)
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)

version = version_module.VERSION


setup_kwargs = dict(
    name='fastats',
    version=version,
    description='A pure Python library for benchmarked, scalable numerics '
                'using numba',
    url='https://github.com/fastats/fastats',
    author_email='fastats@googlegroups.com',

    # Represents the body of text which users will see when they visit PyPI
    long_description=long_description,

    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering',

        'Operating System :: OS Independent',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='algorithmics ast linear-algebra numba numerics',
    packages=find_packages(exclude=['tests', 'tests.*']),

    setup_requires=[
        'pytest-runner',    # to enable pytest for setup.py test via setup.cfg
    ],

    install_requires=[
        'numba>=0.36.1',
        'numpy',
        'scipy',
    ],

    tests_require=[
        'hypothesis',
        'pytest',
        'pytest-cov',
        'scikit-learn',
        'statsmodels',
        'setuptools',   # for pkg_resources
    ],

    extras_require={
        'doc': [    # documentation
            'sphinx',
            'sphinx_rtd_theme',
        ],
    },
)


# CI-specific test utilities, e.g. travis, appveyor
setup_kwargs['extras_require']['ci_test'] = (
    setup_kwargs['tests_require']
    + [
        'codecov',
        'httpie',
    ]
)

# All ("development") requirements, including docs generation and tests,
# but no CI-specific ones
setup_kwargs['extras_require']['dev'] = (
    setup_kwargs['tests_require']
    + setup_kwargs['extras_require']['doc']
)


setup(**setup_kwargs)
