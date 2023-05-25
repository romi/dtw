#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages
from setuptools import setup

short_descr = "Dynamic time warping algorithm(s)."
readme = open('README.md').read()
history = open('HISTORY.rst').read()

pkgs = find_packages('src')

setup_kwds = dict(
    name='dtw',
    version="0.0.1",
    description=short_descr,
    long_description=readme + '\n\n' + history,
    author="Christophe Godin",
    author_email="christophe.godin@inria.fr",
    url='https://gitlab.inria.fr/mosaic/work-in-progress/dtw',
    license='GPL-3.0-or-later',
    zip_safe=False,
    packages=pkgs,
    package_dir={'': 'src'},
    setup_requires=[],
    install_requires=[],
    tests_require=[],
    entry_points={},
    scripts=[
        'src/dtw/bin/romi_compare_to_manual_gt.py',
        'src/dtw/bin/align_csv_database.py',
    ],
    keywords='',
    test_suite='nose.collector',
)

setup(**setup_kwds)
