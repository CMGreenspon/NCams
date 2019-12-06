#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams
"""

from setuptools import setup, find_packages

setup(
    name='ncams',
    version='0.0.1',
    description='Recording, triangulation and mapping using multiple cameras.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/CMGreenspon/NCams',
    install_requires=[
        'moviepy',
        'opencv-contrib-python',
        'reportlab',
        'pyyaml',
        'matplotlib',
        'numpy',
        'scipy'],
    author1='Charles M Greenspon',
    author1_email='charles.greenspon@gmail.com',
    author2='Anton Sobinov',
    author2_email='a.sobinov@gmail.com',
    license='Apache 2.0',
    packages=find_packages())
