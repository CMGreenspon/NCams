#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Please see AUTHORS for contributors.
https://github.com/CMGreenspon/NCams/blob/master/README.md
Licensed under the Apache License, Version 2.0
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
        'deeplabcut',
        'moviepy',
        'cv2'],
    author='Charles M Greenspon',
    author_email='charles.greenspon@gmail.com',
    license='Apache 2.0',
    packages=find_packages())
