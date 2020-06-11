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
    version='0.1.5',
    description='Calibration, pose estimation, and triangulation using multiple cameras.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/CMGreenspon/NCams',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'moviepy',
        'opencv-contrib-python',
        'reportlab',
        'pyyaml',
        'easygui',
        'astropy'],
    author='Charles M. Greenspon',
    author_email='charles.greenspon@gmail.com',
    license='MIT',
    packages=find_packages())
