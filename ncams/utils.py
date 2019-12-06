#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Please see AUTHORS for contributors.
https://github.com/CMGreenspon/NCams/blob/master/README.md
Licensed under the Apache License, Version 2.0

Utilities for general use in multiple functions
"""

import os
import re
from glob import glob


def get_file_list(file_extensions, path=None, sort=True):
    '''Returns a list of all filenames with a specific extension.

    Files with extensions .jpg, .jpeg, .png, .bmp are considered images. Searches shell-style for
        <path>/*.<file extension>

    Arguments:
        file_extensions {list} -- file extensions to return, with or without the dot.
            If None or empty, returns all files with extensions.
    Keyword Arguments:
        path {string} -- directory to explore. (default: current directory)
        sort {bool} -- alphanumeric sort the output list (default: {True})
    Output:
        strings {list} -- list of all filenames with specified extension.
    '''
    if file_extensions is None or len(file_extensions) == 0:
        file_extensions = ('*', )
    else:
        file_extensions = [ifx.strip('.') for ifx in file_extensions]

    files = []
    for file_extension in file_extensions:
        if path is None:
            files += glob('*.' + file_extension)
        else:
            files += glob(os.path.join(path, '*.' + file_extension))

    if sort:
        files = alphanumeric_sort(files)

    return files


def get_image_list(path=None, sort=True):
    '''Returns a list of all image filenames.

    Wrapper for 'utilities.get_file_list' with file_extensions defined as:
        ('jpg', 'jpeg', 'png', 'bmp')

    Keyword Arguments:
        path {string} -- directory to explore. (default: current directory)
        sort {bool} -- alphanumeric sort the output list (default: {True})
    Output:
        strings {list} -- list of all image filenames.
    '''

    return get_file_list(('jpg', 'jpeg', 'png', 'bmp'), path=path, sort=sort)


def alphanumeric_sort(strings):
    '''Alphanumeric sorter that considers parts of the numerical parts of the string independently.

    For example, 'text9moretext' < 'text10moretext' when using this sorting function.
    Useful for:
        sorting out very high framerate images
        sorting by framerate, because '11'<'9', but 11>9
        recording for more than 999.9999 seconds (in the current format the generic sort does not
            give the desired result).

    Arguments:
        strings {list} -- list of strings (e.g. filenames to be sorted)
    Output:
        strings {list} -- sorted list of strings
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(strings, key=alphanum_key)
