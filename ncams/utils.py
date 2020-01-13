#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Utilities for general use in multiple functions.
"""

import os
import re
from glob import glob
from copy import deepcopy
import yaml
import csv

import numpy as np


def get_file_list(file_extensions, path=None, sort=True):
    '''Returns a list of all filenames with a specific extension.

    Files with extensions .jpg, .jpeg, .png, .bmp are considered images. Searches shell-style for
        <path>/*.<file extension>

    Arguments:
        file_extensions {list of strings} -- file extensions to return, with or without the dot.
            If None or empty, returns all files with extensions.
    Keyword Arguments:
        path {string} -- directory to explore. (default: current directory)
        sort {bool} -- alphanumeric sort the output list (default: {True})
    Output:
        strings {list of strings} -- list of all filenames with specified extension.
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
        strings {list of strings} -- list of all image filenames.
    '''

    return get_file_list(('jpg', 'jpeg', 'png', 'bmp'), path=path, sort=sort)


def iterative_filename(input_path):
    ''' Checks the given path and if the file exists and appends to it if necessary.'''

    filetype = os.path.splitext(os.path.split(input_path)[1])[1]

    output_filename = input_path
    idx = 1
    while os.path.exists(output_filename):
        output_filename = input_path[:-len(filetype)] + '({}){}'.format(idx,filetype)
        idx += 1

    return output_filename


def filter_file_list(list_of_files, list_of_filters):
    ''' Returns a filtered list of the input list for each filter given.

    In cases where images where the images from multiple cameras are stored in
    one folder this provides an easy way of filtering by the serial number.

    Keyword Arguments:
        list_of_images {list of strings} -- The unflitered list of images/files.
        list_of_filters {list of strings} -- The filter keywords.
    Output:
        filtered_lists {list of lists of strings} -- list with filtered sub-lists.
    '''

    filtered_lists = []
    for filt in list_of_filters:
        if not isinstance(filt, str):
            filt = str(filt)

        filtered_list = [fn for fn in list_of_files if filt in fn]
        filtered_lists.append(filtered_list)

    return filtered_lists


def alphanumeric_sort(strings):
    '''Alphanumeric sorter that considers parts of the numerical parts of the string independently.

    For example, 'text9moretext' < 'text10moretext' when using this sorting function.
    Useful for:
        sorting out very high framerate images
        sorting by framerate, because '11'<'9', but 11>9
        recording for more than 999.9999 seconds (in the current format the generic sort does not
            give the desired result).

    Arguments:
        strings {list of strings} -- list of strings (e.g. filenames to be sorted)
    Output:
        strings {list of strings} -- sorted list of strings
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(strings, key=alphanum_key)


def dict_values_numpy_to_list(dic):
    '''Checks each value of a dictionary and converts it to list if it was a np.ndarray or tuple.

    Arguments:
        dic {dict} -- any dictionary. Not immutable.
    Output:
        dic {dict} -- changed dictionary.
    '''
    for key in dic.keys():
        if isinstance(dic[key], np.ndarray):
            dic[key] = dic[key].tolist()
        if isinstance(dic[key], tuple):
            dic[key] = list(dic[key])

    return dic


def export_session_config(session_config, session_path=None, session_filename=None):
    '''Export experimental recording session config into a YAML file.

    Arguments:
        session_config {dict} -- information about session configuration. Mostly user-defined. This
                function uses following keys:
            session_path {string} -- directory where the session setup and data are located,
                including config.yaml.
            session_filename {string} -- config has been loaded from os.path.join(
                session_path, session_filename) and/or will be saved into this directory.

    Keyword Arguments:
        output_path {string} -- overrides the directory where the config is saved. (default: {None})
        session_filename {string} - overrides the session_filename of the config file. (default:
            {None})
    '''
    out_dict = deepcopy(session_config)

    # If we want to save everything as a list instead of numpy ndarray:
    out_dict = dict_values_numpy_to_list(out_dict)

    if session_path is not None:
        out_dict['session_path'] = session_path
    if session_filename is not None:
        out_dict['session_filename'] = session_filename

    if not os.path.isdir(out_dict['session_path']):
        os.mkdir(out_dict['session_path'])

    session_filename = os.path.join(out_dict['session_path'], out_dict['session_filename'])

    with open(session_filename, 'w') as yaml_file:
        yaml.dump(out_dict, yaml_file, default_flow_style=False)


def import_session_config(filename):
    '''Imports session config from a YAML file.

    Arguments:
        filename {string} -- filename of the YAML session_config file.

    Output:
        session_config {dict} -- see help(ncams.utils.export_session_config). Mostly defined by
            user.
    '''
    with open(filename, 'r') as yaml_file:
        session_config = yaml.safe_load(yaml_file)

    return session_config


def compare_labels(reference_csv_path, comparison_csv_path, row_skip=3, col_skip=1, verbose=False):
    ''' Compares the XY coordinates of two CSVs.

    The false positive & negative output variables assume the first csv (reference_csv_path) is
    accurate.

    Arguments:
        reference_csv_path {str} -- the 'ground truth' CSV
        comparison_csv_path {str} -- the CSV to compare against it

    Keyword Arguments:
        row_skip {int} -- how many rows of the CSV to skip. (default: {3})
        col_skip {int} -- how many columns of the CSV to skip. (default: {1})
        verbose {bool} -- print additional information. (default: {False})

    Output:
        bp_error {array} -- euclidian pixel error for each bodypart
        false_positives {array} -- number of false positives for each bodypart
        false_negatives {array} -- number of false negatives for each bodypart
    '''

    # Load the arrays
    csv_arrays = [[], []]
    for ifile, csv_fname in enumerate([reference_csv_path, comparison_csv_path]):
        csv_arrays[ifile] = np.genfromtxt(csv_fname, delimiter=',', skip_header=row_skip)
        if not col_skip is None:
            csv_arrays[ifile] = csv_arrays[ifile][:, col_skip:]

    # Check number of bodyparts - if not the same then csvs aren't comparable
    if np.shape(csv_arrays[0])[1] != np.shape(csv_arrays[1])[1]:
        raise ValueError('Different number of bodyparts in each CSV.')
    num_bp = int(np.shape(csv_arrays[0])[1] / 2)
    n_frames = np.shape(csv_arrays[0])[0]
    total_bps = num_bp*n_frames

    # Get the list of bodyparts
    with open(reference_csv_path, 'r') as f:
        bp_csv = list(csv.reader(f))
    bp_csv = bp_csv[1]
    bodyparts = []
    for bp in range(1, len(bp_csv), 2):
        bodyparts.append(bp_csv[bp])

    # Let's do some counting
    false_positives = np.zeros((1, num_bp))
    false_negatives = np.zeros((1, num_bp))
    bp_error = np.zeros((1, num_bp))
    for bp in range(num_bp):
        xy_error = []
        fps = 0
        fns = 0
        for frame in range(n_frames):
            ref_vals = csv_arrays[0][frame, bp*2:(bp*2)+2]
            comp_vals = csv_arrays[1][frame, bp*2:(bp*2)+2]

            # Evaluate:
            # XY error
            if not np.all(np.isnan(ref_vals)) and not np.all(np.isnan(comp_vals)):
                err = np.sqrt((ref_vals[0] - comp_vals[0])**2 + (ref_vals[1] - comp_vals[1])**2)
                xy_error.append(err)
            # False negative
            elif np.all(np.isnan(ref_vals)) and not np.all(np.isnan(comp_vals)):
                fns += 1
            # False positive
            elif not np.all(np.isnan(ref_vals)) and np.all(np.isnan(comp_vals)):
                fps += 1

        bp_error[0, bp] = np.nanmean(xy_error)
        false_positives[0, bp] = fps
        false_negatives[0, bp] = fns

    print('Mean error: {} pixels \nFalse positives: {}/{} \nFalse negatives: {}/{}\n'.format(
        np.round(np.nanmean(bp_error), 3), int(np.sum(false_positives)), int(total_bps),
        int(np.sum(false_negatives)), int(total_bps)))

    if verbose:
        for bp in range(num_bp):
            print('{}: Error = {}, FPs = {}, FNs = {}'.format(
                bodyparts[bp], np.round(bp_error[0, bp], 3), int(false_positives[0, bp]),
                int(false_negatives[0, bp])))

    return bp_error, false_positives, false_negatives
