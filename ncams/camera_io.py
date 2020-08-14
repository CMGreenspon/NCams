#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019-2020 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

File I/O functions for cameras.

For more details on the camera data structures and dicts, see help(ncams.camera_tools).
"""

import os
from copy import deepcopy
import pickle

import yaml
import numpy as np

from . import utils


################### NCams config
def config_to_yaml(ncams_config, setup_path=None, setup_filename=None):
    '''Export camera config into a YAML file.

    Arguments:
        ncams_config {dict} -- information about camera configuration. For the full description,
                see help(ncams.camera_tools). This function uses following keys:
            setup_path {string} -- directory where the camera setup is located, including
                config.yaml.
            setup_filename {string} -- config has been loaded from os.path.join(
                setup_path, setup_filename) and/or will be saved into this directory.

    Keyword Arguments:
        output_path {string} -- overrides the directory where the config is saved. (default: {None})
        setup_filename {string} - overrides the filename of the config file. (default:
            {None})
    '''
    serials = ncams_config['serials']

    # the camera objects are not pickleable, need to remove them before copy
    if 'dicts' in ncams_config.keys():
        if 'obj' in ncams_config['dicts'][serials[0]].keys():
            cam_objs = []
            for serial in serials:
                cam_objs.append(ncams_config['dicts'][serial]['obj'])
                del ncams_config['dicts'][serial]['obj']  # not picklable
        else:
            cam_objs = None

        if 'system' in ncams_config.keys():
            system = ncams_config['system']
            del ncams_config['system']
        else:
            system = None
    
        out_dict = deepcopy(ncams_config)
    
        # and then restore
        if cam_objs is not None:
            for serial, cam_obj in zip(serials, cam_objs):
                ncams_config['dicts'][serial]['obj'] = cam_obj
        if system is not None:
            ncams_config['system'] = system
    
    else:
        out_dict = deepcopy(ncams_config)

    # If we want to save everything as a list instead of numpy ndarray:
    out_dict = utils.dict_values_numpy_to_list(out_dict)

    if setup_path is not None:
        out_dict['setup_path'] = setup_path
    if setup_filename is not None:
        out_dict['setup_filename'] = setup_filename

    if not os.path.isdir(out_dict['setup_path']):
        os.mkdir(out_dict['setup_path'])

    filename = os.path.join(out_dict['setup_path'], out_dict['setup_filename'])

    with open(filename, 'w') as yaml_file:
        yaml.dump(out_dict, yaml_file, default_flow_style=False)


def yaml_to_config(filename):
    '''Imports camera config from a YAML file.

    Arguments:
        filename {string} -- filename of the YAML ncams_config file.

    Output:
        ncams_config {dict} -- see help(ncams.camera_tools).
    '''
    with open(filename, 'r') as yaml_file:
        ncams_config = yaml.safe_load(yaml_file)
        
    current_config_path = os.path.join(ncams_config['setup_path'], ncams_config['setup_filename'])
    
    check_filename = False
    if not os.path.exists(current_config_path):
        check_filename = True
        
    if not check_filename and not os.path.samefile(current_config_path, filename):
        check_filename = True
    
    if check_filename:
        print('The setup path in the loaded ncams_config does not match its current location.')
        user_input_str = 'Would you like to overwrite the setup path?\n'
        user_input = input(user_input_str).lower()
        if user_input in ('yes', 'y'):
            (new_path, new_filename) = os.path.split(filename)
            ncams_config['setup_path'] = new_path
            ncams_config['setup_filename'] = new_filename
            print('The workspace variable has been overwritten but the original file has not.',
                  'Use the "config_to_yaml" function to overwrite the original file.')

    return ncams_config


################### Intrinsic calibration
# Single camera:
def intrinsic_to_yaml(filename, camera_calib_dict):
    '''Exports camera calibration info for a single camera into a YAML file.

    Arguments:
        filename {string} -- where to save the calibration dictionary.
        camera_calib_dict {dict} -- info on calibration of a single camera. Sould have following
                keys:
            serial {number} - UID of the camera.
            distortion_coefficients {np.array} -- distortion coefficients for the camera.
            camera_matrix {np.array} -- camera calibration matrifor the camera.
            reprojection_error {number} -- reprojection error for the camera.
    '''
    out_dict = deepcopy(camera_calib_dict)

    out_dict = utils.dict_values_numpy_to_list(out_dict)

    with open(filename, 'w') as yaml_output:
        yaml.dump(out_dict, yaml_output, default_flow_style=False)


def yaml_to_calibration(filename):
    '''Imports camera calibration info for a single camera from a YAML file.

    Arguments:
        filename {string} -- the filename from which to load the calibration.
    Output:
        camera_calib_dict {dict} -- info on calibration of a single camera. Sould have following
                keys:
            serial {number} - UID of the camera.
            distortion_coefficients {np.array} -- distortion coefficients for the camera.
            camera_matrix {np.array} -- camera calibration matrifor the camera.
            reprojection_error {number} -- reprojection error for the camera.
    '''
    with open(filename, 'r') as yaml_file:
        dic = yaml.safe_load(yaml_file)

    dic['camera_matrix'] = np.asarray(dic['camera_matrix'])
    dic['distortion_coefficients'] = np.asarray(dic['distortion_coefficients'])
    dic['reprojection_error'] = np.asarray(dic['reprojection_error'])

    return dic


# Multiple cameras:
def export_intrinsics(intrinsic_config, path=None, filename=None):
    '''Exports camera calibration info for all cameras in the setup into a pickle file.

    Does NOT export the 'dicts' key because of redundancy.

    Arguments:
        intrinsic_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            path {string} -- directory where calibration information is stored. Should be same as
                information in ncams_config.
            filename {string} -- name of the YAML file to store the config in/load from.
    Keyword Arguments:
        path {string} -- overrides the 'path' key in the dictionary and saves in that location.
        filename {string} -- overrides the 'filename' key in the dictionary and saves in that
            location.
    '''
    out_dict = deepcopy(intrinsic_config)

    if 'dicts' in out_dict.keys():
        del out_dict['dicts']

    if path is not None:
        out_dict['path'] = path
    if filename is not None:
        out_dict['filename'] = filename

    if not os.path.isdir(out_dict['path']):
        os.mkdir(out_dict['path'])

    fname = os.path.join(out_dict['path'], out_dict['filename'])
    with open(fname, 'wb') as f:
        pickle.dump(out_dict, f)
        
    print('Intrinsics exported to: "' + fname + '"')


def import_intrinsics(ncams_config):
    '''Imports camera calibration info for all cameras in the setup from a pickle file.

    Reorders the loaded information to adhere to 'serials' in ncams_config.

    Arguments:
        ncams_config {dict} -- information about camera configuration. Should have following keys:
            serials {list of numbers} -- list of camera serials.
            setup_path {string} -- directory where the camera setup is located.
            intrinsic_path {string} -- relative path to where calibration information is stored
                from 'setup_path'.
            intrinsic_filename {string} -- name of the pickle file to store the calibration
                config in/load from.
    Output:
        intrinsic_config {dict} -- information on camera calibration and the results of said
                calibraion. Order of each list MUST adhere to intrinsic_config['serials'] AND
                ncams_config['serials']. Should have following keys:
            serials {list of numbers} -- list of camera serials.
            distortion_coefficients {list of np.arrays} -- distortion coefficients for each camera
            camera_matrices {list of np.arrays} -- camera calibration matrices for each camera
            reprojection_errors {list of numbers} -- reprojection errors for each camera
            path {string} -- directory where calibration information is stored. Should be same as
                information in ncams_config.
            dicts {dict of 'camera_calib_dict's} -- keys are serials, values are
                'camera_calib_dict', see help(ncams.camera_tools).
            filename {string} -- name of the pickle file to store the config in/load from.
    '''
    # Get the path name
    filename = os.path.join(ncams_config['setup_path'], ncams_config['intrinsic_path'],
                            ncams_config['intrinsic_filename'])

    # Load the file
    with open(filename, 'rb') as f:
        _intrinsic_config = pickle.load(f)

    intrinsic_config = deepcopy(_intrinsic_config)
    # we want to keep whatever other info was stored just in case
    intrinsic_config['serials'] = []
    intrinsic_config['distortion_coefficients'] = []
    intrinsic_config['camera_matrices'] = []
    intrinsic_config['reprojection_errors'] = []
    intrinsic_config['dicts'] = {}

    for serial in ncams_config['serials']:
        idx = _intrinsic_config['serials'].index(serial)

        intrinsic_config['serials'].append(
            _intrinsic_config['serials'][idx])
        intrinsic_config['distortion_coefficients'].append(
            _intrinsic_config['distortion_coefficients'][idx])
        intrinsic_config['camera_matrices'].append(
            _intrinsic_config['camera_matrices'][idx])
        intrinsic_config['reprojection_errors'].append(
            _intrinsic_config['reprojection_errors'][idx])
        intrinsic_config['calibration_images'].append(
            _intrinsic_config['calibration_images'][idx])
        intrinsic_config['detected_markers'].append(
            _intrinsic_config['detected_markers'][idx])

        intrinsic_config['dicts'][serial] = {
            'serial': serial,
            'distortion_coefficients': _intrinsic_config['distortion_coefficients'][idx],
            'camera_matrix': _intrinsic_config['camera_matrices'][idx],
            'reprojection_error': _intrinsic_config['reprojection_errors'][idx],
            'detected_markers': _intrinsic_config['detected_markers'][idx],
            'calibration_images': _intrinsic_config['calibration_images'][idx]
        }

    return intrinsic_config


################### Extrinsic Calibration
def export_extrinsics(extrinsic_config, path=None, filename=None):
    '''Exports relative position estimation info for all cameras in the setup into a pickle file.

    Does NOT export the 'dicts' key because of redundancy.

    Arguments:
        extrinsic_config {dict} -- information on estimation of relative position of all
                cameras and the results of said pose estimation. See help(ncams.camera_tools).
                Should have following keys:
            path {string} -- directory where pose estimation information is stored. Should be same
                as information in ncams_config.
            filename {string} -- name of the YAML file to store the config in/load from.
    Keyword Arguments:
        path {string} -- overrides the 'path' key in the dictionary and saves in that location.
        filename {string} -- overrides the 'filename' key in the dictionary and saves in that
            location.
    '''
    out_dict = deepcopy(extrinsic_config)

    if 'dicts' in out_dict.keys():
        del out_dict['dicts']

    if path is not None:
        out_dict['path'] = path
    if filename is not None:
        out_dict['filename'] = filename

    if not os.path.isdir(out_dict['path']):
        os.mkdir(out_dict['path'])

    fname = os.path.join(out_dict['path'], out_dict['filename'])
    with open(fname, 'wb') as f:
        pickle.dump(out_dict, f)
        
    print('Extrinsics exported to: "' + fname + '"')


def import_extrinsics(ncams_config):
    '''Imports camera calibration info for all cameras in the setup from a pickle file.

    Reorders the loaded information to adhere to 'serials' in ncams_config.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            extrinsic_path {string} -- relative path to where pose estimation information is
                stored from 'setup_path'.
            extrinsic_filename {string} -- name of the pickle file to store the pose
                estimation config in/load from.
    Output:
        extrinsic_config {dict} -- information on estimation of relative position of all
                cameras and the results of said pose estimation. Order of each list MUST adhere to
                extrinsic_config['serials'] and ncams_config['serials']. Should
                have following keys:
            serials {list of numbers} -- list of camera serials.
            world_locations {list of np.arrays} -- world locations of each camera.
            world_orientations {list of np.arrays} -- world orientation of each camera.
            path {string} -- directory where pose estimation information is stored. Should be same
                as information in ncams_config.
            filename {string} -- name of the YAML file to store the config in/load from.
    '''
    # Get the path name
    filename = os.path.join(ncams_config['setup_path'], ncams_config['extrinsic_path'],
                            ncams_config['extrinsic_filename'])

    # Load the file
    with open(filename, 'rb') as f:
        _extrinsic_config = pickle.load(f)

    extrinsic_config = deepcopy(_extrinsic_config)
    # we want to keep whatever other info was stored just in case
    extrinsic_config['serials'] = []
    extrinsic_config['world_locations'] = []
    extrinsic_config['world_orientations'] = []
    extrinsic_config['dicts'] = {}

    for serial in ncams_config['serials']:
        idx = _extrinsic_config['serials'].index(serial)

        extrinsic_config['serials'].append(
            _extrinsic_config['serials'][idx])
        extrinsic_config['world_locations'].append(
            _extrinsic_config['world_locations'][idx])
        extrinsic_config['world_orientations'].append(
            _extrinsic_config['world_orientations'][idx])

        extrinsic_config['dicts'][serial] = {
            'world_location': _extrinsic_config['world_locations'][idx],
            'world_orientation': _extrinsic_config['world_orientations'][idx]
        }

    return extrinsic_config


################### General I/O
def load_calibrations(ncams_config):
    '''Safely loads pose estimation and camera calibration from files.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            setup_path {string} -- directory where the camera setup is located.
            intrinsic_path {string} -- directory where calibration information is stored.
            intrinsic_filename {string} -- name of the pickle file to store the calibration
                config in/load from.
            extrinsic_path {string} -- relative path to where pose estimation information is
                stored from 'setup_path'.
            extrinsic_filename {string} -- name of the pickle file to store the pose
                estimation config in/load from.
    Output:
        intrinsic_config {dict} -- see help(ncams.camera_tools), None if not found
        extrinsics_config {dict} -- see help(ncams.camera_tools), None if not found
    '''
    try:
        intrinsics_config = import_intrinsics(ncams_config)
        print('Camera calibration loaded.')
    except FileNotFoundError:
        intrinsics_config = None
        print('No camera calibration file found.')

    try:
        extrinsics_config = import_extrinsics(ncams_config)
        print('Pose estimation loaded.')
    except FileNotFoundError:
        extrinsics_config = None
        print('No pose estimation file found.')

    return (intrinsics_config, extrinsics_config)
