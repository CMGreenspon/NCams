#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
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


################### Camera config
def config_to_yaml(camera_config, setup_path=None, setup_filename=None):
    '''Export camera config into a YAML file.

    Arguments:
        camera_config {dict} -- information about camera configuration. For the full description,
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
    serials = camera_config['serials']

    # the camera objects are not pickleable, need to remove them before copy
    if 'obj' in camera_config['dicts'][serials[0]].keys():
        cam_objs = []
        for serial in serials:
            cam_objs.append(camera_config['dicts'][serial]['obj'])
            del camera_config['dicts'][serial]['obj']  # not picklable
    else:
        cam_objs = None

    if 'system' in camera_config.keys():
        system = camera_config['system']
        del camera_config['system']
    else:
        system = None

    out_dict = deepcopy(camera_config)

    # and then restore
    if cam_objs is not None:
        for serial, cam_obj in zip(serials, cam_objs):
            camera_config['dicts'][serial]['obj'] = cam_obj
    if system is not None:
        camera_config['system'] = system

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
        filename {string} -- filename of the YAML camera_config file.

    Output:
        camera_config {dict} -- see help(ncams.camera_tools).
    '''
    with open(filename, 'r') as yaml_file:
        camera_config = yaml.safe_load(yaml_file)

    return camera_config


################### Camera calibration
# Single camera:
def calibration_to_yaml(filename, camera_calib_dict):
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
def export_calibration(calibration_config, path=None, filename=None):
    '''Exports camera calibration info for all cameras in the setup into a pickle file.

    Does NOT export the 'dicts' key because of redundancy.

    Arguments:
        calibration_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            path {string} -- directory where calibration information is stored. Should be same as
                information in camera_config.
            filename {string} -- name of the YAML file to store the config in/load from.
    Keyword Arguments:
        path {string} -- overrides the 'path' key in the dictionary and saves in that location.
        filename {string} -- overrides the 'filename' key in the dictionary and saves in that
            location.
    '''
    out_dict = deepcopy(calibration_config)

    if 'dicts' in out_dict.keys():
        del out_dict['dicts']

    if path is not None:
        out_dict['path'] = path
    if filename is not None:
        out_dict['filename'] = filename

    if not os.path.isdir(out_dict['path']):
        os.mkdir(out_dict['path'])

    with open(os.path.join(out_dict['path'], out_dict['filename']), 'wb') as f:
        pickle.dump(out_dict, f)


def import_calibration(camera_config):
    '''Imports camera calibration info for all cameras in the setup from a pickle file.

    Reorders the loaded information to adhere to 'serials' in camera_config.

    Arguments:
        camera_config {dict} -- information about camera configuration. Should have following keys:
            serials {list of numbers} -- list of camera serials.
            calibration_path {string} -- directory where calibration information is stored.
            calibration_filename {string} -- name of the pickle file to store the calibration
                config in/load from.
    Output:
        calibration_config {dict} -- information on camera calibration and the results of said
                calibraion. Order of each list MUST adhere to calibration_config['serials'] AND
                camera_config['serials']. Should have following keys:
            serials {list of numbers} -- list of camera serials.
            distortion_coefficientss {list of np.arrays} -- distortion coefficients for each camera
            camera_matrices {list of np.arrays} -- camera calibration matrices for each camera
            reprojection_errors {list of numbers} -- reprojection errors for each camera
            path {string} -- directory where calibration information is stored. Should be same as
                information in camera_config.
            dicts {dict of 'camera_calib_dict's} -- keys are serials, values are
                'camera_calib_dict', see help(ncams.camera_tools).
            filename {string} -- name of the pickle file to store the config in/load from.
    '''
    # Get the path name
    filename = os.path.join(camera_config['calibration_path'],
                            camera_config['calibration_filename'])

    # Load the file
    with open(filename, 'rb') as f:
        _calibration_config = pickle.load(f)

    calibration_config = deepcopy(_calibration_config)
    # we want to keep whatever other info was stored just in case
    calibration_config['serials'] = []
    calibration_config['distortion_coefficientss'] = []
    calibration_config['camera_matrices'] = []
    calibration_config['reprojection_errors'] = []
    calibration_config['dicts'] = {}

    for serial in camera_config['serials']:
        idx = _calibration_config['serials'].index(serial)

        calibration_config['serials'].append(
            _calibration_config['serials'][idx])
        calibration_config['distortion_coefficientss'].append(
            _calibration_config['distortion_coefficientss'][idx])
        calibration_config['camera_matrices'].append(
            _calibration_config['camera_matrices'][idx])
        calibration_config['reprojection_errors'].append(
            _calibration_config['reprojection_errors'][idx])

        calibration_config['dicts'][serial] = {
            'serial': serial,
            'distortion_coefficients': _calibration_config['distortion_coefficientss'][idx],
            'camera_matrix': _calibration_config['camera_matrices'][idx],
            'reprojection_error': _calibration_config['reprojection_errors'][idx]
        }

    return calibration_config


################### Pose estimation
def export_pose_estimation(pose_estimation_config, path=None, filename=None):
    '''Exports relative position estimation info for all cameras in the setup into a pickle file.

    Does NOT export the 'dicts' key because of redundancy.

    Arguments:
        pose_estimation_config {dict} -- information on estimation of relative position of all
                cameras and the results of said pose estimation. See help(ncams.camera_tools). Should
                have following keys:
            path {string} -- directory where pose estimation information is stored. Should be same as
                information in camera_config.
            filename {string} -- name of the YAML file to store the config in/load from.
    Keyword Arguments:
        path {string} -- overrides the 'path' key in the dictionary and saves in that location.
        filename {string} -- overrides the 'filename' key in the dictionary and saves in that
            location.
    '''
    out_dict = deepcopy(pose_estimation_config)

    if 'dicts' in out_dict.keys():
        del out_dict['dicts']

    if path is not None:
        out_dict['path'] = path
    if filename is not None:
        out_dict['filename'] = filename

    if not os.path.isdir(out_dict['path']):
        os.mkdir(out_dict['path'])

    with open(os.path.join(out_dict['path'], out_dict['filename']), 'wb') as f:
        pickle.dump(out_dict, f)


def import_pose_estimation(camera_config):
    '''Imports camera calibration info for all cameras in the setup from a pickle file.

    Reorders the loaded information to adhere to 'serials' in camera_config.

    Arguments:
        camera_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            pose_estimation_path {string} -- directory where pose estimation information is stored.
            pose_estimation_filename {string} -- name of the pickle file to store the pose
                estimation config in/load from.
    Output:
        pose_estimation_config {dict} -- information on estimation of relative position of all
                cameras and the results of said pose estimation. Order of each list MUST adhere to
                pose_estimation_config['serials'] and camera_config['serials']. Should
                have following keys:
            serials {list of numbers} -- list of camera serials.
            world_locations {list of np.arrays} -- world locations of each camera.
            world_orientations {list of np.arrays} -- world orientation of each camera.
            path {string} -- directory where pose estimation information is stored. Should be same
                as information in camera_config.
            filename {string} -- name of the YAML file to store the config in/load from.
    '''
    # Get the path name
    filename = os.path.join(camera_config['pose_estimation_path'],
                            camera_config['pose_estimation_filename'])

    # Load the file
    with open(filename, 'rb') as f:
        _pose_estimation_config = pickle.load(f)

    pose_estimation_config = deepcopy(_pose_estimation_config)
    # we want to keep whatever other info was stored just in case
    pose_estimation_config['serials'] = []
    pose_estimation_config['world_locations'] = []
    pose_estimation_config['world_orientations'] = []
    pose_estimation_config['camera_pe_dict'] = {}

    for serial in camera_config['serials']:
        idx = _pose_estimation_config['serials'].index(serial)

        pose_estimation_config['serials'].append(
            _pose_estimation_config['serials'][idx])
        pose_estimation_config['world_locations'].append(
            _pose_estimation_config['world_locations'][idx])
        pose_estimation_config['world_orientations'].append(
            _pose_estimation_config['world_orientations'][idx])

        pose_estimation_config['camera_pe_dict'][serial] = {
            'world_location': _pose_estimation_config['world_locations'][idx],
            'world_orientation': _pose_estimation_config['world_orientations'][idx]
        }

    return pose_estimation_config


################### General I/O
def load_camera_config(camera_config):
    '''Safely loads pose estimation and camera calibration from files.

    Arguments:
        camera_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            calibration_path {string} -- directory where calibration information is stored.
            calibration_filename {string} -- name of the pickle file to store the calibration
                config in/load from.
            pose_estimation_path {string} -- directory where pose estimation information is stored.
            pose_estimation_filename {string} -- name of the pickle file to store the pose
                estimation config in/load from.
    Output:
        calibration_config {dict} -- see help(ncams.camera_tools), None if not found
        pose_estimation_config {dict} -- see help(ncams.camera_tools), None if not found
    '''
    try:
        calibration_config = import_calibration(camera_config)
        print('Camera calibration loaded.')
    except FileNotFoundError:
        calibration_config = None
        print('No camera calibration file found.')

    try:
        pose_estimation_config = import_pose_estimation(camera_config)
        print('Pose estimation loaded.')
    except FileNotFoundError:
        pose_estimation_config = None
        print('No pose estimation file found.')

    return (calibration_config, pose_estimation_config)
