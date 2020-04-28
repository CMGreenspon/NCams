#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Functions related to estimation of relative positions and orientations of the cameras.

For more details on the camera data structures and dicts, see help(ncams.camera_tools).
"""

import os
import tkinter
from tkinter.filedialog import askopenfilename

import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as mpl_pp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from . import utils
from . import camera_io
from . import camera_tools
from . import image_tools
from . import reconstruction


#################### Board detectors
def charuco_board_detector(ncams_config):
    '''Detects charuco board in all cameras.

    (Should be run after cameras have been calibrated.)
    A general function for bulk identifying all charuco corners across cameras and storing them in
    usable arrays for subsequent pose estimation.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
            extrinsic_path {string} -- relative path to where pose estimation information is
                stored from 'setup_path'.

    Output:
        cam_image_points {list} -- x,y coordinates of identified points
        cam_charuco_ids {list} -- ids of the points
    '''
    # Unpack the dict
    serials = ncams_config['serials']
    if 'dicts' in ncams_config.keys():
        names = [ncams_config['dicts'][serial]['name'] for serial in serials]
    else:
        names = [str(serial) for serial in serials]
        
    extrinsic_path = os.path.join(ncams_config['setup_path'],
                                        ncams_config['extrinsic_path'])

    # Get number of cameras
    num_cameras = len(serials)
    charuco_dict, charuco_board, _ = camera_tools.create_board(ncams_config)

    # Get list of images for each camera
    cam_image_list = []
    num_images = np.zeros((1, num_cameras), dtype=int)
    for icam, name in enumerate(names):
        path_check = os.path.isdir(os.path.join(extrinsic_path, name))
        if path_check is False:
            full_image_list = utils.get_image_list(path=os.path.join(extrinsic_path))
            image_list = [fn for fn in full_image_list if name in fn]
        else:
            image_list = utils.get_image_list(path=os.path.join(extrinsic_path, name))

        num_images[0, icam] = len(image_list)
        cam_image_list.append(image_list)

    # Crucial: each camera must have the same number of images so that we can assume the order is
    # maintained and that they are synced
    if not np.ma.allequal(num_images, np.mean(num_images)):
        raise Exception('Image lists are of unequal size and may not be synced.')

    num_images = num_images[0, 0]
    cam_image_points = []
    cam_charuco_ids = []
    # Look at one synced image across cameras and find the points
    for image in range(num_images):
        im_ids, image_points = [], []  # reset for each image
        for icam, name in enumerate(names):
            # Load the image
            img = cv2.imread(os.path.join(extrinsic_path, name,
                                          cam_image_list[icam][image]))
            # Detect the aruco markers and get IDs
            corners, ids, _ = cv2.aruco.detectMarkers(img, charuco_dict)
            if ids is not None:
                # Find the corners and IDs
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, img, charuco_board)
                if isinstance(charuco_corners, np.ndarray):  # If present then append
                    image_points.append(charuco_corners)
                    im_ids.append(charuco_ids)
                else:  # For formatting/indexing
                    image_points.append([])
                    im_ids.append([])
            else:
                image_points.append([])
                im_ids.append([])
        # Concatenate them to get super list which can be parsed later
        cam_image_points.append(image_points)
        cam_charuco_ids.append(im_ids)

    return cam_image_points, cam_charuco_ids


def checkerboard_detector(ncams_config):
    '''Get all image points and determine which calibration mode is better. ???
        AS: description seems wrong

    Should be run after cameras have been calibrated.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            board_dim: list with the number of checks [height, width]
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
            extrinsic_path {string} -- relative path to where pose estimation information is
                stored from 'setup_path'.

    Output:
        cam_board_logit {list} -- if checkerboard: logical array (num_cameras, num_images)
            indicating in which images each camera detected a checkerboard.
        cam_image_points {list} -- if checkerboard: array of image points (num_cameras, image,
            (x, y))
        pose_strategy {string} -- string indicating which pose estimation strategy is ideal.
    '''
    # Unpack the dict
    serials = ncams_config['serials']
    num_cameras = len(serials) # How many cameras are there
    extrinsic_path = os.path.join(ncams_config['setup_path'],
                                        ncams_config['extrinsic_path'])
    board_dim = ncams_config['board_dim']

    # Begin the checkerboard detection for each camera
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Default criteria
    cam_board_logit = []
    cam_image_points = []

    print('Beginning checkerboard detection.')
    for icam, cam_name in enumerate(names):
        print('- Camera {} of {}.'.format(icam+1, num_cameras))
        cam_image_list = utils.get_image_list(path=os.path.join(extrinsic_path, cam_name))

        # Analyze the images to get checkerboard corners
        image_points = []  # x,y image points
        board_logit = np.zeros((1, len(cam_image_list)), dtype=bool)

        for iimage, image_name in enumerate(cam_image_list):
            img = cv2.imread(image_name, 0) # Load as grayscale
            board_logit[0, iimage], corners = cv2.findChessboardCorners(
                img, (board_dim[0]-1, board_dim[1]-1), None)

            # If a checkerboard was found then append the image points variable for calibration
            if board_logit[0, iimage]:
                corners_refined = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
                image_points.append(corners_refined)
            else:
                image_points.append([]) # To keep consistent with the board_logit list

        # Add exports to list structure
        cam_board_logit.append(board_logit)
        cam_image_points.append(image_points)

    print('* Checkerboard detection complete.')
    return cam_board_logit, cam_image_points


#################### Automated calibration
def multi_camera_pose_estimation(ncams_config, show_poses=True):
    '''[Short description]

    [Long description]

    Arguments:
        []
    Keyword Arguments:
        []
    Output:
        []
    '''
    raise NotImplementedError
    num_cameras = len(ncams_config['camera_names'])
    if num_cameras == 1:
        raise Exception('Only one camera present, pose cannot be calculated with this function.')
        return []


#################### Pose estimation methods
def get_world_pose(image, image_size, charuco_dict, charuco_board, world_points, camera_matrix,
                   cam_distortion_coefficients, ch_ids_to_ignore=None):
    # Get the image points
    # Detect the aruco markers and IDs
    corners, ids, _ = cv2.aruco.detectMarkers(image, charuco_dict)
    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, image, charuco_board)

    # Sort out the charuco markers to identify
    if not ch_ids_to_ignore is None:
        ch_filter_idx = np.isin(charuco_ids, ch_ids_to_ignore)
        charuco_ids = charuco_ids[~ch_filter_idx]
        charuco_corners = charuco_corners[~ch_filter_idx]
    
    # Match to world points
    filtered_world_points = []
    for cid in charuco_ids:
        filtered_world_points.append(world_points[cid, :, :])
    filtered_world_points = np.vstack(filtered_world_points)

    # PnP
    _, cam_orientation, camera_location = cv2.solvePnP(
        filtered_world_points, charuco_corners,
        camera_matrix, cam_distortion_coefficients)

    return camera_location, cam_orientation, charuco_corners, charuco_ids


def one_shot_multi_PnP(ncams_config, intrinsics_config, export_full=True, show_extrinsics=False,
                       inspect = False, ch_ids_to_ignore=None):
    '''Position estimation based on a single frame from each camera.

    Assumes that a single synchronized image was taken where all cameras can see the calibration
    board. Will then use the world points to compute the position of each camera independently.
    This method utilizes the fewest images and points to compute the positions and orientations but
    is also the simplest to implement.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            image_size {(height, width)} -- size of the images captured by the cameras.
            board_dim: list with the number of checks [height, width]
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
            extrinsic_path {string} -- directory where pose estimation information is stored.
        intrinsics_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            distortion_coefficients {list of np.arrays} -- distortion coefficients for each camera
            camera_matrices {list of np.arrays} -- the essential camera matrix for each camera.
            dicts {dict of 'camera_calib_dict's} -- keys are serials, values are
                'camera_calib_dict', see below.
    Keyword Arguments:
        export_full {bool} -- save the pose estimation to a dedicated file. (default: {True})
        show_extrinsics {bool} -- whether or not to call the plot_extrinsics function. (default: False)
        inspect {bool} -- call the inspect extrinsics function to estimate accuracy. (default: False)
        ch_ids_to_ignore {array} -- list of markers that are poorly detected and should be ignored.
            (default: None)
    Output:
        extrinsics_config {dict} -- information on estimation of relative position of all
                cameras and the results of said pose estimation. For more info, see
                help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            world_locations {list of np.arrays} -- world locations of each camera.
            world_orientations {list of np.arrays} -- world orientation of each camera.
            path {string} -- directory where pose estimation information is stored. Should be same
                as information in ncams_config.
            filename {string} -- name of the YAML file to store the config in/load from.
            dicts {dict} -- info on pose estimation of a single camera. 
                Sould have following keys:
                serial {number} -- UID of the camera.
                world_location {np.array} -- world location of the camera.
                world_orientation {np.array} -- world orientation of the camera.
                    
        image_info {dict} -- outputs of the intermediary steps, useful for inspection of results.
            Uses camera serials as keys:
            image_paths {list of strings} -- the paths of the images used
            charuco_corners {list of arrays} -- the x,y coordinates of the detected markers.
            charuco_ids {list of arrays} -- the ID number of the detected corners
    '''
    # Format inputs
    if 'dicts' in ncams_config.keys():
        names = [ncams_config['dicts'][serial]['name'] for serial in ncams_config['serials']]
    else:
        names = ['cam'+str(serial) for serial in ncams_config['serials']]
        
    extrinsic_path = os.path.join(ncams_config['setup_path'],
                                        ncams_config['extrinsic_path'])
    camera_matrices = intrinsics_config['camera_matrices']
    distortion_coefficients = intrinsics_config['distortion_coefficients']
    # Construct the board
    charuco_dict, charuco_board, _ = camera_tools.create_board(ncams_config)
    world_points = camera_tools.create_world_points(ncams_config)
    h, w = ncams_config['image_size']
    # Get the images
    im_list = utils.get_image_list(path=extrinsic_path)
    # Prepare outputs
    world_locations = []
    world_orientations = []
    paths_used = []
    charuco_corners = []
    charuco_ids = []
    
    # Get 
    for icam, name in enumerate(names):
        # Find the correct image
        im_name = [i for i in im_list if name in i]
        # If more than one image contains the camera name ask user to select
        if len(im_name) > 1:
            print('--> Multiple images contain the camera name. Select the correct file for'
                  ' "{}".'.format(name))
            # Select the file
            root = tkinter.Tk()
            root.update()
            im_path = askopenfilename(initialdir=extrinsic_path,
                                      title='select the image for "{}".'.format(name))
            root.destroy()

        else:
            im_path = os.path.join(extrinsic_path, im_name[0])

        world_image = cv2.imread(im_path)

        cam_location, cam_orientation, ch_corners, ch_ids = get_world_pose(
            world_image, (w, h), charuco_dict,  charuco_board, world_points, camera_matrices[icam],
            distortion_coefficients[icam], ch_ids_to_ignore)

        paths_used.append(im_path)
        charuco_corners.append(ch_corners)
        charuco_ids.append(ch_ids)
        
        world_locations.append(cam_location)
        world_orientations.append(cam_orientation)

    # Make the output structure
    dicts = {}
    extrinsics_info = {}
    for icam, serial in enumerate(ncams_config['serials']):
        dicts[serial] = {
            'serial': serial,
            'world_location': world_locations[icam],
            'world_orientation': world_orientations[icam]
        }
        
        extrinsics_info[serial] = {
            'serial': serial,
            'image_path': paths_used[icam],
            'charuco_corners': charuco_corners[icam],
            'charuco_ids': charuco_ids[icam]
            }

    extrinsics_config = {
        'serials': ncams_config['serials'],
        'world_locations': world_locations,
        'world_orientations': world_orientations,
        'path': extrinsic_path,
        'filename': ncams_config['extrinsic_filename'],
        'dicts': dicts,
        'estimate_method': 'one-shot'
    }

    if export_full:
        camera_io.export_extrinsics(extrinsics_config)

    if show_extrinsics:
        plot_extrinsics(extrinsics_config, ncams_config)
        
    if inspect:
        inspect_extrinsics(ncams_config, intrinsics_config, extrinsics_config, extrinsics_info)

    return extrinsics_config, extrinsics_info


def common_pose_estimation(ncams_config, intrinsics_config, cam_image_points, detection_logit,
                           export_full=True, show_extrinsics=False):
    '''Position estimation based on frames from multiple cameras simultaneously.

    If there are sufficient shared world points across all cameras then camera pose can
    be estimated from all of them simultaneously. This allows for more points to be used
    than with the one_shot method.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            reference_camera_serial {number} -- serial number of the reference camera.
            image_size {(height, width)} -- size of the images captured by the cameras.
            board_dim: list with the number of checks [height, width]
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
            extrinsic_path {string} -- relative path to where pose estimation information is
                stored from 'setup_path'.
            extrinsic_filename {string} -- name of the pickle file to store the pose
                estimation config in/load from.
        intrinsics_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            distortion_coefficients {list of np.arrays} -- distortion coefficients for each camera
            camera_matrices {list of np.arrays} -- the essential camera matrix for each camera.
            dicts {dict of 'camera_calib_dict's} -- keys are serials, values are
                'camera_calib_dict', see below.
        cam_image_points {[type]} -- [description]
        detection_logit {[type]} -- [description]
    Keyword Arguments:
        export_full {bool} -- save the pose estimation to a dedicated file. (default: {True})
        show_extrinsics {bool} -- whether or not to call the plot_extrinsics function. (default: False)
    Output:
        extrinsics_config {dict} -- information on estimation of relative position of all
                cameras and the results of said pose estimation. For more info, see
                help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            world_locations {list of np.arrays} -- world locations of each camera.
            world_orientations {list of np.arrays} -- world orientation of each camera.
            path {string} -- directory where pose estimation information is stored. Should be same
                as information in ncams_config.
            filename {string} -- name of the YAML file to store the config in/load from.
            dicts {dict} -- pose estimation of a single camera. Sould have following
                    keys:
                serial {number} -- UID of the camera.
                world_location {np.array} -- world location of the camera.
                world_orientation {np.array} -- world orientation of the camera.
    '''
    num_cameras = len(ncams_config['serials'])
    camera_matrices = intrinsics_config['camera_matrices']
    distortion_coefficients = intrinsics_config['distortion_coefficients']

    # Determine the reference camera
    ireference_cam = ncams_config['serials'].index(ncams_config['reference_camera_serial'])
    cam_idx = np.arange(0, num_cameras)
    secondary_cam = cam_idx[cam_idx != ireference_cam][0] # Get the indices of a secondary camera
    # Get the world points
    world_points = camera_tools.create_world_points(ncams_config)
    corner_idx = np.arange(0, len(world_points))
    
    if ncams_config['board_type'] == 'charuco':
        # Get all the points shared across cameras
        filtered_object_points = []
        filtered_image_points = [[] for icam in range(num_cameras)]
        common_point_counter = 0
        for cip, detl in zip(cam_image_points, detection_logit):
            # Empty array corresponding to each camera and point
            point_logit = np.zeros((len(world_points), num_cameras), dtype=bool)
            for icam in range(num_cameras):
                temp = detl[icam]  # Get the array specific to the camera and image
                if isinstance(temp, np.ndarray):
                    for corner in temp:  # For each detected corner
                        point_logit[int(corner), icam] = True
            sum_point_logit = np.sum(point_logit.astype(int), 1)

            # Find which points are shared across all cameras
            common_points = sum_point_logit == num_cameras
            common_point_counter += np.sum(common_points)
            if np.sum(common_points) >= 6:
                # Append only those points
                filtered_object_points.append(world_points[common_points, :].astype('float32'))
                for icam in range(num_cameras):
                    temp_corners = np.zeros((np.sum(common_points), 2), dtype=float)
                    temp_ids = detl[icam]
                    temp_points = cip[icam]
                    running_idx = 0
                    for corner in corner_idx[common_points]:  # Only append
                        idx = int(np.where(temp_ids == corner)[0])
                        temp_corners[running_idx, :] = temp_points[idx, :, :]
                        running_idx += 1
                    filtered_image_points[icam].append(temp_corners.astype('float32'))

    elif ncams_config['board_type'] == 'checkerboard':
        raise NotImplementedError
    
    if common_point_counter < 50:
        print('Insufficent matching points for common pose estimation, consider sequential stereo.')
        return []

    # Get the optimal matrices and undistorted points for the reference and secondary cam
    undistorted_points = []
    for icam in [ireference_cam, secondary_cam]:
        undistorted_points.append(cv2.undistortPoints(
            np.vstack(filtered_image_points[icam]), camera_matrices[icam],
            distortion_coefficients[icam]))

    # Perform the initial stereo calibration
    stereo_calib_flags = 0
    stereo_calib_flags |= cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.001)

    reprojection_error, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        filtered_object_points, filtered_image_points[ireference_cam],
        filtered_image_points[secondary_cam],
        camera_matrices[ireference_cam], distortion_coefficients[ireference_cam],
        camera_matrices[secondary_cam], distortion_coefficients[secondary_cam],
        (ncams_config['image_size'][1], ncams_config['image_size'][0]),
        None, None, None, criteria, stereo_calib_flags)

    if reprojection_error > 1:
        print('Poor initial stereo-calibration. Subsequent pose estimates may be innacurate.')

    # Make projection matrices
    # This keeps the reference frame to that of the primary camera
    projection_matrix_primary = camera_tools.make_projection_matrix(
        camera_matrices[ireference_cam], np.identity(3), np.zeros((3, 1)))
    # Make the secondary projection matrix from stereo-calibration
    projection_matrix_secondary = camera_tools.make_projection_matrix(
        camera_matrices[secondary_cam], R, T)

    # Triangulate those same points
    triangulated_points_norm = cv2.triangulatePoints(
        projection_matrix_primary, projection_matrix_secondary,
        undistorted_points[0], undistorted_points[1])
    # Normalize:
    triangulated_points = triangulated_points_norm[:3, :] / np.transpose(
        np.repeat(triangulated_points_norm[3, :], 3).reshape((-1, 3)))

    world_orientations = []
    world_locations = []
    # Get pose from the triangulated points for all cameras
    for icam in range(num_cameras):
        _, rvec, tvec = cv2.solvePnP(
            np.transpose(triangulated_points), np.vstack(filtered_image_points[icam]),
            camera_matrices[icam], distortion_coefficients[icam])
        world_orientations.append(rvec)
        world_locations.append(tvec)

    # Make the output structure
    dicts = {}
    for icam, serial in enumerate(ncams_config['serials']):
        dicts[serial] = {
            'serial': serial,
            'world_location': world_locations[icam],
            'world_orientation': world_orientations[icam]
        }

    extrinsics_config = {
        'serials': ncams_config['serials'],
        'world_locations': world_locations,
        'world_orientations': world_orientations,
        'path': ncams_config['extrinsic_path'],
        'filename': ncams_config['extrinsic_filename'],
        'dicts': dicts,
        'estimate_method': 'common'
    }

    if export_full:
        camera_io.export_extrinsics(extrinsics_config)
        
    if show_extrinsics:
        plot_extrinsics(extrinsics_config, ncams_config)

    return extrinsics_config


def sequential_pose_estimation(ncams_config, intrinsics_config, cam_image_points, daisy_chain=True,
                               max_links=3, matching_threshold=250, export_full=True,
                               show_extrinsics=False):
    ''' Build a network of stereo-calibrations by iteratively calibrating cameras whilst maintaining
    world origin.

    If not all cameras share a single view but at least have overlapping views then the relative 
    translations and rotations can be daisy-chained. This function takes the reference camera as the
    seed and attempts to calibrate all cameras with it. Attempts are then made to calibrate the 
    remaining cameras with the newly calibrated ones. This occurs iteratively until all have been
    added to the chain.

    Arguments:
        []
    Keyword Arguments:
        daisy_chain {bool} -- whether or not to enable daisy chaining. If False then will only 
            perform stereo-calibration with the reference camera and any cameras that cannot be
            calibrated with it are ignored. (default: True)
        max_links {int} -- how many edges can be between the reference camera and any other camera.
            Equivalent to the number of iterations over which the graph will be built. May need to
            be increased to the number of cameras if there is very little overlap. It is likely that
            more edges produce cumulative error. (default: 3)
        matching_threshold {int} -- the minim number of points to use when attempting to stereo-
            calibrate. Lower thresholds may allow fewer edges to be used but may also result in lower
            quality calibrations. (default: 250)
        export_full {bool} -- save the pose estimation to a dedicated file. (default: True)
        show_extrinsics {bool} -- whether or not to call the plot_extrinsics function. (default: False)
    Output:
        extrinsics_config {dict} -- information on estimation of relative position of all
                cameras and the results of said pose estimation. For more info, see
                help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            world_locations {list of np.arrays} -- world locations of each camera.
            world_orientations {list of np.arrays} -- world orientation of each camera.
            path {string} -- directory where pose estimation information is stored. Should be same
                as information in ncams_config.
            filename {string} -- name of the YAML file to store the config in/load from.
            dicts {dict} -- pose estimation of a single camera. Sould have following
                    keys:
                serial {number} -- UID of the camera.
                world_location {np.array} -- world location of the camera.
                world_orientation {np.array} -- world orientation of the camera.

    '''
    num_images = len(cam_image_points)
    
    num_cameras = len(ncams_config['serials'])
    camera_matrices = intrinsics_config['camera_matrices']
    distortion_coefficients = intrinsics_config['distortion_coefficients']
    
    # Get the world points
    world_points = camera_tools.create_world_points(ncams_config)
    corner_idx = np.arange(0, len(world_points))
    
    # Determine the reference camera
    ireference_cam = ncams_config['serials'].index(ncams_config['reference_camera_serial'])
    cam_idx = np.arange(0, num_cameras)
    isecondary_cams = [i for i in cam_idx if i!=ireference_cam]
    
    # Prepare outputs
    world_locations = [[] for _ in range(num_cameras)]
    world_orientations = [[] for _ in range(num_cameras)]
    
    world_locations[ireference_cam] = np.zeros((3,1))
    world_orientations[ireference_cam] = np.zeros((3,1))
    
    calibrated_cameras = np.zeros((len(cam_idx),1), dtype=bool)
    calibrated_cameras[ireference_cam] = True
    
    # Iterate through images and cameras to make a large array of points that can be matched
    point_detection_bool = np.zeros((len(corner_idx), num_images, num_cameras), dtype=bool)
    for cam in range(num_cameras):
        for image in range(num_images):
            cids = cam_charuco_ids[image][cam]
            ips = cam_image_points[image][cam]
            for cid, ip in zip(cids, ips):
                point_detection_bool[cid, image, cam] = True
        
    # Find overlapping points for each pair of cameras
    n_matching_points = np.zeros((len(cam_idx), len(cam_idx)))
    n_matching_points.fill(np.nan)
    for i in cam_idx:
        for j in cam_idx:
            if i == j: # Can't use own points
                continue
            if np.isnan(n_matching_points[j,i]): # Only check if the inverse has not already been done
                # Get point logits for each camera
                itemp_points = point_detection_bool[:,:,i]
                jtemp_points = point_detection_bool[:,:,j]
                # Find maching
                matching_points = np.logical_and(itemp_points, jtemp_points)
                # Remove images with fewer than 6 points
                matching_points_per_images = np.sum(matching_points, axis=0)
                sufficient_points = np.where(matching_points_per_images >= 6)[0]
                # Allocate sum to array
                n_matching_points[i,j] = np.sum(np.sum(matching_points_per_images[sufficient_points]))
            else:
                n_matching_points[i,j] = n_matching_points[j,i]
    
    # First we try and stereo-calibrate every camera with the primary camera
    icam1 = ireference_cam
    print('Calibrating with reference camera: '+ str(icam1))
    for icam2 in isecondary_cams:
        if n_matching_points[icam1, icam2] < matching_threshold:
            continue
    
        print('\tCamera '+ str(icam2))
        
        matching_points = np.squeeze(np.logical_and(
            point_detection_bool[:,:,icam1], point_detection_bool[:,:,icam2]))
        
        # Collect the necessary world and image points
        matching_world_points, matching_image_points1, matching_image_points2 = [],[],[]
        for im in range(num_images):
            matching_ids = np.where(matching_points[:,im])[0]
            if len(matching_ids) < 6: # CV2 requires at least 6 points per image, not sure why
                continue
            # Allocate empty arrays
            temp_ip1 = np.zeros((len(matching_ids),1,2))
            temp_ip2 = np.zeros((len(matching_ids),1,2))
            # Format points
            for ic, ch_id in enumerate(matching_ids):
                # Find the correct corner for the camera and allocate it
                ip1_idx = np.where(cam_charuco_ids[im][icam1] == ch_id)[0][0]
                temp_ip1[ic,:,:] = cam_image_points[im][icam1][ip1_idx]
                ip2_idx = np.where(cam_charuco_ids[im][icam2] == ch_id)[0][0]
                temp_ip2[ic,:,:] = cam_image_points[im][icam2][ip2_idx]
            # Append and force float32 otherwise throws errors
            matching_world_points.append(world_points[matching_ids,:].astype('float32'))
            matching_image_points1.append(temp_ip1.astype('float32'))
            matching_image_points2.append(temp_ip2.astype('float32'))
        
        stereo_calib_flags = 0
        # We already know the intrinsics (hopefully) and this allows for fewer points to be used
        stereo_calib_flags |= cv2.CALIB_FIX_INTRINSIC 
        criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.001)
        # Perform stereo calibration with the reference camera
        reprojection_error, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            matching_world_points, matching_image_points1, matching_image_points2,
            camera_matrices[icam1], distortion_coefficients[icam1],
            camera_matrices[icam2], distortion_coefficients[icam2],
            (ncams_config['image_size'][1], ncams_config['image_size'][0]),
            None, None, None, criteria, stereo_calib_flags)
    
        R = cv2.Rodrigues(R)[0] # Convert to vector form for consistency
        # Allocate to outputs
        world_locations[icam2] = T
        world_orientations[icam2] = R
        
        calibrated_cameras[icam2] = True
    
    # Daisy chain any remaining cameras
    for l in range(max_links):
        print('Link '+ str(l + 1) + ':')
        
        linkable_matching_points = np.zeros((len(cam_idx), len(cam_idx)))
        linkable_matching_points.fill(np.nan)
        for (i,t) in enumerate(calibrated_cameras):
            if t[0]:
                linkable_matching_points[i,:] = n_matching_points[i,:]
        
        if any(~calibrated_cameras):
            remaining_camera_idx = [i for (i,t) in enumerate(calibrated_cameras) if not t[0]]
            for icam2 in remaining_camera_idx:
                max_points = int(np.nanmax(linkable_matching_points[:,icam2]))
                if max_points < matching_threshold:
                    continue
                
                icam1 = np.where(linkable_matching_points[:,icam2] == max_points)[0][0]
                print('\tCamera '+ str(icam1) + ':' + str(icam2))
            
                matching_points = np.squeeze(np.logical_and(
                point_detection_bool[:,:,icam1], point_detection_bool[:,:,icam2]))
            
                # Collect the necessary world and image points
                matching_world_points, matching_image_points1, matching_image_points2 = [],[],[]
                for im in range(num_images):
                    matching_ids = np.where(matching_points[:,im])[0]
                    if len(matching_ids) < 6: # CV2 requires at least 6 points per image, not sure why
                        continue
                    # Allocate empty arrays
                    temp_ip1 = np.zeros((len(matching_ids),1,2))
                    temp_ip2 = np.zeros((len(matching_ids),1,2))
                    # Format points
                    for ic, ch_id in enumerate(matching_ids):
                        # Find the correct corner for the camera and allocate it
                        ip1_idx = np.where(cam_charuco_ids[im][icam1] == ch_id)[0][0]
                        temp_ip1[ic,:,:] = cam_image_points[im][icam1][ip1_idx]
                        ip2_idx = np.where(cam_charuco_ids[im][icam2] == ch_id)[0][0]
                        temp_ip2[ic,:,:] = cam_image_points[im][icam2][ip2_idx]
                    # Append and force float32 otherwise throws errors
                    matching_world_points.append(world_points[matching_ids,:].astype('float32'))
                    matching_image_points1.append(temp_ip1.astype('float32'))
                    matching_image_points2.append(temp_ip2.astype('float32'))
            
                stereo_calib_flags = 0
                # We already know the intrinsics (hopefully) and this allows for fewer points to be used
                stereo_calib_flags |= cv2.CALIB_FIX_INTRINSIC 
                criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.001)
                # Perform stereo calibration with the daisy-chain camera
                reprojection_error, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
                    matching_world_points, matching_image_points1, matching_image_points2,
                    camera_matrices[icam1], distortion_coefficients[icam1],
                    camera_matrices[icam2], distortion_coefficients[icam2],
                    (ncams_config['image_size'][1], ncams_config['image_size'][0]),
                    None, None, None, criteria, stereo_calib_flags)
                
                R = cv2.Rodrigues(R)[0] # Must be in vector format for composeRT
        
                # Adjust the computed R&T relative to the daisy chained camera
                relR, relT = cv2.composeRT(world_orientations[icam1], world_locations[icam1], R, T)[:2]
                # Allocate relative values
                world_locations[icam2] = relT
                world_orientations[icam2] = relR
                calibrated_cameras[icam2] = True
            
    dicts = {}
    for icam, serial in enumerate(ncams_config['serials']):
        dicts[serial] = {
            'serial': serial,
            'world_location': world_locations[icam],
            'world_orientation': world_orientations[icam]
        }
    
    extrinsics_config = {
        'serials': ncams_config['serials'],
        'world_locations': world_locations,
        'world_orientations': world_orientations,
        'path': ncams_config['extrinsic_path'],
        'filename': ncams_config['extrinsic_filename'],
        'dicts': dicts,
        'estimate_method': 'stereo-sequential'
    }
    
    
    if show_extrinsics:
        ncams.camera_pose.plot_extrinsics(extrinsics_config, ncams_config)
    
    if export_full:
        camera_io.export_extrinsics(extrinsics_config)

def adjust_calibration_origin(world_rotation_vector, world_translation_vector,
                              relative_rotations, relative_translations):
    '''Adjusts orientations and locations based on world rotation and translation.

    If the camera setup is thus that the desired world origin cannot be observed by all cameras
    but you wish to have the coordinate frame be relative to the world origin (or any other origin)
    then the values can be updated with this function. This is particularly useful for sequential
    pose estimates or any generic stereo-calibration.

    Arguments:
        world_rotation_vector {np.array} -- The rotation vector for the reference camera
        world_translation_vector {np.array} -- The translation vector for the reference camera
        relative_rotations {list of 'np.array's} -- List of rotations in the original coordinate frame
        relative_translations {list of 'np.array's} -- List of translations in the original coordinate frame

    Output:
        adjusted_rotation_vectors {list of np.array} -- rotations in space of the world
        adjusted_translation_vectors {list of np.array} -- locations in space of the world
    '''
    adjusted_rotation_vectors = []
    adjusted_translation_vectors = []

    # Format rotation for composeRT
    if world_rotation_vector.shape == (3, 3):
        world_rotation_vector = cv2.Rodrigues(world_rotation_vector)[0]

    for rel_rot, rel_trans in zip(relative_rotations, relative_translations):
        sec_r_vec = rel_rot
        # Format rotation for composeRT
        if sec_r_vec.shape == (3, 3):
            sec_r_vec = cv2.Rodrigues(sec_r_vec)[0]

        adjusted_orientation, adjusted_location = cv2.composeRT(
            world_rotation_vector, world_translation_vector, sec_r_vec, rel_trans)[:2]

        adjusted_rotation_vectors.append(adjusted_orientation)
        adjusted_translation_vectors.append(adjusted_location)

    return adjusted_rotation_vectors, adjusted_translation_vectors


#################### Pose assessement functions
def inspect_extrinsics(ncams_config, intrinsics_config, extrinsics_config, extrinsics_info,
                            error_threshold=0.1, world_points=None):
    ''' Examines the outputs of the pose estimate (currently only supports one_shot_multi_PnP) to
    provide metrics pertaining to the system accuracy (triangulation and 2D reprojection accuracy).

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). Must have following keys:
            serials {list of numbers} -- list of camera serials.
            board_dim {list} -- size of the board used for the pose estimation
            board_type {str} -- though assumed to be a charucoboard this must still be present.
            check_size {int} -- size of the checks of the charucoboard
        intrinsics_config {dict} -- dictionary containing the intrinsic parameters of each camera.
            camera_matrices {list of arrays} -- 3x3 camera matrix for each camera
            distortion_coefficients {list of arrays} -- distortion coefficients for each camera
        extrinsics_config {dict} -- dictionary with the extrinsic parameters.
            world_locations {list of arrays} -- translation vectors
            world_orientations {list of arrays} -- rotation matrices-
        extrinsics_info {dict} -- secondary output of the one_shot_multi_PnP function, contains a dict 
            for each camera (serials matching those in ncams_config) with:
                serial {int} -- serial number for the camera (redundant)
                image_path {str} -- full path to the image used for pose estimation
                charuco_corners {array} -- xy coordinates of each detected corner in the image
                charuco_ids {array} -- id of each corner (relates to the constructed board)
    Keyword Arguments:
        error_threshold {double} -- threshold for triangulation error considered too high
            (default = 0.1)
        world_points {array} -- if the user is not creating world points with the built in function
            then they can create their own reference world points to use for calculations
            (default = None)
    Output:
        This function primarily outputs plots for inspection but will also update the extrinsics_info 
        dictionary. 3D & 2D error values will also be printed to the console.
        The entire dictionary will have a 'pose_accuracy' key added to it containing:
            'bad_points' {list} -- indices of world points that have triangulation errors greater
                than the error threshold
            'reprojection_error' {array} -- average reprojection error between the world points
                and the detected charuco corners
            'triangulation_error' {array} -- euclidian error for each marker and the triangulated
                point if more than two cameras detected it.
        Each cameras dictionary will also be appended with:
            'reprojection_error' {array} -- the reprojection error for each charuco corner detected.
    '''
    
    # Unpack everything
    serials = ncams_config['serials']
    if 'dicts' in ncams_config.keys():
        names = [ncams_config['dicts'][serial]['name'] for serial in ncams_config['serials']]
    else:
        names = ['cam'+str(serial) for serial in ncams_config['serials']]
    num_cameras = len(serials)
    camera_matrices = intrinsics_config['camera_matrices']
    distortion_coefficients = intrinsics_config['distortion_coefficients']
    world_locations = extrinsics_config['world_locations']
    world_orientations = extrinsics_config['world_orientations']
    # Use world_points as ground truth
    if world_points is None: # Allows for user to pass arbitrary world points if desired
        world_points = camera_tools.create_world_points(ncams_config)
    # Make projection matrices for triangulation
    projection_matrices = []
    optimal_matrices = []
    for icam in range(num_cameras):
        projection_matrices.append(camera_tools.make_projection_matrix(
            camera_matrices[icam], world_orientations[icam], world_locations[icam]))
        optimal_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrices[icam], distortion_coefficients[icam],
                (ncams_config['image_size'][1], ncams_config['image_size'][0]), 1,
                (ncams_config['image_size'][1], ncams_config['image_size'][0]))
        optimal_matrices.append(optimal_matrix)
    
    # Triangulate the points - 3d error
    n_world_points = np.shape(world_points)[0]
    triangulated_points = np.empty((n_world_points,3))
    triangulated_points.fill(np.nan)
    
    for p in range(np.shape(world_points)[0]):
        cam_idx = np.zeros((num_cameras,), dtype=bool)
        coords_2d = []
        ch_projection_matrices = []
        for icam, serial in enumerate(serials): # For each camera see if the ch_id is present
            ch_idx = np.where(extrinsics_info[serial]['charuco_ids'] == p)[0]
            if np.shape(ch_idx) == (1,): # If present then undistort the point and add projection mat
                cam_idx[icam] = True
                distorted_point = extrinsics_info[serial]['charuco_corners'][ch_idx,0,:]
                undistorted_point = np.reshape(
                    cv2.undistortPoints(distorted_point, camera_matrices[icam],
                    distortion_coefficients[icam], None, P=camera_matrices[icam]), (1,2))
                coords_2d.append(undistorted_point)
                ch_projection_matrices.append(projection_matrices[icam])
        
        if len(coords_2d) > 1:
            triangulated_points[p,:] = reconstruction.triangulate(coords_2d, ch_projection_matrices)
        
    # Compute the euclidian error for each point
    triang_error = np.zeros((n_world_points,1))
    for p in range(n_world_points):
        triang_error[p] = np.sqrt(np.sum([(a - b) ** 2 for a, b in zip(world_points[p,0,:],
                                                                       triangulated_points[p, :])]))  
    # Find any points with concerningly high errors
    error_boolean = triang_error > error_threshold
    extrinsics_info['pose_accuracy'] = {'bad_points': [], 'triangulation_error': '',
                                  'reprojection_error': ''}
    if np.sum(error_boolean) > 0:
        print('\tSome markers have not triangulated well.\n')
        error_idx = np.where(error_boolean == True)[0]
        extrinsics_info['pose_accuracy']['bad_points'] = error_idx
        
    mean_3d_error = np.round(np.nanmean(triang_error),3)
    median_3d_error = np.round(np.nanmedian(triang_error),3)
    extrinsics_info['pose_accuracy']['triangulation_error'] = triang_error
    print('Mean/median 3D error = {}, {} {}'.format(mean_3d_error, median_3d_error,
                                                    ncams_config['world_units']))
    
    fig = mpl_pp.figure()
    fig.canvas.set_window_title('NCams: Charucoboard Triangulations')
    ax = fig.gca(projection='3d')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_zlabel('Z')
    for p in range(n_world_points):
        ax.scatter(world_points[p,0,0],world_points[p,0,1],world_points[p,0,2], color='b')
        ax.text(world_points[p,0,0], world_points[p,0,1], world_points[p,0,2], str(p), color='b')
        if not any(np.isnan(triangulated_points[p,:])):
            ax.scatter(triangulated_points[p,0],triangulated_points[p,1],triangulated_points[p,2],
                       color='r')
            ax.text(triangulated_points[p,0], triangulated_points[p,1], triangulated_points[p,2],
                    str(p), color='r')
    
    # Back project the points for measuring 2d error
    projected_world_points = []
    reprojection_error = []
    for icam, serial in enumerate(serials):
        # Get the points to compare
        p_idx = extrinsics_info[serial]['charuco_ids']
        world_points_to_project = np.vstack([p for ip, p in enumerate(world_points) if ip in p_idx])
        # Project them to 2d and save for later
        world_points_2d = np.squeeze(
            cv2.projectPoints(world_points_to_project, world_orientations[icam],
                              world_locations[icam], camera_matrices[icam],
                              distortion_coefficients[icam])[0])
        projected_world_points.append(world_points_2d)
        # Measure the error
        detected_points = np.squeeze(extrinsics_info[serial]['charuco_corners'])
        reproj_error = np.zeros((len(p_idx),1))
        for p in range(len(p_idx)):
            reproj_error[p] = np.sqrt(np.sum([(a - b) ** 2 for a, b in zip(world_points_2d[p,:], 
                                                                         detected_points[p,:])]))
        reprojection_error.append(np.nanmean(reproj_error))
        extrinsics_info[serial]['reprojection_error'] = reproj_error
        
    reprojection_error = np.vstack(reprojection_error)
    extrinsics_info['pose_accuracy']['reprojection_error'] = reprojection_error
    mean_2d_error = np.round(np.nanmean(reprojection_error),3)
    median_2d_error = np.round(np.nanmedian(reprojection_error),3)
    print('Mean/median 2D error = {}, {} pixels'.format(mean_2d_error, median_2d_error))
        
    # Show the ground truth reprojections vs identified locations
    num_vert_plots = int(np.floor(np.sqrt(num_cameras)))
    num_horz_plots = int(np.ceil(num_cameras/num_vert_plots))
    fig, axs = mpl_pp.subplots(num_vert_plots, num_horz_plots, squeeze=False)
    fig.canvas.set_window_title('NCams: Charucoboard Reprojections')
    
    for icam, serial in enumerate(serials):
        # Get the correct axis
        vert_ind = int(np.floor(icam / num_horz_plots))
        horz_ind = icam - num_horz_plots * vert_ind
        # Load and plot the image
        img = matplotlib.image.imread(extrinsics_info[serial]['image_path'])
        axs[vert_ind, horz_ind].imshow(img)
        
        # Overlay the matching world and detected points
        axs[vert_ind, horz_ind].scatter(projected_world_points[icam][:,0],
                                        projected_world_points[icam][:,1],
                                        color='b')
        axs[vert_ind, horz_ind].scatter(extrinsics_info[serial]['charuco_corners'][:,0,0],
                                        extrinsics_info[serial]['charuco_corners'][:,0,1],
                                        color='r')
        # Clean up the graph
        axs[vert_ind, horz_ind].set_title(names[icam])
        axs[vert_ind, horz_ind].set_xticks([])
        axs[vert_ind, horz_ind].set_yticks([])
    

def plot_extrinsics(extrinsics_config, ncams_config, scale_unit=None):
    '''Creates a plot showing the location and orientation of all cameras.

    Creates a plot showing the location and orientation of all cameras given based on translation
    and rotation vectors. If your cameras are very close together or far apart you can change the
    scaling factor as necessary.

    Arguments:
        extrinsics_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            world_locations {list of np.arrays} -- world locations of each camera.
            world_orientations {list of np.arrays} -- world orientation of each camera.
    '''
    world_locations = extrinsics_config['world_locations']
    world_orientations = extrinsics_config['world_orientations']
    serials = extrinsics_config['serials']
    num_cameras = len(serials)
    
    if scale_unit is None:
        if ncams_config['world_units'] == 'mm':
            scale_unit = 100
        elif ncams_config['world_units'] == 'cm':
            scale_unit = 10
        elif ncams_config['world_units'] == 'dm':
            scale_unit = 1
        elif ncams_config['world_units'] == 'm':
            scale_unit = 0.1
    
    # Create a figure with axes
    fig = mpl_pp.figure()
    fig.canvas.set_window_title('NCams: Camera Extrinsics')
    ax = fig.gca(projection='3d')

    # Keep the verts for setting the axes later
    cam_verts = [[] for _ in range(num_cameras)]
    for icam in range(num_cameras):
        # Get the vertices to plot appropriate to the translation and rotation
        cam_verts[icam], cam_center = create_camera(
            scale_unit=scale_unit,
            rotation_vector=world_orientations[icam],
            translation_vector=world_locations[icam])

        # Plot it and change the color according to it's number
        ax.add_collection3d(Poly3DCollection(
            cam_verts[icam], facecolors='C'+str(icam), linewidths=1, edgecolors='k', alpha=1))

        # Give each camera a label
        ax.text(np.asscalar(cam_center[0]), np.asscalar(cam_center[1]), np.asscalar(cam_center[2]),
                'Cam ' + str(serials[icam]))
        
    if extrinsics_config['estimate_method'] == 'one-shot':
        world_points = np.squeeze(camera_tools.create_world_points(ncams_config))
        ax.scatter(world_points[:,0],world_points[:,1],world_points[:,2], c='k', marker='s', alpha=1)

    # mpl is weird about maintaining aspect ratios so this has to be done
    ax_min = np.min(np.hstack(cam_verts))
    ax_max = np.max(np.hstack(cam_verts))

    # Set the axes and viewing angle
    # Note that this is reversed so that the cameras are looking towards us
    ax.set_xlim([ax_max, ax_min])
    ax.set_ylim([ax_min, ax_max])
    ax.set_zlim([ax_min, ax_max])
    ax.view_init(elev=105, azim=-90)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


#################### Camera plotting helper functions
def create_camera(scale_unit=1, rotation_vector=None, translation_vector=None):
    '''Create a typical camera shape.

    [description]

    Keyword Arguments:
        scale_unit {number} -- [description] (default: {1})
        rotation_vector {[type]} -- [description] (default: {None})
        translation_vector {[type]} -- [description] (default: {None})
    Output:
        camera_vertices {np.array} -- [description]
        cam_center {np.array} -- [description]
    '''
    # Lines:
    # Back of camera body
    #  Front of camera body/back of lens
    # Back of camera body
    cam_points = np.array([
        [0, 0, 0],       [1, 0, 0],       [1, 1, 0],       [0, 1, 0],
        [0.2, 0.2, 0.5], [0.8, 0.2, 0.5], [0.8, 0.8, 0.5], [0.2, 0.8, 0.5],
        [0.2, 0.2, 1],   [0.8, 0.2, 1],   [0.8, 0.8, 1],   [0.2, 0.8, 1]])

    # Set the origin as the back of the lens
    centering_vector = [0.5, 0.5, 0.5]
    cam_points = cam_points - centering_vector

    # Scale the points
    cam_points = cam_points * scale_unit

    # Move the camera
    cam_points = move_camera(cam_points, rotation_vector, translation_vector)

    # Get the vertices & center
    camera_vertices = get_camera_vertices(cam_points)
    cam_center = np.mean(cam_points[4:8, :], 0)
    cam_center[1] = cam_center[1] + scale_unit

    return camera_vertices, cam_center


def move_camera(cam_points, rotation_vector=None, translation_vector=None):
    '''Applies the appropriate rotation and translation to the camera points.

    [description]

    Arguments:
        cam_points {[type]} -- [description]

    Keyword Arguments:
        rotation_vector {np.array} -- [description] (default: {None})
        translation_vector {np.array} -- [description] (default: {None})
    '''
    # Check rotation vector format
    if rotation_vector is None:
        rotation_vector = np.identity(3) # Assume it's not rotating
    elif rotation_vector.shape == (3, 1) or rotation_vector.shape == (1, 3):
        # Make matrix if necessary
        rotation_vector = cv2.Rodrigues(rotation_vector)[0] # Convert to matrix

    if translation_vector is None:
        translation_vector = np.zeros((3, 1)) # Assume there is no translation
    elif translation_vector.shape == (1, 3):
        translation_vector = np.transpose(translation_vector) # Format

    # Create the translation vector
    translation_vector = np.matmul(-np.transpose(rotation_vector), translation_vector)

    # Rotate and then translate
    cam_points = np.transpose(np.matmul(np.transpose(rotation_vector), np.transpose(cam_points)))
    cam_points = cam_points - np.transpose(translation_vector)

    return cam_points


def get_camera_vertices(cam_points):
    '''Manual mapping of the camera points from in create_camera.

    [description]

    Arguments:
        cam_points {list} -- 12-element array.
    Output:
        cam_verts {list 9x4} -- [description]
    '''
    cam_verts = [
        [cam_points[0], cam_points[4], cam_points[5], cam_points[1]],
        [cam_points[1], cam_points[5], cam_points[6], cam_points[2]],
        [cam_points[2], cam_points[6], cam_points[7], cam_points[3]],
        [cam_points[3], cam_points[7], cam_points[4], cam_points[0]], # Sides of lenses
        [cam_points[4], cam_points[8], cam_points[9], cam_points[5]],
        [cam_points[5], cam_points[9], cam_points[10], cam_points[6]],
        [cam_points[6], cam_points[10], cam_points[11], cam_points[7]],
        [cam_points[7], cam_points[11], cam_points[8], cam_points[4]],  # Sides of body
        [cam_points[8], cam_points[9], cam_points[10], cam_points[11]]]  # Back of body

    return cam_verts
