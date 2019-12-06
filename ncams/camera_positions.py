#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Please see AUTHORS for contributors.
https://github.com/CMGreenspon/NCams/blob/master/README.md
Licensed under the Apache License, Version 2.0

Functions related to estimation of relative positions of the cameras.
"""

import os
import datetime
import pickle
from copy import deepcopy

import glob
import cv2
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as mpl_pp

from . import utils
from . import image_t


def multi_camera_pose_estimation():
    '''[Short description]

    [Long description]

    Arguments:
        []
    Keyword Arguments:
        []
    Output:
        []
    '''
    # num_cameras = len(camera_config['camera_names'])
    # if num_cameras == 1:
    #     raise Exception('Only one camera present, pose cannot be calculated with this function.')
    #     return []
    raise NotImplementedError


def WORKING_auto_pose_estimation(camera_config, reference_camera):
    # Unpack the dict
    board_type = camera_config['board_type']
    num_cameras = len(camera_config['camera_names'])

    if board_type == 'charuco':
        # Create the board
        _, charuco_board, _ = create_board(camera_config)
        board_dims = charuco_board.getChessboardSize()
        num_corners = (board_dims[0]-1) * (board_dims[1]-1) # Need to know how to format the logit table
        # Get the ids and image points across images and cameras
        cam_image_points, cam_charuco_ids = charuco_board_detector(camera_config)
        # Parse the total number of shared points across all cameras
        pose_method = get_optimal_pose_method(cam_charuco_ids, board_type, num_corners)
        # Get all the poses
        if pose_method == 'common':
            world_locations, world_orientations = common_pose_estimation(camera_config,
                                                                         cam_image_points, cam_charuco_ids,
                                                                         camera_matrices, distortion_coefficients)
        else:
          return[]


    #if pose_strategy == 'stereo_sequential':
    export_pose_estimation(os.path.join(camera_config['folder_path'], 'pose_estimation'), camera_config['camera_names'],
                           world_locations, world_orientations)
    return world_locations, world_orientations


def WORKING_get_optimal_pose_method(input_array, board_type, num_corners):

    if board_type == 'charuco':
        num_images = len(input_array)
        num_cameras = len(input_array[0])
        shared_points_counter = np.zeros((len(input_array),1), dtype = int)
        for image in range(num_images):
            # Empty array with a spot for each corner and ID
            point_logit = np.zeros((num_corners, num_cameras), dtype = bool)
            for cam in range(num_cameras):
                # Get the array specific to the camera and image
                temp = input_array[image][cam]
                if isinstance(temp, np.ndarray):
                    for corner in temp:
                      point_logit[int(corner),cam] = True

    sum_point_logit = np.sum(point_logit.astype(int), 1)
    common_points = sum_point_logit == num_cameras
    shared_points_counter[image,0] = np.sum(common_points.astype(int))
    num_common_points = np.sum(shared_points_counter)

    if num_common_points >= 250:
      optimal_method = 'common'
    else:
      optimal_method = 'sequential-stereo'

    return optimal_method


def TODO_get_world_pose(camera_config):

    return []

def one_shot_multi_PnP(camera_config, camera_matrices, distortion_coefficients,
                       pose_estimation_path=None):
    charuco_dict, charuco_board, _ = create_board(camera_config)
    world_points = create_world_points(camera_config)
    h,w = camera_config['image_size']
    if pose_estimation_path is None:
        pose_estimation_path = os.path.join(camera_config['folder_path'],
                                            'pose_estimation')
    im_list = utils.get_image_list(pose_estimation_path)

    world_locations, world_orientations = [],[]
    for cam in range(len(camera_config['camera_names'])):
        # Find the correct image
        im_name = [i for i in im_list if camera_config['camera_names'][cam] in i]
        # If more than one image contains the camera name ask user to select
        if len(im_name) > 1:
            import easygui
            cwd = os.getcwd
            print('--> Multiple images contain the camera name. Select the'
                  ' correct file for "{}".'.format(
                    camera_config['camera_names'][cam]))
            os.chdir(pose_estimation_path)
            im_path = easygui.fileopenbox(
                title='Image selector',
                msg='select the image for "{}".'.format(
                    camera_config['camera_names'][cam]))
            os.chdir(cwd)
        else:
            im_path = os.path.join(pose_estimation_path, im_name[0])

        world_image = cv2.imread(im_path)

        # Get the image points
        # Detect the aruco markers and IDs
        corners, ids, rejected_points = cv2.aruco.detectMarkers(world_image, charuco_dict)
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, world_image, charuco_board)
        # Get the optimal camera matrix
        temp_optim, _ = cv2.getOptimalNewCameraMatrix(camera_matrices[cam],
                                                      distortion_coefficients[cam],
                                                      (w,h), 1, (w,h))
        # Undistort image points
        undistorted_points = cv2.undistortPoints(np.vstack(charuco_corners),
                                                 camera_matrices[cam],
                                                 distortion_coefficients[cam],
                                                 P = temp_optim)
        # Match to world points
        filtered_world_points = []
        for id in charuco_ids:
            filtered_world_points.append(world_points[id, :,:])
        filtered_world_points = np.vstack(filtered_world_points)
        # PnP
        _, rvec, tvec = cv2.solvePnP(
                filtered_world_points, undistorted_points,
                camera_matrices[cam], distortion_coefficients[cam])
        # Append
        world_locations.append(tvec)
        world_orientations.append(rvec)

    return world_locations, world_orientations


def common_pose_estimation(camera_config, cam_image_points, detection_logit, camera_matrices, distortion_coefficients):
    ''' If there are sufficient shared world points across all cameras then camera pose can be estimated from all of them simultaneously.'''
    #
    num_cameras = len(camera_config['camera_names'])
    # Determine the reference camera
    if 'reference_cam' in camera_config:
      reference_cam = camera_config['reference_cam']
    else:
      reference_cam = 0

    h,w = camera_config['image_size']

    cam_idx = np.arange(0,num_cameras)
    secondary_idx = cam_idx[cam_idx != reference_cam] # Get the indices of the non-primary cameras
    # Get the world points
    world_points = create_world_points(camera_config)
    corner_idx = np.arange(0, len(world_points))

    if camera_config['board_type'] == 'charuco':
    # Get all the points shared across cameras
      filtered_object_points = []
      filtered_image_points = [[] for cam in range(num_cameras)]

      for image in range(len(cam_image_points)):
        point_logit = np.zeros((len(world_points), num_cameras), dtype = bool) # Empty array with a spot for each corner and ID
        for cam in range(num_cameras):
          temp = detection_logit[image][cam] # Get the array specific to the camera and image
          if isinstance(temp, np.ndarray):
            for corner in temp: # For each detected corner
              point_logit[int(corner),cam] = True # Assign true to the logit array

        sum_point_logit = np.sum(point_logit.astype(int),1)
        common_points = sum_point_logit == num_cameras # Find which points are shared across all cameras

        if np.sum(common_points) >= 6:
          # Append only those points
          filtered_object_points.append(world_points[common_points,:].astype('float32'))
          for cam in range(num_cameras):
            temp_corners = np.zeros((np.sum(common_points),2), dtype=float)
            temp_ids = detection_logit[image][cam]
            temp_points = cam_image_points[image][cam]

            running_idx = 0
            for corner in corner_idx[common_points]: # Only append
              idx = int(np.where(temp_ids == corner)[0])
              temp_corners[running_idx,:] = temp_points[idx,:,:]
              running_idx += 1

            filtered_image_points[cam].append(temp_corners.astype('float32'))

    elif camera_config['board_type'] == 'checkerboard':
        x = 1

    # Get the optimal matrices and undistorted points
    optimal_matrices, undistorted_points = [],[]
    for cam in range(num_cameras):
        temp_optim, _ = cv2.getOptimalNewCameraMatrix(camera_matrices[cam],
                                                              distortion_coefficients[cam],
                                                              (w,h), 1, (w,h))
        optimal_matrices.append(temp_optim)
        undistorted_points.append(cv2.undistortPoints(np.vstack(filtered_image_points[cam]),
                                                      camera_matrices[cam], distortion_coefficients[cam],
                                                      P = optimal_matrices[cam]))

    # Perform the initial stereo calibration
    secondary_cam = secondary_idx[0]
    stereo_calib_flags = 0
    stereo_calib_flags |= cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.001)

    reprojection_error, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        filtered_object_points, filtered_image_points[reference_cam], filtered_image_points[secondary_cam],
        camera_matrices[reference_cam], distortion_coefficients[reference_cam],
        camera_matrices[secondary_cam], distortion_coefficients[secondary_cam],
        (w,h), None, None, None, criteria, stereo_calib_flags)

    if reprojection_error > 1:
        print('Poor initial stereo-calibration. Subsequent pose estimates may be innacurate.')


    # Make projection matrices
    projection_matrix_primary = np.dot(camera_matrices[reference_cam],
                                         np.hstack((np.identity(3), np.zeros((3,1))))) # This keeps the reference frame to that of the primary camera
    # Make the secondary projection matrix from calculated rotation & translation matrix
    projection_matrix_secondary = np.dot(camera_matrices[secondary_cam],np.hstack((R, T)))

    # Triangulate those same points
    triangulated_points_norm = cv2.triangulatePoints(projection_matrix_primary, projection_matrix_secondary,
                                                     undistorted_points[reference_cam], undistorted_points[secondary_cam])
    triangulated_points = triangulated_points_norm[:3,:]/np.transpose(np.repeat(triangulated_points_norm[3,:], 3).reshape((-1,3))) # Normalize

    # Construct outputs
    world_orientations, world_locations = [], []
    # Get pose from the triangulated points for all cameras
    for cam in cam_idx:
        _, rvec, tvec = cv2.solvePnP(
                np.transpose(triangulated_points), np.vstack(filtered_image_points[cam]),
                camera_matrices[cam], distortion_coefficients[cam])
        world_orientations.append(rvec)
        world_locations.append(tvec)

    return world_locations, world_orientations

def TODO_sequential_pose_estimation(cam_board_logit, cam_image_points, reference_camera,
                                    camera_matrices, distortion_coefficients):
    ''' If insufficient shared points then we can instead use the reference pair of cameras and iteratively calibrate all other cameras. '''

    return []


def TODO_inspect_pose_estimation():
    import matplotlib.pyplot

    image_index = 10
    cam_indices = [0, 1]

    # Load the images
    im_list1 = utils.get_image_list(os.path.join(camera_config['folder_path'],
                                         'pose_estimation', cam_names[cam_indices[0]]))
    im1 = matplotlib.image.imread(os.path.join(camera_config['folder_path'], 'pose_estimation',
                                  cam_names[cam_indices[0]], im_list1[image_index]))

    im_list2 = utils.get_image_list(os.path.join(camera_config['folder_path'],
                                                     'pose_estimation', cam_names[cam_indices[1]]))
    im2 = matplotlib.image.imread(os.path.join(camera_config['folder_path'], 'pose_estimation',
                                               cam_names[cam_indices[1]], im_list2[image_index]))

    # Detect the markers
    corners1, ids1, _ = cv2.aruco.detectMarkers(im1, charuco_dict)
    corners2, ids2, _ = cv2.aruco.detectMarkers(im2, charuco_dict)
    # Get the chessboard
    _, charuco_corners1, charuco_ids1 = cv2.aruco.interpolateCornersCharuco(corners1, ids1, im1, charuco_board)
    _, charuco_corners2, charuco_ids2 = cv2.aruco.interpolateCornersCharuco(corners2, ids2, im2, charuco_board)

    # Just get overlapping ones
    shared_ids, shared_world_points = [], []
    for id in charuco_ids1:
        if any(charuco_ids2 == id):
            shared_ids.append(id)
            shared_world_points.append(world_points[id,0,:])
    shared_ids = np.vstack(shared_ids)
    shared_world_points = np.vstack(shared_world_points).reshape(len(shared_ids), 1, 3)

    shared_corners1, shared_corners2 = [], []
    for id in shared_ids:
        idx1, _ = np.where(charuco_ids1 == id)
        shared_corners1.append(charuco_corners1[idx1,0,:])
        idx2, _ = np.where(charuco_ids2 == id)
        shared_corners2.append(charuco_corners2[idx2,0,:])

    shared_corners1 = np.vstack(shared_corners1).reshape(len(shared_ids), 1, 2)
    shared_corners2 = np.vstack(shared_corners2).reshape(len(shared_ids), 1, 2)

    cam_image_points, cam_charuco_ids = charuco_board_detector(camera_config)
    R, T = WORKING_common_pose_estimation(camera_config, cam_image_points, cam_charuco_ids, camera_matrices, distortion_coefficients)

    projection_primary = np.matmul(camera_matrices[0],np.hstack((np.identity(3), np.zeros((3,1)))))
    projection_secondary = np.matmul(camera_matrices[1],np.hstack((R, T)))

    # Now lets try to triangulate these shared points
    new_cam_mat1, _ = cv2.getOptimalNewCameraMatrix(camera_matrices[cam_indices[0]], distortion_coefficients[cam_indices[0]], (w,h), 1, (w,h))
    undistorted_points1 = cv2.undistortPoints(np.vstack(shared_corners1), camera_matrices[cam_indices[0]], distortion_coefficients[cam_indices[0]], P = new_cam_mat1)

    new_cam_mat2, _ = cv2.getOptimalNewCameraMatrix(camera_matrices[cam_indices[1]], distortion_coefficients[cam_indices[1]], (w,h), 1, (w,h))
    undistorted_points2 = cv2.undistortPoints(np.vstack(shared_corners2), camera_matrices[cam_indices[1]], distortion_coefficients[cam_indices[1]], P = new_cam_mat2)

    # Triangulate the points
    triangulated_points_norm = cv2.triangulatePoints(projection_primary, projection_secondary, undistorted_points1, undistorted_points2)
    triangulated_points = triangulated_points_norm[:3,:]/np.transpose(np.repeat(triangulated_points_norm[3,:], 3).reshape((-1,3)))

    # Reproject the points to each camera and verify
    reprojected_corners1,_ = cv2.projectPoints(triangulated_points, np.identity(3), np.zeros((3,1)),
                                             new_cam_mat1, distortion_coefficients[cam_indices[0]])
    reprojected_corners2,_ = cv2.projectPoints(triangulated_points, R, T,
                                             new_cam_mat2, distortion_coefficients[cam_indices[1]])

    fig, axs = matplotlib.pyplot.subplots(1,2, squeeze=False)
    axs[0,0].imshow(im1)
    axs[0,1].imshow(im2)
    for corner in range(len(shared_corners1)):
        axs[0,0].scatter(shared_corners1[corner, 0, 0], shared_corners1[corner, 0, 1], facecolors='none', edgecolors='b')
        axs[0,0].scatter(shared_corners1[corner, 0, 0], reprojected_corners1[corner, 0, 1], facecolors='none', edgecolors='r')

        axs[0,1].scatter(shared_corners2[corner, 0, 0], shared_corners2[corner, 0, 1], facecolors='none', edgecolors='b')
        axs[0,1].scatter(shared_corners2[corner, 0, 0], reprojected_corners2[corner, 0, 1], facecolors='none', edgecolors='r')


def plot_poses(world_locations, world_orientations, scale_factor = 1):
    ''' Creates a plot showing the location and orientation of all cameras given based on translation and rotation vectors.
    If your cameras are very close together or far apart you can change the scaling factor as necessary.'''
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    num_cameras = len(world_locations)
    if num_cameras == 1: # Only accepts list format so check if this is true only when a single camera is present
        if isinstance(world_locations, numpy.ndarray):
            world_locations = [world_locations]
        if isinstance(world_orientations, numpy.ndarray):
            world_orientations = [world_orientations]

    # Create a figure with axes
    fig = mpl_pp.figure()
    ax = fig.gca(projection='3d')
    # Keep the verts for setting the axes later
    cam_verts = [[] for _ in range(num_cameras)]
    for cam in range(num_cameras):
        # Get the vertices to plot appropriate to the translation and rotation
        cam_verts[cam], cam_center = create_camera(scale_factor = scale_factor,
                rotation_vector = world_orientations[cam],
                translation_vector = world_locations[cam])
        # Plot it and change the color according to it's number
        ax.add_collection3d(Poly3DCollection(cam_verts[cam],
                facecolors='C'+str(cam), linewidths=1, edgecolors='k', alpha=1))
        # Give each camera a label
        ax.text(np.asscalar(cam_center[0]),
                np.asscalar(cam_center[1]),
                np.asscalar(cam_center[2]),
                'Camera ' + str(cam+1))
    # mpl is weird about maintaining aspect ratios so this has to be done
    ax_min = np.min(np.hstack(cam_verts))
    ax_max = np.max(np.hstack(cam_verts))
    # Set the axes and viewing angle
    ax.set_xlim([ax_max, ax_min]) # Note that this is reversed so that the cameras are looking towards us
    ax.set_ylim([ax_min, ax_max])
    ax.set_zlim([ax_min, ax_max])
    ax.view_init(elev = 105, azim = -90)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
