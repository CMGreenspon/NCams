#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019-2020 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

General camera functions and tools used for calibration, pose estimation, etc.

Important structures:
    ncams_config {dict} -- information about camera configuration. Should have following keys:
        serials {list of numbers} -- list of camera serials. Wherever camera-specific
            information is not in a dictionary but is in a list, the order MUST adhere to the order
            in serials.
        reference_camera_serial {number} -- serial number of the reference camera.
        image_size {(height, width)} -- size of the images captured by the cameras.
        board_type {'checkerboard' or 'charuco'} -- what type of board was used for calibration.
        board_dim {list with 2 numbers} -- number of checks on the calibration board.
        check_size {number} -- height and width of a single check mark.
        world_units {str} -- Units to use for all calibrations ('m', 'dm', 'cm', 'mm')
        setup_path {string} -- directory where the camera setup is located, including config.yaml.
        setup_filename {string} -- config has been loaded from os.path.join(
            setup_path, setup_filename) and/or will be saved into this directory.
        calibration_path {string} -- relative path to where calibration information is stored from
            'setup_path'.
        calibration_filename {string} -- name of the pickle file to store the calibration config
            in/load from.
        pose_estimation_path {string} -- relative path to where pose estimation information is
            stored from 'setup_path'.
        pose_estimation_filename {string} -- name of the pickle file to store the pose estimation
            config in/load from.
        system {PySpin.System instance} -- needed for init/close of the FLIR cameras system.

    camera_dictionary ('dicts') {dict} -- information about a single camera. Should have following keys:
        serial {number} -- serial number (ID) of the camera
        name {string} -- unique string that identifies the camera. Usually, 'cam_'+str(serial) or
            'top_left', 'top_right', etc.
        obj {PySpin.CameraPtr} -- reference to the PySpin camera object.

    calibration_config {dict} -- information on camera calibration and the results of said
            calibraion. Order of each list MUST adhere to calibration_config['serials'] AND
            ncams_config['serials']. Should have following keys:
        serials {list of numbers} -- list of camera serials.
        distortion_coefficients {list of np.arrays} -- distortion coefficients for each camera
        camera_matrices {list of np.arrays} -- the essential camera matrix for each camera.
        reprojection_errors {list of numbers} -- average error in pixels for each camera.
        path {string} -- directory where calibration information is stored. Should be same as
            information in ncams_config.
        filename {string} -- name of the pickle file to store the config in/load from.
        dicts {dict of 'camera_calib_dict's} -- keys are serials, values are 'camera_calib_dict',
            see below.

    camera_calib_dict {dict} -- info on calibration of a single camera. Sould have following keys:
        serial {number} -- UID of the camera.
        distortion_coefficients {np.array 1x5} -- distortion coefficients for the camera.
        camera_matrix {np.array 3x3} -- camera calibration matrix for the camera.
        reprojection_error {number} -- reprojection error for the camera.

    pose_estimation_config {dict} -- information on estimation of relative position of all cameras
            and the results of said pose estimation. Order of each list MUST adhere to
            pose_estimation_config['serials'] and ncams_config['serials']. Should
            have following keys:
        serials {list of numbers} -- list of camera serials.
        world_locations {list of np.arrays} -- world locations of each camera.
        world_orientations {list of np.arrays} -- world orientation of each camera.
        path {string} -- directory where pose estimation information is stored. Should be same as
            information in ncams_config.
        filename {string} -- name of the YAML file to store the config in/load from.
        dicts {dict of 'camera_pe_dict's} -- keys are serials, values are 'camera_pe_dict',
            see below.

    camera_dict {dict} -- info on pose estimation of a single camera. Sould have following keys:
        serial {number} -- UID of the camera.
        world_location {np.array} -- world location of the camera.
        world_orientation {np.array} -- world orientation of the camera.
"""

import os
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as mpl_pp

import reportlab
import reportlab.lib
import reportlab.platypus


def make_projection_matrix(camera_matrix, world_orientation, world_location):
    '''Makes a projection matrix from camera calibration and pose estimation info.

    Arguments:
        camera_matrix {np.array} -- camera calibration matrix for the camera.
        world_orientation {np.array} -- world orientation of the camera.
        world_location {np.array} -- world location of the camera.

    Output:
        projection_matrix {np.array} -- projection matrix of the camera
    '''
    # Make matrix if necessary
    if world_orientation.shape == (3, 1) or world_orientation.shape == (1, 3):
        world_orientation = cv2.Rodrigues(world_orientation)[0]  # Convert to matrix

    if world_location.shape == (1, 3):  # Format
        world_location = np.transpose(world_location)

    projection_matrix = np.matmul(camera_matrix, np.hstack((world_orientation, world_location)))

    return projection_matrix


def create_board(ncams_config, output=False, plotting=False, dpi=300, output_format='pdf',
                 padding=None, target_size=None, dictionary=None):
    '''Creates a board image.

    Creates either a checkerboard or charucoboard that can be printed and used for camera
    calibration and pose estimation.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            board_type {'checkerboard' or 'charuco'} -- what type of board was used for calibration.
            board_dim {list with 2 numbers} -- number of checks on the calibration board.
            check_size {number} -- height and width of a single check mark, mm.
            setup_path {string} -- directory where the camera setup is located, including
                config.yaml.
            units {str} -- the desired unit of scale for the world. ('m' = meter, 'dm' = decimeter,
                 'cm' = centimeter, 'mm' = millimeter)

    Keyword Arguments:
        output {bool} -- save the image to the drive. (default: {False})
        plotting {bool} -- plot the board as matplotlib image. (default: {False})
        dpi {number} -- resolution, dots per inch. (default: {300})
        output_format {str} -- file extension of the image printed to the drive (default: {'pdf'})
        padding {number} -- add padding to the image (default: {0})
        target_size {list (width, height)} -- size of the target printed image (default: {None})
        dictionary {cv2.aruco.Dictionary} -- [description] (default: {None})

    Output:
        output_dict {cv2.aruco.Dictionary} -- [description]. None if checkerboard is selected.
        output_board {[type]} -- [description]. None if checkerboard is selected.
        board_img {np.array} -- board image.
    '''
    # Unpack dict
    board_type = ncams_config['board_type']
    board_dim = ncams_config['board_dim']
    check_size = ncams_config['check_size']
    if output:
        output_path = ncams_config['setup_path']

    dpmm = dpi / 25.4 # Convert inches to mm

    # Make the dictionary
    total_markers = int(np.floor((board_dim[0] * board_dim[1]) / 2))

    # Make the board & array for image
    board_width = (board_dim[0] * check_size)
    board_height = (board_dim[1] * check_size)
    board_img = np.zeros((int(board_width * dpmm) + board_dim[0],
                          int(board_height * dpmm) + board_dim[1]))

    if board_type == 'checkerboard':
        # Litearlly just tile black and white squares
        check_length_in_pixels = int(np.round(check_size * dpmm))
        black_check = np.ones((check_length_in_pixels, check_length_in_pixels)) * 255
        white_check = np.zeros((check_length_in_pixels, check_length_in_pixels))
        board_img = np.empty((0, check_length_in_pixels*board_dim[0]), int)

        white = True
        for _ in range(board_dim[1]):
            col = np.empty((check_length_in_pixels, 0), int)
            for __ in range(board_dim[0]):
                if white:
                    col = np.append(col, white_check, axis=1)
                else:
                    col = np.append(col, black_check, axis=1)
                white = not white

            board_img = np.append(board_img, col, axis=0)
    elif board_type == 'charuco':
        if dictionary is None:
            output_dict = cv2.aruco.Dictionary_create(total_markers, 5)
        else:
            custom_dict = cv2.aruco.Dictionary_get(dictionary)
            output_dict = cv2.aruco.Dictionary_create_from(total_markers, custom_dict.markerSize,
                                                           custom_dict)
        
        if ncams_config['world_units'] == 'mm':
            scale_unit = 1
        elif ncams_config['world_units'] == 'cm':
            scale_unit = 10
        elif ncams_config['world_units'] == 'dm':
            scale_unit = 100
        elif ncams_config['world_units'] == 'm':
            scale_unit = 1000
        else:
            warnings.warn('Invalid scale unit given. Defaulting to centimeters')
            scale_unit = 10
        
        secondary_length = check_size * 0.6 # What portion of the check the aruco marker takes up
        output_board = cv2.aruco.CharucoBoard_create(board_dim[0], board_dim[1],
                                                     check_size/scale_unit,
                                                     secondary_length/scale_unit,
                                                     output_dict)

        # The board is compiled upside down so the top of the image is actually the bottom,
        # to avoid confusion it's rotated here
        board_img = np.rot90(output_board.draw((int(board_width * dpmm), int(board_height * dpmm)),
                                               board_img, 1, 1), 2)
    else:
        raise ValueError('Unknown board_type given.')

    if plotting:
        ax = mpl_pp.subplots()[1]
        ax.imshow(board_img/255, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    if padding is not None:
        board_img = np.pad(board_img, padding*dpi, 'constant', constant_values=255)
    elif target_size is not None:
        larger_board_img = np.ones((target_size[0], target_size[1]))*255
        size_diff = np.array(larger_board_img.shape) - np.array(board_img.shape)
        if any(size_diff < 0):
            raise Exception('Target size is smaller than board size.')
        r_off = int(np.round(size_diff[0]/2))
        c_off = int(np.round(size_diff[1]/2))
        larger_board_img[r_off:r_off+board_img.shape[0], c_off:c_off+board_img.shape[1]] = board_img
        board_img = larger_board_img

    if output:
        if output_format == 'pdf':
            output_name = os.path.join(output_path, board_type + '_board.png')
            cv2.imwrite(output_name, board_img)
            # To vertically center the board
            diff_in_vheight = ((reportlab.lib.pagesizes.letter[1]/72)*25.4 - board_height) / 2

            # Start building
            elements = []
            doc = reportlab.platypus.SimpleDocTemplate(
                os.path.join(output_path, "charuco_board.pdf"),
                pagesize=reportlab.lib.pagesizes.letter, topMargin=0, bottomMargin=0)
            elements.append(reportlab.platypus.Spacer(1, diff_in_vheight*reportlab.lib.units.mm))

            board_element = reportlab.platypus.Image(output_name)
            board_element.drawWidth = board_width*reportlab.lib.units.mm
            board_element.drawHeight = board_height*reportlab.lib.units.mm
            elements.append(board_element)
            doc.build(elements)
        else:
            cv2.imwrite(os.path.join(output_path, board_type+'_board.'+output_format), board_img)

    if board_type == 'checkerboard':
        return None, None, board_img
    if board_type == 'charuco':
        return output_dict, output_board, board_img


def create_world_points(ncams_config):
    '''Creates world points.

    For a given image of a board the relative position of the points is known and is the ground truth.
    This takes the information from the NCams config to create the world points. If the actual position
    of the board is known and it is not wished for the board to be used as the world origin then
    translations and rotations are necessary.

    Arguments:
        ncams_config {dict} -- information about camera configuration. Should have following keys:
            board_type {'checkerboard' or 'charuco'} -- what type of board was used for calibration.
            board_dim {list with 2 numbers} -- number of checks on the calibration board.
            check_size {number} -- height and width of a single check mark, mm.

    Output:
        world_points {np.array} -- world points based on board.
    '''
    board_type = ncams_config['board_type']
    board_dim = ncams_config['board_dim']
    check_size = ncams_config['check_size']

    if board_type == 'checkerboard':
        # The points are the intersections of the checks and so the number of points is the product
        # of the number of checks in each dimensions-1 
        world_points = np.zeros(((board_dim[0]-1) * (board_dim[1]-1), 3), np.float32) # x,y,z points
        # z is always zero:
        world_points[:, :2] = np.mgrid[0:board_dim[0]-1, 0:board_dim[1]-1].T.reshape(-1, 2)
        world_points = world_points * check_size
    elif board_type == 'charuco':
        charuco_board = create_board(ncams_config)[1]
        nc = charuco_board.chessboardCorners.shape[0]
        world_points = charuco_board.chessboardCorners.reshape(nc, 1, 3)

    return world_points

def test_charucoboard_detection(charuco_board, charuco_dict, image_path):
    ''' 
    A quick function for inspecting whether or not the charucoboard is being detected.
    Arguments:
        charuco_board {aruco_Charucobaord} -- the aruco board object.
        charuco_dict {aruco_Dictionary} -- the aruco dictionary object.
        image_path {str} -- full path to the image of interest
    '''
    
    img = cv2.imread(image_path)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Detect the aruco markers and get IDs
    corners, ids, _ = cv2.aruco.detectMarkers(img, charuco_dict)
    if ids is not None:
        # Find the corners and IDs
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, img, charuco_board)
    else:
        print('No markers detected')
        return
        
    image_annotated = cv2.aruco.drawDetectedCornersCharuco(im_rgb, charuco_corners)
    
    ax = mpl_pp.subplots()[1]
    ax.imshow(image_annotated)
    ax.set_xticks([])
    ax.set_yticks([])
