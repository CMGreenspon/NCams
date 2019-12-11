#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Functions related to camera calibration.

For more details on the camera data structures and dicts, see help(ncams.camera_t).
"""

import os

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as mpl_pp

from . import utils
from . import image_t
from . import camera_io
from . import camera_t


def multi_camera_calibration(camera_config, override=False, inspect=False, export_full=False):
    '''Computes distortion coefficients from automatically selected images.

    This will go to the specified path and for each camera isolate the images necessary for
    computing distortion coefficients. The coefficients will then be output as a variable for
    undistortion purposes. In each cameras folder a yaml file will also be saved. In the general
    cam_calibration folder a pickle file containing all calibrations will be saved.

    Arguments:
        camera_config {dict} -- see help(ncams.camera_t). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
            board_type {'checkerboard' or 'charuco'} -- what type of board was used for calibration.
            board_dim {list with 2 numbers} -- number of checks on the calibration board.
            calibration_path {string} -- directory where calibration information is stored.
            calibration_filename {string} -- name of the pickle file to store the calibration config
                in/load from.
            pose_estimation_path {string} -- directory where pose estimation information is stored.
            pose_estimation_filename {string} -- name of the pickle file to store the pose
                estimation config in/load from.

    Keyword Arguments:
        override {bool} -- whether to automatically override detected calibration files.
            (default: {False})
        inspect {bool} -- whether to call the inspection function. (default: {False})
        export_full {bool} -- save the calibration to a dedicated file. (default: {False})

    Output:
        calibration_config {dict} -- information on camera calibration and the results of said
                calibraion. See help(ncams.camera_t). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            distortion_coefficientss {list of np.arrays} -- distortion coefficients for each camera
            camera_matrices {list of np.arrays} -- the essential camera matrix for each camera.
            reprojection_errors {list of numbers} -- average error in pixels for each camera.
            path {string} -- directory where calibration information is stored. Should be same as
                information in camera_config.
            filename {string} -- name of the pickle file to store the config in/load from.
            dicts {dict of 'camera_calib_dict's} -- keys are serials, values are
                'camera_calib_dict', see help(ncams.camera_t).
    '''
    # Unpack the dict
    serials = camera_config['serials']
    calib_dir = camera_config['calibration_path']

    calib_pickle_filename = os.path.join(calib_dir, camera_config['calibration_filename'])

    # First check if there is a calibration file
    if os.path.exists(calib_pickle_filename) and not override:
        print('A calibration file for all cameras has been detected. They may have already been'
              ' calibrated.')
        uinput_string = (
            "Proceed with calibration anyway? (Yes/No/Abort/Override/Help').\n"
            "Continuing will replace that calibration file.\n")
        while True:
            user_input = input(uinput_string).lower()
            if user_input in ('no', 'n'):
                # Let's save time and load that instead
                calibration_config = camera_io.import_calibration(camera_config)
                if inspect:
                    inspect_calibration(camera_config, calibration_config)
                return calibration_config
            if user_input in ('yes', 'y'):
                print('- Rerunning calibration.')
                break
            if user_input == 'abort':
                return
            if user_input == 'override':
                print('- Override enabled. Recalibrating all cameras.')
                override = True
                break
            if user_input == 'help':
                print('- Yes: rerun calibration. You will be asked if you would like to override'
                      ' individual files.\n'
                      '- No: load a file and exit the calibration function.\n'
                      '- Abort: exit the function without returning anything.\n'
                      '- Override: like \'Yes\', but all files will be overwritten'
                      ' automatically.\n')
            else:
                print("Invalid response given.\n")

    # Initalizes output
    reprojection_errors = []  # Reprojection errors
    camera_matrices = []  # Intrinsic camera parameters
    distortion_coefficientss = []  # Distortion coefficients
    dicts = {}

    # Preliminary stuff
    num_cameras = len(serials)  # How many cameras are there
    print('Beginning calibration of {} camera{}.'.format(
        num_cameras, 's' if num_cameras > 1 else ''))

    for icam, serial in enumerate(serials):
        print('- Camera {} of {}.'.format(icam+1, num_cameras))
        cam_name = camera_config['dicts'][serial]['name']

        # Check if images are in current directory or in a subdirectory
        if num_cameras == 1:
            if os.path.exists(os.path.join(calib_dir, cam_name)):
                # Go to the subdirectory
                cam_calib_dir = os.path.join(calib_dir, cam_name)
            else:
                # Stay where we are
                cam_calib_dir = calib_dir
        else:
            # Go to the subdirectory of that camera
            cam_calib_dir = os.path.join(calib_dir, cam_name)

        # Check if there is already a calibration file
        cam_calib_filename = os.path.join(cam_calib_dir, cam_name + '_calib.yaml')
        calibrate_camera = True
        if os.path.exists(cam_calib_filename) and not override:
            print('-> Calibration file for "{}" detected and may have already been'
                  ' calibrated.'.format(cam_name))
            uinput_string = "-- Calibrate anyway? (Yes/No/Abort).\n"
            while True:
                user_input = input(uinput_string).lower()
                if user_input in ('no', 'n'):
                    # We can instead load in the existing calibration file
                    camera_calib_dict = camera_io.yaml_to_calibration(cam_calib_filename)
                    camera_matrix = camera_calib_dict['camera_matrix']
                    distortion_coefficients = camera_calib_dict['distortion_coefficients']
                    reprojection_error = camera_calib_dict['reprojection_error']
                    calibrate_camera = False
                    break
                if user_input in ('yes', 'y'):
                    break
                if user_input == 'abort':
                    print('Aborting...')
                    return
                print("Invalid response given.\n")

        if calibrate_camera:
            # Check if there are images
            cam_image_list = utils.get_image_list(path=cam_calib_dir)
            if len(cam_image_list) == 0:
                # If there are no cameras append empty arrays so that the
                # index order is preserved and begin looking at next camera
                print('-> No images found in directory "{}" for camera {}.\n'
                      ' Continuing to the next camera...'.format(
                          cam_calib_dir, cam_name))
                reprojection_error = []
                distortion_coefficients = []
                camera_matrix = []
            else:
                if len(cam_image_list) < 25:
                    print('  Only {} images found. Calibration may be poor.'.format(
                        len(cam_image_list)))

                # Get coefficients and matrices for each camera
                if camera_config['board_type'] == 'checkerboard':
                    world_points = camera_t.create_world_points(camera_config)
                    # Run the calibration:
                    (reprojection_error, camera_matrix,
                     distortion_coefficients) = checkerboard_calibration(
                         cam_image_list, camera_config['board_dim'], world_points)
                elif camera_config['board_type'] == 'charuco':
                    # Create the board - world points included
                    charuco_dict, charuco_board, _ = camera_t.create_board(camera_config)
                    # Run the calibration:
                    (reprojection_error, camera_matrix,
                     distortion_coefficients) = charuco_calibration(cam_image_list, charuco_dict,
                                                                    charuco_board)

            # Export them to the camera folder in a readable format
            camera_calib_dict = {
                'serial': serial,
                'distortion_coefficients': distortion_coefficients,
                'camera_matrix': camera_matrix,
                'reprojection_error': reprojection_error
            }
            camera_io.calibration_to_yaml(cam_calib_filename, camera_calib_dict)

        # Return these to the workspace
        reprojection_errors.append(reprojection_error)
        distortion_coefficientss.append(distortion_coefficients)
        camera_matrices.append(camera_matrix)
        dicts[serial] = camera_calib_dict

    calibration_config = {
        'serials': serials,
        'distortion_coefficientss': distortion_coefficientss,
        'camera_matrices': camera_matrices,
        'reprojection_errors': reprojection_errors,
        'path': calib_dir,
        'filename': camera_config['calibration_filename'],
        'dicts': dicts
    }

    print('* Calibration complete.')
    if export_full:
        camera_io.export_calibration(calibration_config)

    if inspect:
        inspect_calibration(camera_config, calibration_config)

    return calibration_config


def checkerboard_calibration(cam_image_list, board_dim, world_points):
    '''Calibrates cameras using a checkerboard.

    Attempts to find a checkerboard in each image and performs the basic calibration. It is
    suggested to use the inspect_calibration tool to check if the calibration is good.

    Arguments:
        cam_image_list {list} -- list of images to search for checkerboards in.
        board_dim {list (cols, rows)} -- size of the board.
        world_points {list} -- ground truth x, y, z values.
    Output:
        reprojection_error {number} -- float indicating the average error of the calibration in
            pixels.
        camera_matrix {list} -- the calculated intrinsics of the camera
        distortion_coefficients {list} -- the calculated distortion parameters for the lens.
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Default criteria

    # Analyze the images to get checkerboard corners
    object_points = [] # 3d points in space
    image_points = [] # x,y image points
    board_logit = np.zeros((1, len(cam_image_list)), dtype=bool)

    for im_name in cam_image_list:
        img = cv2.imread(im_name, 0) # Load as grayscale & find the corners
        board_logit, corners = cv2.findChessboardCorners(
            img, (board_dim[0]-1, board_dim[1]-1), None)

        # If a checkerboard was found then append the world points and image points
        if board_logit:
            object_points.append(world_points)
            corners_refined = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners_refined)

    img_height = img.shape[0] # Necessary for the calibration step
    img_width = img.shape[1]
    # Calibrate
    reprojection_error, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
        object_points, image_points, (img_width, img_height), None, None)
    return reprojection_error, camera_matrix, distortion_coefficients


def charuco_calibration(cam_image_list, charuco_dict, charuco_board):
    '''Calibrates cameras using a charuco board.

    Attempts to find the given charucoboard in each image and performs the basic calibration. It is
    suggested to use the inspect_calibration tool to check if the calibration is good.

    Arguments:
        cam_image_list {list} -- list of images to search for checkerboards in.
        charuco_dict {cv2.aruco.Dictionary} -- [description]
        charuco_board {list} -- ground truth x, y, z values.
    Output:
        reprojection_error {number} -- float indicating the average error of the calibration in
            pixels.
        camera_matrix {np.array} -- the calculated intrinsics of the camera
        distortion_coefficients {np.array} -- the calculated distortion parameters for the lens.
    '''
    # Initalize
    ch_ids = []  # charuco ID list
    image_points = []  # x,y coordinates of charuco corners
    board_logit = np.zeros((1, len(cam_image_list)), dtype = bool)

    calib_flags = 0
    calib_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    calib_flags |= cv2.CALIB_ZERO_TANGENT_DIST

    # Iterate through each image and find corners & IDs
    for im, im_name in enumerate(cam_image_list):
        img = cv2.imread(im_name, 0) # Load as grayscale
        # Detect the aruco markers and get IDs
        corners, ids, _ = cv2.aruco.detectMarkers(img, charuco_dict)
        if ids is not None:  # Only try to find corners if markers were detected
            board_logit[0, im] = True # Keep track of detections
            # Find the corners and IDs
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, img, charuco_board)
            # In some instances a 'NoneType' is produced - this causes issues
            if isinstance(charuco_corners, np.ndarray):
                # If there are too few points this also won't work
                if len(charuco_corners[:, 0, 0]) > 4:
                    # Append values
                    ch_ids.append(charuco_ids)
                    image_points.append(charuco_corners)
            else:
                print('-> Markers could not be identified in "' + im_name + '".')

    # Calibrate
    img_width, img_height = img.shape[1], img.shape[0]
    f_length = max(img_width, img_height)
    # Make a guess at the inital state of the principal point based on size of image
    principal_point_init = np.array([
        [f_length, 0, img_width//2],
        [0, f_length, img_height//2],
        [0, 0, 1]
        ])

    (reprojection_error, camera_matrix, distortion_coefficients, _, _
     ) = cv2.aruco.calibrateCameraCharuco(
         image_points, ch_ids, charuco_board, (img_width, img_height), principal_point_init,
         None, flags=calib_flags)

    # Check output format - seems to be version dependent
    if isinstance(camera_matrix, cv2.UMat):
        camera_matrix = camera_matrix.get()
    if isinstance(distortion_coefficients, cv2.UMat):
        distortion_coefficients = distortion_coefficients.get()

    return reprojection_error, camera_matrix, distortion_coefficients


def inspect_calibration(camera_config, calibration_config, image_index=None):
    '''Provides the user with undistorted and distorted images for them to compare and inspect.

    Searches through images in calibration directory or uses a selected image and shows the user
    both the undistorted and distorted images (based on given undistortion parameters) so that they
    may inspect whether or not the calibration is sucessful. This is particularly useful for
    variable focus lenses where OpenCV may not be able to compute distortion coefficients if there
    is too much fisheye distortion.

    Arguments:
        camera_config {dict} -- see help(ncams.camera_t). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict', see
                below.
            board_type {'checkerboard' or 'charuco'} -- what type of board was used for calibration.
            board_dim {list with 2 numbers} -- number of checks on the calibration board.
            calibration_path {string} -- directory where calibration information is stored.
    Keyword Arguments:
        image_index {int} -- a specific frame number to look at (default: {first image w/ board in
            it})
    '''
    serials = camera_config['serials']
    board_type = camera_config['board_type']
    board_dim = camera_config['board_dim']
    calibration_path = camera_config['calibration_path']

    num_markers = (board_dim[0]-1) * (board_dim[1]-1)
    # Get layout of output array
    num_cameras = len(serials)

    num_vert_plots = int(np.ceil(np.sqrt(num_cameras)))
    num_horz_plots = int(np.ceil(num_cameras/num_vert_plots))

    _, axs = mpl_pp.subplots(num_horz_plots, num_vert_plots, squeeze=False)
    vert_ind = 0
    horz_ind = 0
    # Get the images and plot for each camera
    for serial in serials:
        # Folder navigation
        # If there is more than one camera assume subdirectories are present
        cam_calib_dir = os.path.join(calibration_path, camera_config['dicts'][serial]['name'])
        # Check if images are in current directory or in a subdirectory
        if num_cameras == 1 and not os.path.exists(cam_calib_dir):
            # camera pictures are not mixed
            cam_calib_dir = calibration_path

        # Get the appropriate camera matrices
        cam_mat = calibration_config['dicts'][serial]['camera_matrix']
        dist_coeffs = calibration_config['dicts'][serial]['distortion_coefficients']

        image_list = utils.get_image_list(path=cam_calib_dir)

        board_in_image = False
        idx = 0
        while not board_in_image:
            if image_index is None:
                image_ind = idx  # user-selected image
            else:
                image_ind = image_index
            example_image = matplotlib.image.imread(image_list[image_ind])

            # Get new camera matrix
            h, w = example_image.shape[:2]
            new_cam_mat = cv2.getOptimalNewCameraMatrix(cam_mat, dist_coeffs, (w, h), 1, (w, h))[0]
            if board_type == 'charuco':
                # Detect the markers
                charuco_dict, charuco_board, _ = camera_t.create_board(camera_config)
                corners, ids, rejected_points = cv2.aruco.detectMarkers(example_image, charuco_dict)
                if ids is not None:
                    # Find the checkerboard corners
                    _, example_corners, _ = cv2.aruco.interpolateCornersCharuco(
                        corners, ids, example_image, charuco_board)
                    if isinstance(example_corners, np.ndarray):
                        # Lets only use images with all corners detected
                        if len(example_corners) == num_markers or image_index is not None:
                            board_in_image = True
                            # Annotate example image
                            example_image_annotated = cv2.aruco.drawDetectedCornersCharuco(
                                example_image, example_corners)
                            # Undistort the corners and image
                            undistorted_corners = cv2.undistortPoints(
                                example_corners, cam_mat, dist_coeffs, P=new_cam_mat)
                            undistorted_image = image_t.undistort_image(
                                example_image, calibration_config['dicts'][serial])
                            undistorted_image_annotated = cv2.aruco.drawDetectedCornersCharuco(
                                undistorted_image, undistorted_corners)
                        elif image_index is not None:
                            print(' - Board not detected in requested image.')
                            example_image_annotated = np.zeros(example_image.shape)
                            undistorted_image_annotated = np.zeros(example_image.shape)
                            board_in_image = True
                            
            elif board_type == 'checkerboard':
                # Analyze the images to get checkerboard corners
                board_logit, corners = cv2.findChessboardCorners(
                    example_image, (board_dim[0]-1, board_dim[1]-1), None)
                # If a checkerboard was found then append the image points variable for calibration
                if board_logit:
                    board_in_image = True
                    example_image_annotated = cv2.drawChessboardCorners(
                        example_image, (board_dim[0]-1, board_dim[1]-1), corners, board_logit)
                    undistorted_corners = cv2.undistortPoints(
                        corners, cam_mat, dist_coeffs, P=new_cam_mat)
                    undistorted_image = image_t.undistort_image(
                        example_image, calibration_config['dicts'][serial])
                    undistorted_image_annotated = cv2.drawChessboardCorners(
                        undistorted_image, (board_dim[0]-1, board_dim[1]-1), undistorted_corners,
                        board_logit)
            idx += 1

        # Make the combined image
        padding = np.ones((example_image.shape[0],
                           int(np.floor(example_image.shape[1])/10), example_image.shape[2]),
                          dtype=np.int8) * 255
        cat_image = np.concatenate((example_image_annotated, padding, undistorted_image_annotated),
                                   axis=1)

        # Plot it
        axs[vert_ind, horz_ind].imshow(cat_image)
        axs[vert_ind, horz_ind].set_title('{}, error = {:.3f}'.format(
            camera_config['dicts'][serial]['name'],
            calibration_config['dicts'][serial]['reprojection_error']))
        axs[vert_ind, horz_ind].set_xticks([])
        axs[vert_ind, horz_ind].set_yticks([])

        horz_ind += 1
        if horz_ind > (num_horz_plots-1):
            horz_ind = 0
            vert_ind += 1
