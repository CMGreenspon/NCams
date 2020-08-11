#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Functions related to camera calibration.

For more details on the camera data structures and dicts, see help(ncams.camera_tools).
"""

import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as mpl_pp

from . import utils
from . import image_tools
from . import camera_io
from . import camera_tools


def multi_camera_intrinsic_calibration(ncams_config, override=False, inspect=False, export_full=True,
                             verbose=False, export_marked_images=False):
    '''Computes distortion coefficients from automatically selected images.

    This will go to the specified path and for each camera isolate the images necessary for
    computing distortion coefficients. The coefficients will then be output as a variable for
    undistortion purposes. In each cameras folder a yaml file will also be saved. In the general
    cam_calibration folder a pickle file containing all calibrations will be saved.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
            board_type {'checkerboard' or 'charuco'} -- what type of board was used for calibration.
            board_dim {list with 2 numbers} -- number of checks on the calibration board.
            intrinsic_path {string} -- relative path to where calibration information is stored
                from 'setup_path'.
            intrinsic_filename {string} -- name of the pickle file to store the calibration config
                in/load from.

    Keyword Arguments:
        override {bool} -- whether to automatically override detected calibration files.
            (default: {False})
        inspect {bool} -- whether to call the inspection function. (default: {False})
        export_full {bool} -- save the calibration to a dedicated file. (default: {True})
        verbose {bool} -- extra info given about error checks or bad images.

    Output:
        intrinsics_config {dict} -- information on camera calibration and the results of said
                calibraion. See help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            distortion_coefficients {list of np.arrays} -- distortion coefficients for each camera
            camera_matrices {list of np.arrays} -- the essential camera matrix for each camera.
            reprojection_errors {list of numbers} -- average error in pixels for each camera.
            path {string} -- directory where calibration information is stored. Should be same as
                information in ncams_config.
            filename {string} -- name of the pickle file to store the config in/load from.
            dicts {dict of 'camera_calib_dict's} -- keys are serials, values are
                'camera_calib_dict', see help(ncams.camera_tools).
    '''
    # Unpack the dict
    serials = ncams_config['serials']
    calib_dir = os.path.join(ncams_config['setup_path'], ncams_config['intrinsic_path'])

    calib_pickle_filename = os.path.join(calib_dir, ncams_config['intrinsic_filename'])

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
                intrinsics_config = camera_io.import_intrinsics(ncams_config)
                if inspect:
                    inspect_intrinsics(ncams_config, intrinsics_config)
                return intrinsics_config
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
    distortion_coefficients = []  # Distortion coefficients
    calibration_images = [] # File paths for each cameras images
    detected_markers = [] # Number of detected points in each image
    dicts = {}

    # Preliminary stuff
    num_cameras = len(serials)  # How many cameras are there
    print('Beginning calibration of {} camera{}'.format(
        num_cameras, 's' if num_cameras > 1 else ''))

    for icam, serial in enumerate(serials):
        print('- Camera {} of {}.'.format(icam+1, num_cameras))
        if 'dicts' in ncams_config.keys(): # If the name is explicitly defined then use it
            cam_name = ncams_config['dicts'][serial]['name']
        else: # otherwise assume it's 'cam' + serial
            cam_name = 'cam' + str(serial)

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
                    cam_distortion_coefficients = camera_calib_dict['distortion_coefficients']
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
                cam_distortion_coefficients = []
                camera_matrix = []
            else:
                if len(cam_image_list) < 25:
                    print('  Only {} images found. Calibration may be poor.'.format(
                        len(cam_image_list)))

                # Get coefficients and matrices for each camera
                if ncams_config['board_type'] == 'checkerboard':
                    world_points = camera_tools.create_world_points(ncams_config)
                    # Run the calibration:
                    (reprojection_error, camera_matrix,
                     cam_distortion_coefficients) = checkerboard_calibration(
                         cam_image_list, ncams_config['board_dim'], world_points)
                elif ncams_config['board_type'] == 'charuco':
                    # Create the board - world points included
                    charuco_dict, charuco_board, _ = camera_tools.create_board(ncams_config)
                    # Run the calibration:
                    (reprojection_error, camera_matrix, cam_distortion_coefficients,
                     detected_points) = charuco_calibration(cam_image_list, charuco_dict,
                                                            charuco_board, verbose=verbose,
                                                            export_marked_images=export_marked_images)

            # Export them to the camera folder in a readable format
            camera_calib_dict = {
                'serial': serial,
                'distortion_coefficients': cam_distortion_coefficients,
                'camera_matrix': camera_matrix,
                'reprojection_error': reprojection_error,
                'calibration_images': cam_image_list,
                'detected_markers': detected_points,
            }
            camera_io.intrinsic_to_yaml(cam_calib_filename, camera_calib_dict)

        # Return these to the workspace
        reprojection_errors.append(reprojection_error)
        distortion_coefficients.append(cam_distortion_coefficients)
        camera_matrices.append(camera_matrix)
        dicts[serial] = camera_calib_dict
        calibration_images.append(cam_image_list)
        detected_markers.append(detected_points)

    intrinsics_config = {
        'serials': serials,
        'distortion_coefficients': distortion_coefficients,
        'camera_matrices': camera_matrices,
        'reprojection_errors': np.vstack(reprojection_errors),
        'path': calib_dir,
        'filename': ncams_config['intrinsic_filename'],
        'dicts': dicts,
        'calibration_images': calibration_images,
        'detected_markers': detected_markers
    }

    print('* Calibration complete.')
    if export_full:
        camera_io.export_intrinsics(intrinsics_config)

    if inspect:
        inspect_intrinsics(ncams_config, intrinsics_config)

    return intrinsics_config


def checkerboard_calibration(cam_image_list, board_dim, world_points):
    '''Calibrates cameras using a checkerboard.

    Attempts to find a checkerboard in each image and performs the basic calibration. It is
    suggested to use the inspect_intrinsics tool to check if the calibration is good.

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
    reprojection_error = np.array([np.around(reprojection_error,5)])
    
    return reprojection_error, camera_matrix, distortion_coefficients


def charuco_calibration(cam_image_list, charuco_dict, charuco_board,
                        export_marked_images=False, verbose=False):
    '''Calibrates cameras using a charuco board.

    Attempts to find the given charucoboard in each image and performs the basic calibration. It is
    suggested to use the inspect_intrinsics tool to check if the calibration is good.

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
    detected_points = np.zeros((1, len(cam_image_list)), dtype = int)
    board_logit = np.zeros((1, len(cam_image_list)), dtype = bool)
    image_num = []
    
    if export_marked_images:
        header_path = os.path.split(cam_image_list[0])[0]
        cam_path =  os.path.split(header_path)[1]
        image_export_dir = os.path.join(header_path, cam_path + '_marked_images')
        if not os.path.exists(image_export_dir):
            os.mkdir(image_export_dir)


    # Iterate through each image and find corners & IDs
    for im, im_name in enumerate(cam_image_list):
        img = cv2.imread(im_name, 0) # Load as grayscale
        # Detect the aruco markers and get IDs
        corners, ids, rej_points = cv2.aruco.detectMarkers(img, charuco_dict)
        if ids is not None:  # Only try to find corners if markers were detected
            board_logit[0, im] = True # Keep track of detections
            # Find the corners and IDs
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, img, charuco_board)
            # In some instances a 'NoneType' is produced - this causes issues
            if isinstance(charuco_corners, np.ndarray):
                detected_points[0,im] = len(charuco_corners)
                if export_marked_images:
                    example_image_annotated = cv2.aruco.drawDetectedCornersCharuco(
                                    img, charuco_corners)
                    image_fullpath = os.path.join(image_export_dir, os.path.split(im_name)[1])
                    cv2.imwrite(image_fullpath,example_image_annotated)
                    
                # If there are too few points this also won't work
                if len(charuco_corners[:, 0, 0]) > 5:
                    # Append values
                    ch_ids.append(charuco_ids)
                    image_points.append(charuco_corners)
                    image_num.append(im)
            else:
                if verbose:
                    print('-> Markers could not be identified in "' + im_name + '".')
                    
    # Check if enough images contained relevant points
    if len(image_points) == 0:
        raise Warning('-> Charucoboards were not detected in any images, calibration will exit.')
        return [], [], [], []
    elif len(image_points) < 10:
        print('-> Less than 10 images contain identifiable boards. Consider taking new photos.')

    # Calibrate
    img_width, img_height = img.shape[1], img.shape[0]
    f_length = max(img_width, img_height)
    # Make a guess at the inital state of the principal point based on size of image
    principal_point_init = np.array([
        [f_length, 0, img_width//2],
        [0, f_length, img_height//2],
        [0, 0, 1]
        ])
    # Calibration parameters
    calib_flags = 0
    calib_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    calib_flags |= cv2.CALIB_ZERO_TANGENT_DIST

    (reprojection_error, camera_matrix, distortion_coefficients, _, _, _, _, pve
     ) = cv2.aruco.calibrateCameraCharucoExtended(
         image_points, ch_ids, charuco_board, (img_width, img_height), principal_point_init,
         None, flags=calib_flags)
             
    # Check if there are any bad frames causing issues and try to remove them
    if reprojection_error > 10 and any(pve > 1e3):
        if verbose:
            print('-> Potential bad calibration images detected. Attempting to remove and recalibrate.')
        good_frame_idx = pve < 10
        filtered_image_points = []
        filtered_ch_ids = []
        for im in range(len(image_points)):
            if good_frame_idx[im]:
                filtered_image_points.append(image_points[im])
                filtered_ch_ids.append(ch_ids[im])
                
        (reprojection_error2, camera_matrix2, distortion_coefficients2, _, _, _, _, pve
         ) = cv2.aruco.calibrateCameraCharucoExtended(
             filtered_image_points, filtered_ch_ids, charuco_board, (img_width, img_height),
             principal_point_init, None, flags=calib_flags)
             
        if reprojection_error2 < 10:
            if verbose:
                # List removed images
                print('-> Bad images sucessfuly identified, the following have been ignored:')
                bad_frame_idx = np.where(good_frame_idx == False)[0]
                for im in bad_frame_idx:
                    frame_num = image_num[im]
                    im_name = os.path.split(cam_image_list[frame_num])[1]
                    print('   ' + im_name)
                
            reprojection_error = reprojection_error2
            camera_matrix = camera_matrix2
            distortion_coefficients = distortion_coefficients2
            
    reprojection_error = np.array([np.around(reprojection_error,5)]) # Helps with yaml storage
                    
    # Check output format - seems to be version dependent
    if isinstance(camera_matrix, cv2.UMat):
        camera_matrix = camera_matrix.get()
    if isinstance(distortion_coefficients, cv2.UMat):
        distortion_coefficients = distortion_coefficients.get()

    # Indicate to the user if a likely error ocurred during the calibration
    if (distortion_coefficients == 0).all():
        print('-> No distortion detected. Calibration may have failed.')
    elif np.abs(distortion_coefficients[0, 4]) > 0.5:
        print('-> There may be a fisheye effect. Inspect the calibration.')
            
    if reprojection_error[0] > 1:
        print('-> The reprojection error is high. Inspect the calibration.')
        # Optional view of imagepoints to see where lacking points might be
        uinput_string = ('-> Would you like to view the image point heatmap? (yes/no)\n')
        user_input = input(uinput_string).lower()
        if user_input in ('yes', 'y'):
            show_image_point_heatmap(img.shape, image_points)
        

    return reprojection_error, camera_matrix, distortion_coefficients, detected_points


def inspect_intrinsics(ncams_config, intrinsics_config, image_index=None):
    '''Provides the user with undistorted and distorted images for them to compare and inspect.

    Searches through images in calibration directory or uses a selected image and shows the user
    both the undistorted and distorted images (based on given undistortion parameters) so that they
    may inspect whether or not the calibration is sucessful. This is particularly useful for
    variable focus lenses where OpenCV may not be able to compute distortion coefficients if there
    is too much fisheye distortion.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict', see
                below.
            board_type {'checkerboard' or 'charuco'} -- what type of board was used for calibration.
            board_dim {list with 2 numbers} -- number of checks on the calibration board.
            intrinsic_path {string} -- relative path to where calibration information is stored
                from 'setup_path'.
    Keyword Arguments:
        image_index {int} -- a specific frame number to look at (default: {first image w/ board in
            it})
    '''
    serials = ncams_config['serials']
    board_type = ncams_config['board_type']
    board_dim = ncams_config['board_dim']
    intrinsic_path = os.path.join(ncams_config['setup_path'], ncams_config['intrinsic_path'])

    num_markers = (board_dim[0]-1) * (board_dim[1]-1)
    # Get layout of output array
    num_cameras = len(serials)
    num_vert_plots = int(np.ceil(np.sqrt(num_cameras)))
    num_horz_plots = int(np.ceil(num_cameras/num_vert_plots))
    num_axes = num_vert_plots * num_horz_plots

    fig, axs = mpl_pp.subplots(num_vert_plots, num_horz_plots, squeeze=False)
    fig.canvas.set_window_title('NCams: Calibration inspection')
    # Get the images and plot for each camera
    for icam, serial in enumerate(serials):
        # Folder navigation
        # If there is more than one camera assume subdirectories are present
        if 'dicts' in ncams_config.keys():
            cam_calib_dir = os.path.join(intrinsic_path, ncams_config['dicts'][serial]['name'])
            cam_name = ncams_config['dicts'][serial]['name']
        else:
            cam_name = 'cam' + str(serial)
            cam_calib_dir = os.path.join(intrinsic_path, cam_name)
        # Check if images are in current directory or in a subdirectory
        if num_cameras == 1 and not os.path.exists(cam_calib_dir):
            # camera pictures are not mixed
            cam_calib_dir = intrinsic_path

        # Get the appropriate camera matrices
        cam_mat = intrinsics_config['dicts'][serial]['camera_matrix']
        dist_coeffs = intrinsics_config['dicts'][serial]['distortion_coefficients']

        image_list = utils.get_image_list(path=cam_calib_dir)
        num_markers_images = intrinsics_config['dicts'][serial]['detected_markers']

        image_ind = 0
        if image_index is None and board_type == 'charuco':
            image_ind = np.where(num_markers_images == np.amax(num_markers_images))[1][0]
        else:
            image_ind = image_index  # user-selected image
            
        example_image = matplotlib.image.imread(image_list[image_ind])

        if board_type == 'charuco':
            # Detect the markers
            charuco_dict, charuco_board, _ = camera_tools.create_board(ncams_config)
            corners, ids, rejected_points = cv2.aruco.detectMarkers(example_image, charuco_dict)
            if ids is not None:
                # Find the checkerboard corners
                _, example_corners, _ = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, example_image, charuco_board)
                if isinstance(example_corners, np.ndarray):
                    # Lets only use images with all corners detected
                    if len(example_corners) >= num_markers or image_index is not None:
                        # Undistort the corners and image
                        undistorted_corners = cv2.undistortPoints(example_corners, cam_mat,
                                                                  dist_coeffs, P=cam_mat)
                        undistorted_image = image_tools.undistort_image(
                            example_image, intrinsics_config['dicts'][serial])
                    elif image_index is not None:
                        print(' - Board not detected in requested image')

        elif board_type == 'checkerboard':
            # Analyze the images to get checkerboard corners
            board_logit, corners = cv2.findChessboardCorners(
                example_image, (board_dim[0]-1, board_dim[1]-1), None)
            # If a checkerboard was found then append the image points variable for calibration
            if board_logit:
                undistorted_corners = cv2.undistortPoints(
                    corners, cam_mat, dist_coeffs, P=cam_mat)
                undistorted_image = image_tools.undistort_image(
                    example_image, intrinsics_config['dicts'][serial])

        # Make the combined image
        padding_size = list(example_image.shape)
        padding_size[1] = int(padding_size[1]/10)
        padding = np.ones(tuple(padding_size), dtype=np.int8) * 255
        x_offset = example_image.shape[1] + padding_size[1]
                
        cat_image = np.concatenate((example_image, padding, undistorted_image), axis=1)

        # Plot it        
        vert_ind = int(np.floor(icam / num_horz_plots))
        horz_ind = icam - num_horz_plots * vert_ind
        
        if len(example_image.shape) == 3:
            axs[vert_ind, horz_ind].imshow(cat_image)
        else:
            axs[vert_ind, horz_ind].imshow(cat_image, cmap='gray')
            
        # Add markers
        axs[vert_ind, horz_ind].scatter(np.squeeze(example_corners[:,:,0]),
                                        np.squeeze(example_corners[:,:,1]),
                                        marker='x', color='b')
        axs[vert_ind, horz_ind].scatter(np.squeeze(undistorted_corners[:,:,0])+x_offset,
                                np.squeeze(undistorted_corners[:,:,1]),
                                marker='x', color='darkorange')
            
        axs[vert_ind, horz_ind].set_title('{}, error = {:.3f}'.format(
            cam_name, intrinsics_config['dicts'][serial]['reprojection_error'][0]))
        axs[vert_ind, horz_ind].set_xticks([])
        axs[vert_ind, horz_ind].set_yticks([])
        
    if num_axes > num_cameras:
        for i in range(num_axes - num_cameras):
            axs.flat[-(i+1)].set_visible(False)


def show_image_point_heatmap(image_size, image_points, long_ax_bins=16): 
    #TODO add documentation
    image_points = np.vstack(image_points)
    image_points = np.reshape(image_points, (image_points.shape[0], 2))
    
    short_ax_bins = int(np.ceil((np.min(image_size) / np.max(image_size)) * long_ax_bins))
    
    if image_size[1] > image_size[0]:
        x_edges = np.linspace(0,np.max(image_size), long_ax_bins)
        y_edges = np.linspace(0,np.min(image_size), short_ax_bins)
    else:
        y_edges = np.linspace(0,np.max(image_size), long_ax_bins)
        x_edges = np.linspace(0,np.min(image_size), short_ax_bins)
    
    h, x, y = np.histogram2d(image_points[:,0], image_points[:,1], bins=(x_edges, y_edges))
    X,Y = np.meshgrid(x,y)
    
    fig,ax= mpl_pp.subplots(1,2)
    fig.canvas.set_window_title('NCams: Calibration marker point distribution')
    ax[0].scatter(image_points[:,0], image_points[:,1])
    p = ax[1].pcolormesh(X,Y,h.transpose())
    
    ax1_pos = ax[1].get_position().bounds
    cb_ax_pos = [ax1_pos[0] + ax1_pos[2]*1.05, ax1_pos[1], ax1_pos[2]*0.05, ax1_pos[3]]
    cb_ax = fig.add_subplot(position=cb_ax_pos)
    fig.colorbar(p, cax=cb_ax, label='num points')
    
    for a in range(2):
        ax[a].set_xticks([x_edges[0], x_edges[-1]])
        ax[a].set_yticks([y_edges[0], y_edges[-1]])

    
    