'''
author(s): Charles M. Greenspon
           Anton Sobinov
lab: Sliman Bensmaia
'''
import os
import datetime
import pickle
from copy import deepcopy

import glob
import cv2
import yaml
import ImageTools
import numpy as np
import matplotlib
import matplotlib.pyplot as mpl_pp


# %% Main multi-camera functions
def multi_camera_calibration(camera_config, override=False, inspect=False,
                             calib_dir=None):
    '''CHANGED
    This will go to the specified path and for each camera isolate the images
    necessary for computing distortion coefficients.
    The coefficients will then be output as a variable for undistortion
    purposes.
    In each cameras folder a yaml file will also be saved.
    In the general cam_calibration folder a pickle file containing all
    calibrations will be saved.

    Inputs:
        camera_config dictionary containing:
            cam_names: list of camera names ['top_left', 'top_right', ...]
            board_type: as we need to know if checkerboard or charuco board
            board_dim: list with the number of checks [height, width]
            check_size = height/width of the check in mm
            folder_path: Path containing containing folders with both calibration images and pose
              estimation images. Folders should have subfolders for each camera.
        override: logit indicating whether or not to ask when other calibration
          files are detected.
        inspect: logit for whether or not to automatically call the inspection
          function
        calib_dir: path to look in if not using the default ('cam_calibration')
    Outputs: (for each camera)
        cam_reprojection_errors: average error in pixels.
        camera_matrices: the essential camera matrix.
        camera_distortion_coefficients: list of distortion coefficients.
    '''
    # Unpack the dict
    cam_names = camera_config['camera_names']
    cam_serials = camera_config['camera_serials']
    board_type = camera_config['board_type']
    board_dim = camera_config['board_dim']
    folder_path = camera_config['folder_path']

    if calib_dir is None:
        # Append as appropriate for the calibration folder
        calib_dir = os.path.join(folder_path, 'cam_calibration')

    calib_pickle_filename = os.path.join(calib_dir, 'camera_calib.pickle')

    # First check if there is a calibration file
    if os.path.exists(calib_pickle_filename) and not override:
        print('A calibration file for all cameras has been detected.'
              ' They may have already been calibrated.')
        uinput_string = (
            "Proceed with calibration anyway? "
            "(Yes/No/Abort/Override/Help')."
            "\nContinuing will replace that calibration file.\n")
        while True:
            user_input = input(uinput_string).lower()
            if user_input == 'no' or user_input == 'n':
                # Let's save time and load that instead
                (cam_reprojection_errors, cam_names, camera_matrices,
                    camera_distortion_coefficients) = import_calibration(
                    calib_pickle_filename, current_cam_serials=cam_serials)
                if inspect:
                    inspect_calibration(camera_config, camera_matrices,
                                        camera_distortion_coefficients,
                                        cam_reprojection_errors,
                                        calib_dir=calib_dir)
                return (cam_reprojection_errors, camera_matrices,
                        camera_distortion_coefficients)
            elif user_input == 'yes' or user_input == 'y':
                print('- Rerunning calibration.')
                break
            elif user_input == 'abort':
                return
            elif user_input == 'override':
                print('- Override enabled. Recalibrating all cameras.')
                override = True
                break
            elif user_input == 'help':
                print('- Yes: rerun calibration. You will be asked if you '
                      'would like to override individual files.\n'
                      '- No: load a file and exit the calibration function.\n'
                      '- Abort: exit the function without returning '
                      'anything.\n'
                      '- Override: like \'Yes\', but all files will be '
                      'overwritten automatically.\n')
            else:
                print("Invalid response given.\n")

    # Initalizes outputs
    cam_reprojection_errors = []  # Reprojection errors
    camera_matrices = []  # Intrinsic camera parameters
    camera_distortion_coefficients = []  # Distortion coefficients

    # Preliminary stuff
    num_cameras = len(cam_names)  # How many cameras are there
    print('Beginning calibration of {} camera{}.'.format(
        num_cameras, '(s)' if num_cameras > 1 else ''))

    for icam, (cam_name, cam_serial) in enumerate(zip(cam_names, cam_serials)):
        print('- Camera {} of {}.'.format(icam+1, num_cameras))

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

        # Initalize variables
        calibrate_camera = True
        camera_matrix = []
        distortion_coefficients = []

        # Check if there is already a calibration file
        cam_calib_filename = os.path.join(
            cam_calib_dir, cam_name + '_calib.yaml')
        if os.path.exists(cam_calib_filename) and not override:
            print('-> Calibration file for "{}" detected and may have already'
                  ' been calibrated.'.format(cam_name))
            uinput_string = "-- Calibrate anyway? (Yes/No/Abort).\n"
            while True:
                user_input = input(uinput_string).lower()
                if user_input == 'no' or user_input == 'n':
                    # We can instead load in the existing calibration file
                    (camera_matrix, distortion_coefficients, reprojection_error
                     ) = yaml_to_calibration( cam_calib_filename)
                    calibrate_camera = False
                    break
                elif user_input == 'yes' or user_input == 'y':
                    break
                elif user_input == 'abort':
                    return
                else:
                    print("Invalid response given.\n")

        if calibrate_camera:
            # Check if there are images
            cam_image_list = ImageTools.get_image_list(path=cam_calib_dir)
            if len(cam_image_list) == 0:
                # If there are no cameras append empty arrays so that the
                # index order is preserved and begin looking at next camera
                print('-> No images found in directory "{}" for camera {}.\n'
                      ' Continuing to the next camera...'.format(
                        cam_calib_dir, cam_name))
                cam_reprojection_errors.append([])
                camera_distortion_coefficients.append([])
                camera_matrices.append([])
                continue
            elif len(cam_image_list) < 25:
                print('  Only {} images found. Calibration may be poor.'.format(
                        len(cam_image_list)))

            # Get coefficients and matrices for each camera
            if board_type == 'checkerboard':
                world_points = create_world_points(camera_config)
                # Run the calibration:
                (reprojection_errors, camera_matrix,
                 distortion_coefficients) = checkerboard_calibration(
                    cam_image_list, board_dim, world_points)
            elif board_type == 'charuco':
                # Create the board - world points included
                charuco_dict, charuco_board, _ = create_board(
                    camera_config)
                # Run the calibration:
                (reprojection_errors, camera_matrix,
                 distortion_coefficients) = charuco_calibration(
                    cam_image_list, charuco_dict, charuco_board)

            # Export them to the camera folder in a readable format
            calibration_to_yaml(cam_calib_filename, cam_name, cam_serial, camera_matrix,
                                distortion_coefficients, reprojection_errors)

        # Return these to the workspace
        cam_reprojection_errors.append(reprojection_errors)
        camera_distortion_coefficients.append(distortion_coefficients)
        camera_matrices.append(camera_matrix)

    print('* Calibration complete.')
    export_calibration(calib_pickle_filename, cam_names, cam_serials,
                       camera_distortion_coefficients, camera_matrices, cam_reprojection_errors)

    if inspect is True:
        inspect_calibration(camera_config, camera_matrices, camera_distortion_coefficients,
                            cam_reprojection_errors, calib_dir=calib_dir)

    return (cam_reprojection_errors, camera_matrices, camera_distortion_coefficients)


def TODO_multi_camera_pose_estimation():

    num_cameras = len(camera_config['camera_names'])
    if num_cameras == 1:
        raise Exception('Only one camera present, pose cannot be calculated with this function.')
        return []


#%% Camera calibration functions
def checkerboard_calibration(cam_image_list, board_dim, world_points):
    '''
    Attempts to find a checkerboard in each image and performs the basic calibration.
    It is suggested to use the inspect_calibration tool to check if the calibration is good.

    Inputs:
        cam_image_list: list of images to search for checkerboards in.
        board_dim: size of the board (cols, rows)
        world_points: ground truth x,y,z values.
    Outputs:
        reprojection_errors: float indicating the average error of the calibration in pixels.
        camera_matrix: the calculated intrinsics of the camera
        distortion_coeffictions: the calculated distortion parameters for the lens.
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Default criteria

    # Analyze the images to get checkerboard corners
    object_points = [] # 3d points in space
    image_points = [] # x,y image points
    board_logit = np.zeros((1,len(cam_image_list)), dtype = bool)

    for im in range(len(cam_image_list)):
        im_name = cam_image_list[im]
        img = cv2.imread(im_name, 0) # Load as grayscale & find the corners
        board_logit, corners = cv2.findChessboardCorners(img, (board_dim[0]-1,board_dim[1]-1), None)

        # If a checkerboard was found then append the world points and image points
        if board_logit:
            object_points.append(world_points)
            corners_refined = cv2.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
            image_points.append(corners_refined)

    img_width, img_height = img.shape[1], img.shape[0] # Necessary for the calibration step
    # Calibrate
    reprojection_errors, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
            object_points, image_points, (img_width,img_height), None, None)
    return reprojection_errors, camera_matrix, distortion_coefficients


def charuco_calibration(cam_image_list, charuco_dict, charuco_board):
    '''
    Read multi_camera_calibration for info.
    Only exists separately so that if you want to quickly calibrate a single camera based on charuco-board images you can.
    '''
    # Initalize
    ch_ids = [] # charuco ID list
    image_points = [] # x,y coordinates of charuco corners
    board_logit = np.zeros((1,len(cam_image_list)), dtype = bool)

    calib_flags = 0
    calib_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    calib_flags |= cv2.CALIB_ZERO_TANGENT_DIST

    # Iterate through each image and find corners & IDs
    for im in range(len(cam_image_list)):
        im_name = cam_image_list[im]
        img = cv2.imread(im_name, 0) # Load as grayscale
        corners, ids, _ = cv2.aruco.detectMarkers(img, charuco_dict) # Detect the aruco markers and get IDs
        if ids is not None: # Only try to find corners if markers were detected
            board_logit[0,im] = True # Keep track of detections
            # Find the corners and IDs
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, img, charuco_board)
            # In some instances a 'NoneType' is produced - this causes issues
            if isinstance(charuco_corners, np.ndarray):
              # If there are too few points this also won't work
              if len(charuco_corners[:,0,0]) > 4:
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
        [f_length,0,img_width//2],
        [0, f_length, img_height//2],
        [0, 0, 1]
        ])

    reprojection_error, camera_matrix, distortion_coefficients, _, _,  = cv2.aruco.calibrateCameraCharuco(
        image_points, ch_ids, charuco_board, (img_width,img_height), principal_point_init, None, flags = calib_flags)

    # Check output format - seems to be version dependent
    if type(camera_matrix) == cv2.UMat:
        camera_matrix = camera_matrix.get()
    if type(distortion_coefficients) == cv2.UMat:
        distortion_coefficients = distortion_coefficients.get()

    return reprojection_error, camera_matrix, distortion_coefficients

#%% Pose estimation functions
### Automatic scripts
def WORKING_auto_pose_estimation(camera_config, reference_camera):
    # Unpack the dict
    board_type = camera_config['board_type']
    num_cameras = len(camera_config['camera_names'])

    if board_type == 'charuco':
        # Create the board
        _, charuco_board, _ = CameraTools.create_board(camera_config)
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


### Board detectors
def charuco_board_detector(camera_config):
    # Unpack the dict
    cam_names, board_dim, check_size, folder_path = camera_config['camera_names'], camera_config['board_dim'], \
    camera_config['check_size'], camera_config['folder_path']
    folder_path = os.path.join(folder_path, 'pose_estimation')
    # Get number of cameras
    num_cameras = len(cam_names)
    charuco_dict, charuco_board, _ = create_board(camera_config)
    # Get list of images for each camera
    cam_image_list = []
    num_images = np.zeros((1, num_cameras), dtype = int)
    for cam in range(num_cameras):
        image_list = ImageTools.get_image_list(os.path.join(folder_path, cam_names[cam]))
        num_images[0,cam] = len(image_list)
        cam_image_list.append(image_list)
    # Crucial: Each camera must have the same number of images so that we can assume the order is maintained and that they are synced
    if np.ma.allequal(num_images, np.mean(num_images)) == False:
        raise Exception('Image lists are of unequal size and may not be synced.')

    num_images = num_images[0,0]
    cam_image_points, cam_charuco_ids = [], []
    # Look at one synced image across cameras and find the matching points
    for image in range(num_images):
      im_ids, image_points = [], [] # reset for each image
      for cam in range(num_cameras):
        img = cv2.imread(os.path.join(folder_path, cam_names[cam], cam_image_list[cam][image])) # Load the image
        corners, ids, rejected_points = cv2.aruco.detectMarkers(img, charuco_dict) # Detect the aruco markers and get IDs
        if ids is not None:
            # Find the corners and IDs
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, img, charuco_board)
            if isinstance(charuco_corners, np.ndarray): # If present then append
                image_points.append(charuco_corners)
                im_ids.append(charuco_ids)
            else: # For formatting/indexing
                image_points.append([])
                im_ids.append([])
        else:
            image_points.append([])
            im_ids.append([])
      # Concatenate them to get super list which can be parsed later
      cam_image_points.append(image_points)
      cam_charuco_ids.append(im_ids)

    return cam_image_points, cam_charuco_ids


def checkerboard_detector(camera_config, override = False):
    '''
    Get all image points and determine which calibration mode is better.
    Can only be run after cameras have been calibrated.
    Inputs: Dict containing
        cam_names: list of camera names ['cam1', 'cam2', ...]
        board_dim: list with the number of checks [height, width]
        check_size: height/width of the check in mm
        folder_path: Path containing a 'cam_calibration' folder with subfolders with camera names that contain images of boards.
        board_type: String indicating whether the board used is a checkerboard or a charuco-board.
    Outputs:
        cam_board_logit:
          if checkerboard: logical array (num_cameras, num_images) indicating in which images each camera detected a checkerboard.
        cam_image_points:
          if checkerboard: array of image points (num_cameras, image, (x,y))
        pose_strategy:
          string indicating which pose estimation strategy is ideal.
    '''
    # Unpack the dict
    cam_names, board_dim, board_type, folder_path = camera_config['camera_names'], \
    camera_config['board_dim'], camera_config['board_type'], camera_config['folder_path'], \
    num_cameras = len(cam_names) # How many cameras are there
    # Get the correct folder
    cam_pose_path = os.path.join(folder_path, 'pose_estimation') # Append as appropriate for the pose folder
    os.chdir(cam_pose_path)
    # First check if there is a pose estimation file
    if os.path.exists('pose_estimation.pickle') and override is False:
        print('A pose estimation file for has been detected in the specified path.')
        user_input = input("Would you like to load that file instead? 'Yes', 'No', or 'Abort'.\
                           \nContinuing will replace that file.\n").lower()
        valid_input = False
        while valid_input == False:
            if user_input == 'yes': # Let's save the time and load that instead
                #something, something = import_pose_estimate(os.path.join(cam_pose_path, 'camera_calib.pickle'))
                valid_input = True
                #return camera_matrices, camera_distortion_coefficients
            elif user_input == 'no':
                print('- Rerunning pose estimation.')
                valid_input = True
            elif user_input == 'abort':
                valid_input = True
                break
            else:
                user_input = input("Invalid response given.\nLoad previous pose estimation? 'Yes', 'No', 'Abort'.\n").lower()

    # Begin the checkerboard detection for each camera
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # Default criteria
    cam_board_logit = []
    cam_image_points = []

    print('Beginning checkerboard detection.')
    for cam in range(len(cam_names)):
        print('- Camera ' + str(cam+1) + ' of', str(num_cameras) +'.')
        os.chdir(os.path.join(cam_pose_path, cam_names[cam]))
        cam_image_list = glob.glob('*.png') + glob.glob('*.jpg') + glob.glob('*.jpeg') + glob.glob('*.bmp')
        # Analyze the images to get checkerboard corners
        image_points = [] # x,y image points
        board_logit = np.zeros((1,len(cam_image_list)), dtype = bool)

        for im in range(len(cam_image_list)):
            im_name = cam_image_list[im]
            img = cv2.imread(im_name, 0) # Load as grayscale
            board_logit[0,im], corners = cv2.findChessboardCorners(img, (board_dim[0]-1,board_dim[1]-1), None)

            # If a checkerboard was found then append the image points variable for calibration
            if board_logit[0,im]:
                corners_refined = cv2.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
                image_points.append(corners_refined)
            else:
                image_points.append([]) # To keep consistent with the board_logit list

        # Add exports to list structure
        cam_board_logit.append(board_logit)
        cam_image_points.append(image_points)

    print('* Checkerboard detection complete.')

    combined_board_logit = np.sum(np.vstack(cam_board_logit),0) # Combine and sum the logits
    num_common_cb = np.sum(combined_board_logit == num_cameras) # See how many checkerboard detections are present across all cameras

    if num_common_cb < 10:
        pose_strategy = 'stereo_sequential'
    elif num_common_cb >= 10:
        pose_strategy = 'common'

    print('* Optimal pose strategy: \"' + pose_strategy + '\".')

    return cam_board_logit, cam_image_points, pose_strategy


### Pose estimation methods
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
    im_list = ImageTools.get_image_list(pose_estimation_path)

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


#%% Accessory functions
def make_projection_matrix(camera_matrix, orientation, translation):
    # Make matrix if necessary
    if orientation.shape == (3,1) or orientation.shape == (1,3):
        import cv2
        orientation,_ = cv2.Rodrigues(orientation) # Convert to matrix

    if translation.shape == (1,3): # Format
        translation = np.transpose(translation)

    projection_matrix = temp = np.matmul(camera_matrix, np.hstack((orientation, translation)))

    return projection_matrix


def adjust_stereo_calibration_origin(world_rotation_vector, world_translation_vector, relative_rotations, relative_translations):
    adjusted_rotation_vectors, adjusted_translation_vectors = [],[]

    # Format rotation for composeRT
    if world_rotation_vector.shape == (3,3):
            world_rotation_vector,_ = cv2.Rodrigues(world_rotation_vector)

    for cam in range(len(relative_rotations)):
        sec_r_vec = relative_rotations[cam]
        # Format rotation for composeRT
        if sec_r_vec.shape == (3,3):
            sec_r_vec,_ = cv2.Rodrigues(sec_r_vec)

        adjusted_orientation, adjusted_location,_,_,_,_,_,_,_,_ = cv2.composeRT(
                world_rotation_vector, world_translation_vector,
                sec_r_vec, relative_translations[cam])

        adjusted_rotation_vectors.append(adjusted_orientation)
        adjusted_translation_vectors.append(adjusted_location)

    return adjusted_rotation_vectors, adjusted_translation_vectors


### Board functions
def create_board(camera_config, output = False, first_marker = 0, plotting = False,
                 dpi = 300, output_format = 'pdf', padding = 0, target_size = None,
                 dictionary = None):
    '''
    Creates an aruco board image that can be printed and used for camera calibration & pose estimation.
    '''
    # Unpack dict
    board_type = camera_config['board_type']
    board_dim = camera_config['board_dim']
    check_size = camera_config['check_size']
    if output:
        output_path = camera_config['folder_path']

    dpmm = dpi / 25.4 # Convert inches to mm

    # Make the dictionary
    total_markers = int(np.floor((board_dim[0] * board_dim[1]) / 2))

    # Make the board & array for image
    board_width = (board_dim[0] * check_size)
    board_height = (board_dim[1] * check_size)
    board_img = np.zeros((int(board_width * dpmm) + board_dim[0], int(board_height * dpmm) + board_dim[1]))

    if board_type == 'checkerboard':
        # Litearlly just tile black and white squares
        check_length_in_pixels = int(np.round(check_size * dpmm))
        black_check = np.ones((check_length_in_pixels,check_length_in_pixels)) * 255
        white_check = np.zeros((check_length_in_pixels,check_length_in_pixels))
        board_img = np.empty((0,check_length_in_pixels*board_dim[0]), int)

        idx = 1
        for r in range(board_dim[1]):
            col = np.empty((check_length_in_pixels,0), int)
            for c in range(board_dim[0]):
                if idx % 2 == 0:
                    col = np.append(col, black_check, axis = 1)
                else:
                    col = np.append(col, white_check, axis = 1)

                idx += 1

            board_img = np.append(board_img, col, axis = 0)

    elif board_type == 'charuco':
        if dictionary is None:
            output_dict = cv2.aruco.Dictionary_create(total_markers,5)
        else:
            custom_dict = cv2.aruco.Dictionary_get(dictionary)
            output_dict = cv2.aruco.Dictionary_create_from(total_markers, custom_dict.markerSize, custom_dict)

        secondary_length = check_size * 0.6 # What portion of the check the aruco marker takes up
        output_board = cv2.aruco.CharucoBoard_create(board_dim[0], board_dim[1], check_size/100, secondary_length/100, output_dict)
        # The board is compiled upside down so the top of the image is actually the bottom, to avoid confusion it's rotated below
        board_img = np.rot90(output_board.draw((int(board_width * dpmm), int(board_height * dpmm)), board_img, 1, 1),2)
    else:
        print('Invalid \"board_type\" given.')
        return []

    if plotting:
        fig, ax = mpl_pp.subplots()
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
        larger_board_img[r_off:r_off + board_img.shape[0], c_off:c_off + board_img.shape[1]] = board_img
        board_img = larger_board_img

    if output is True:
      if output_format == 'pdf':
          output_name = os.path.join(output_path, board_type + '_board.png')
          cv2.imwrite(output_name, board_img)
          from reportlab.lib.pagesizes import letter
          from reportlab import platypus
          from reportlab.lib.units import mm
          # To vertically center the board
          diff_in_vheight = ((letter[1]/72)*25.4 - board_height) / 2
          # Start building
          elements = []
          doc = platypus.SimpleDocTemplate(os.path.join(output_path, "charuco_board.pdf"), pagesize=letter,
                                           topMargin = 0, bottomMargin = 0)
          elements.append(platypus.Spacer(1,diff_in_vheight*mm))
          board_element = platypus.Image(output_name)
          board_element.drawWidth = board_width*mm
          board_element.drawHeight = board_height*mm
          elements.append(board_element)
          doc.build(elements)
      else:
          cv2.imwrite(os.path.join(output_path, board_type + '_board.' + output_format), board_img)

    if board_type == 'checkerboard':
        return [], [], board_img
    elif board_type == 'charuco':
        return output_dict, output_board, board_img


def create_world_points(camera_config):
    board_type, board_dim, check_size = camera_config['board_type'], camera_config['board_dim'], camera_config['check_size']

    if board_type == 'checkerboard':
      world_points = np.zeros(((board_dim[0]-1) * (board_dim[1]-1),3), np.float32) # x,y,z points
      world_points[:,:2] = np.mgrid[0:board_dim[0]-1,0:board_dim[1]-1].T.reshape(-1,2) # z is always zero
      world_points = world_points * check_size
    elif board_type == 'charuco':
      _, charuco_board,_ = create_board(camera_config)
      nc, _ = charuco_board.chessboardCorners.shape
      world_points = charuco_board.chessboardCorners.reshape(nc,1,3)

    return world_points


### Inspection of calibration/pose
def inspect_calibration(camera_config, camera_matrices, distortion_coefficients,
                        reprojection_error, image_index=None, calib_dir=None):
    cam_names = camera_config['camera_names']
    board_type = camera_config['board_type']
    board_dim = camera_config['board_dim']
    folder_path = camera_config['folder_path']
    if calib_dir is None:
        folder_path = os.path.join(folder_path, 'cam_calibration')
    else:
        folder_path = calib_dir

    num_markers = (board_dim[0]-1) * (board_dim[1]-1)
    # Get layout of output array
    num_cameras = len(cam_names)
    num_vert_plots = int(np.ceil(np.sqrt(num_cameras)))
    num_horz_plots = int(np.ceil(num_cameras/num_vert_plots))

    fig, axs = mpl_pp.subplots(num_horz_plots, num_vert_plots, squeeze=False)
    vert_ind, horz_ind = 0,0
    # Get the images and plot for each camera
    for cam in range(num_cameras):
        # Folder navigation
        if num_cameras == 1: # Check if images are in current directory or in a subdirectory
            if os.path.exists(os.path.join(folder_path, cam_names[0])):
                new_path = os.path.join(folder_path, cam_names[0]) # Go to the subdirectory
            else:
                new_path = folder_path # Stay where we are
        else: # If there is more than one camera assume subdirectories are present
            new_path = os.path.join(folder_path, cam_names[cam]) # Go to the subdirectory

        # Get the appropriate camera matrices
        cam_mat = camera_matrices[cam]
        dist_coeffs = distortion_coefficients[cam]

        image_list = ImageTools.get_image_list(path=new_path, sort=True)

        board_in_image = False
        idx = 0
        while not board_in_image:
            if image_index is None:
                image_ind = idx # Pick a random image
            else:
                image_ind = image_index
            example_image = matplotlib.image.imread(image_list[image_ind])
            # Get new camera matrix
            h, w = example_image.shape[:2]
            new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(
                cam_mat, dist_coeffs, (w, h), 1, (w, h))
            if board_type == 'charuco':
                # Detect the markers
                charuco_dict, charuco_board, _ = create_board(camera_config)
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
                            undistorted_image = ImageTools.undistort_image(
                                example_image, cam_mat, dist_coeffs)
                            undistorted_image_annotated = cv2.aruco.drawDetectedCornersCharuco(
                                undistorted_image, undistorted_corners)
                        elif image_index is not None:
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
                    undistorted_image = ImageTools.undistort_image(
                        example_image, cam_mat, dist_coeffs)
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
            cam_names[cam], reprojection_error[cam]))
        axs[vert_ind, horz_ind].set_xticks([])
        axs[vert_ind, horz_ind].set_yticks([])

        horz_ind += 1
        if horz_ind > (num_horz_plots-1):
            horz_ind = 0
            vert_ind += 1


def TODO_inspect_pose_estimation():
    import matplotlib.pyplot

    image_index = 10
    cam_indices = [0, 1]

    # Load the images
    im_list1 = ImageTools.get_image_list(os.path.join(camera_config['folder_path'],
                                         'pose_estimation', cam_names[cam_indices[0]]))
    im1 = matplotlib.image.imread(os.path.join(camera_config['folder_path'], 'pose_estimation',
                                  cam_names[cam_indices[0]], im_list1[image_index]))

    im_list2 = ImageTools.get_image_list(os.path.join(camera_config['folder_path'],
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

    cam_image_points, cam_charuco_ids = CameraTools.charuco_board_detector(camera_config)
    R, T = CameraTools.WORKING_common_pose_estimation(camera_config, cam_image_points, cam_charuco_ids, camera_matrices, distortion_coefficients)

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


### Camera plotting helper functions
def create_camera(scale_factor = 1, rotation_vector = None, translation_vector = None):
    '''Create a typical camera shape.'''
    cam_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0],  [0, 1, 0], # Open side of lens
                              [0.2, 0.2, 0.5], [0.8, 0.2, 0.5], [0.8, 0.8, 0.5],  [0.2, 0.8, 0.5], # Front of camera body/back of lens
                              [0.2, 0.2, 1], [0.8, 0.2, 1], [0.8, 0.8, 1],  [0.2, 0.8, 1]]) # Back of camera body

    # Set the origin as the back of the lens
    centering_vector = [0.5, 0.5, 0.5]
    cam_points = cam_points - centering_vector
    # Scale the points
    cam_points = cam_points * scale_factor
    # Move the camera
    cam_points = move_camera(cam_points, rotation_vector, translation_vector)
    # Get the vertices & center
    camera_vertices = get_camera_vertices(cam_points)
    cam_center = np.mean(cam_points[4:8,:],0)
    cam_center[1] = cam_center[1] + scale_factor

    return camera_vertices, cam_center

def move_camera(cam_points, rotation_vector = None, translation_vector = None):
    '''Applies the appropriate rotation and translation to the camera points.'''
    # Check rotation vector format
    if rotation_vector is None:
        rotation_vector = np.identity(3) # Assume it's not rotating
    elif rotation_vector.shape == (3,1) or rotation_vector.shape == (1,3): # Make matrix if necessary
        import cv2
        rotation_vector,_ = cv2.Rodrigues(rotation_vector) # Convert to matrix

    if translation_vector is None:
        translation_vector = np.zeros((3,1)) # Assume there is no translation
    elif translation_vector.shape == (1,3):
        translation_vector = np.transpose(translation_vector) # Format

    translation_vector = np.matmul(-np.transpose(rotation_vector), translation_vector) # Create the translation vector

    # Rotate and then translate
    cam_points = np.transpose(np.matmul(np.transpose(rotation_vector), np.transpose(cam_points)))
    cam_points = cam_points - np.transpose(translation_vector)

    return cam_points

def get_camera_vertices(cam_points):
    '''Litearlly just a manual mapping of the camera points from in create_camera.'''
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


#%% IO Functions
def load_camera_config(camera_config):
    '''Searches the subdirectories of the folder path in the camera_config dict
    for relevant pickles.

    CHANGED to search more files'''
    # Initalize variables just in case we need to export blanks
    camera_matrices = None
    distortion_coefficients = None
    reprojection_errors = None
    world_locations = None
    world_orientations = None

    # Get paths
    if isinstance(camera_config, dict):
        folder_path = camera_config['folder_path']
    elif isinstance(camera_config, str):
        folder_path = camera_config

    camera_calib_file_paths = [
        os.path.join(folder_path, 'cam_calibration', 'camera_calib.pickle'),
        os.path.join(folder_path, 'calibration', 'camera_calib.pickle'),
        os.path.join(folder_path, 'camera_calib.pickle')]
    pose_estimation_file_paths = [
        os.path.join(folder_path, 'pose_estimation', 'pose_estimate.pickle'),
        os.path.join(folder_path, 'pose_estimate.pickle')]

    # Load them if they exist
    for ccfp in camera_calib_file_paths:
        if os.path.isfile(ccfp):
            (reprojection_errors, _, camera_matrices, distortion_coefficients
             ) = import_calibration(
             ccfp, current_cam_serials=camera_config['camera_serials'])
            print('Camera calibration loaded.')
            break
    if camera_matrices is None:
        print('No camera calibration file found.')

    for pefp in pose_estimation_file_paths:
        if os.path.isfile(pefp):
            (_, world_locations, world_orientations
             ) = import_pose_estimation(pefp)
            print('Pose estimation loaded.')
            break
    if world_locations is None:
        print('No pose estimation file found.')

    return (camera_matrices, distortion_coefficients, reprojection_errors,
            world_locations, world_orientations)


def config_to_yaml(camera_config, prefix='', output_path=None):
    '''CHANGED'''
    date = datetime.date.today()

    formatted_data = deepcopy(camera_config)
    formatted_data['date'] = date
    formatted_data['image_size'] = list(formatted_data['image_size'])

    if output_path is None:
        output_path = camera_config['folder_path']

    filename = os.path.join(output_path, prefix + str(date) + '_config.yaml')

    with open(filename, 'w') as yaml_output:
        yaml.dump(formatted_data, yaml_output, default_flow_style=False)


def yaml_to_config(path_to_file):
    temp = load_yaml(path_to_file)

    temp['camera_serials'] = temp['camera_serials']
    temp['image_size'] = (*temp['image_size'],)

    return temp


### Camera calibration IO functions
def load_yaml(path_to_file):
    with open(path_to_file, 'r') as stream:
        data = yaml.safe_load(stream)

    return data


def calibration_to_yaml(cam_calib_filename, cam_name, cam_serial,
                        camera_matrix, distortion_coefficients,
                        reprojection_error):
    calibration_data = {
        'camera_name': cam_name,
        'serial_number': cam_serial,
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': distortion_coefficients.tolist(),
        'reprojection_error': reprojection_error}

    with open(cam_calib_filename, 'w') as yaml_output:
        yaml.dump(calibration_data, yaml_output, default_flow_style=False)


def yaml_to_calibration(cam_calib_filename):
    temp = load_yaml(cam_calib_filename)

    camera_matrix = np.asarray(temp['camera_matrix'])
    distortion_coefficients = np.asarray(temp['distortion_coefficients'])
    reprojection_error = np.asarray(temp['reprojection_error'])

    return camera_matrix, distortion_coefficients, reprojection_error


def export_calibration(export_filename, cam_names, cam_serials,
                       distortion_coefficients, camera_matrices,
                       reprojection_errors):
    calibration_data = {
        'camera_names': cam_names,
        'camera_serials': cam_serials,
        'camera_matrices': camera_matrices,
        'distortion_coefficients': distortion_coefficients,
        'reprojection_errors': reprojection_errors}

    with open(export_filename, 'wb') as f:
        pickle.dump(calibration_data, f)

    print('Calibration file: ' + export_filename)


def import_calibration(path_to_file_OR_dict, current_cam_serials=None):
    # Get the path name
    if isinstance(path_to_file_OR_dict, str):
        path_to_file = path_to_file_OR_dict
    elif isinstance(path_to_file_OR_dict, dict):
        path_to_file = os.path.join(path_to_file_OR_dict['folder_path'],
                                    'cam_calibration', 'camera_calib.pickle')

    # Load the file
    with open(path_to_file, 'rb') as f:
        calibration_data = pickle.load(f)

    cam_names = calibration_data['camera_names']
    cam_serials = calibration_data['camera_serials']
    camera_matrices = calibration_data['camera_matrices']
    cam_distortion_coeffs = calibration_data['distortion_coefficients']
    reprojection_errors = calibration_data['reprojection_errors']

    # change order of calibration to match the current system
    if current_cam_serials is not None:
        l_cam_names = []
        l_camera_matrices = []
        l_cam_distortion_coeffs = []
        l_reprojection_errors = []
        for cam_serial in current_cam_serials:
            cam_imp_id = cam_serials.index(cam_serial)
            l_cam_names.append(cam_names[cam_imp_id])
            l_camera_matrices.append(camera_matrices[cam_imp_id])
            l_cam_distortion_coeffs.append(cam_distortion_coeffs[cam_imp_id])
            l_reprojection_errors.append(reprojection_errors[cam_imp_id])
        cam_serials = current_cam_serials
        cam_names = l_cam_names
        camera_matrices = l_camera_matrices
        cam_distortion_coeffs = l_cam_distortion_coeffs
        reprojection_errors = l_reprojection_errors

    return (reprojection_errors, cam_names, camera_matrices,
            cam_distortion_coeffs)


### Pose estimation IO functions
def export_pose_estimation(export_path, cam_serials, world_locations, world_orientations):
    pose_data = dict(
        camera_serials = cam_serials,
        world_locations = world_locations,
        world_orientations = world_orientations)

    filename = os.path.join(export_path, 'pose_estimate.pickle')
    with open(filename, 'wb') as pickle_file:
        pickle.dump(pose_data, pickle_file)

    print('Pose-estimation file: ' + filename)

def import_pose_estimation(path_to_file_OR_dict):
    # Get the path name
    if isinstance(path_to_file_OR_dict, str):
        path_to_file = path_to_file_OR_dict
    elif isinstance(path_to_file_OR_dict, dict):
        path_to_file = os.path.join(path_to_file_OR_dict['folder_path'], 'pose_estimate.pickle')

    # Load the file
    with open(path_to_file, 'rb') as pickle_file:
        pose_data = pickle.load(pickle_file)

    camera_serials = pose_data['camera_serials']
    world_locations = pose_data['world_locations']
    world_orientations = pose_data['world_orientations']

    return camera_serials, world_locations, world_orientations
