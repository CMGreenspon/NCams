#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Please see AUTHORS for contributors.
https://github.com/CMGreenspon/NCams/blob/master/README.md
Licensed under the Apache License, Version 2.0

General camera functions and tools used for calibration, pose estimation, etc. Includes most file
I/O functions
"""

import os
import datetime
from copy import deepcopy

import glob
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as mpl_pp

from . import utils


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
        image_list = utils.get_image_list(os.path.join(folder_path, cam_names[cam]))
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
    camera_config['board_dim'], camera_config['board_type'], camera_config['folder_path']
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


#################### Accessory functions
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


#################### IO Functions
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



def calibration_to_yaml(cam_calib_filename, cam_name, cam_serial, camera_matrix,
                        distortion_coefficients, reprojection_error):
    '''Exports camera calibration info for a single camera into a YAML file.

    Arguments:
        cam_calib_filename {string} -- the filename where to save the calibration.
        cam_name {string} -- camera name.
        cam_serial {int} -- camera serial number
        camera_matrix {np.array} -- camera calibration matrix
        distortion_coefficients {np.array} -- distortion coefficients
        reprojection_error {float} -- reprojection error ??
    '''
    calibration_data = {
        'camera_name': cam_name,
        'serial_number': cam_serial,
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': distortion_coefficients.tolist(),
        'reprojection_error': reprojection_error}

    with open(cam_calib_filename, 'w') as yaml_output:
        yaml.dump(calibration_data, yaml_output, default_flow_style=False)


def yaml_to_calibration(cam_calib_filename):
    '''Imports camera calibration info for a single camera from a YAML file.

    Arguments:
        cam_calib_filename {string} -- the filename from which to load the calibration.
    Output:
        camera_matrix {np.array} -- camera calibration matrix
        distortion_coefficients {np.array} -- distortion coefficients
        reprojection_error {float} -- reprojection error ??
    '''
    temp = camera_t.load_yaml(cam_calib_filename)

    camera_matrix = np.asarray(temp['camera_matrix'])
    distortion_coefficients = np.asarray(temp['distortion_coefficients'])
    reprojection_error = np.asarray(temp['reprojection_error'])

    return camera_matrix, distortion_coefficients, reprojection_error


def export_calibration(export_filename, cam_names, cam_serials, distortion_coefficients,
                       camera_matrices, reprojection_errors):
    '''Exports camera calibration info for all cameras in the setup into a pickle file.

    Arguments:
        export_filename {string} -- the filename where to save the calibration.
        cam_names {list} -- list of strings with camera names.
        cam_serials {list} -- camera serial numbers
        camera_matrix {np.array} -- camera calibration matrices for each camera
        distortion_coefficients {np.array} -- distortion coefficients for each camera
        reprojection_errors {float} -- reprojection errors for each camera
    '''
    calibration_data = {
        'camera_names': cam_names,
        'camera_serials': cam_serials,
        'camera_matrices': camera_matrices,
        'distortion_coefficients': distortion_coefficients,
        'reprojection_errors': reprojection_errors}

    with open(export_filename, 'wb') as f:
        pickle.dump(calibration_data, f)

    print('Calibration file: ' + export_filename)


def import_calibration(filename, current_cam_serials=None):
    '''Imports camera calibration info for all cameras in the setup from a pickle file.

    Arguments:
        filename {string} -- the filename where to save the calibration.
    Keyword Arguments:
        current_cam_serials {list} -- camera serials of the cameras in the current setup in proper
            order. If not None, returned lists will adhere to this order. (default: {None})
    Output:
        reprojection_errors {float} -- reprojection errors for each camera
        cam_names {list} -- list of strings with camera names.
        camera_matrices {np.array} -- camera calibration matrices for each camera
        distortion_coefficients {np.array} -- distortion coefficients for each camera
    '''
    # Get the path name
    if isinstance(filename, str):
        path_to_file = filename
    elif isinstance(filename, dict):
        path_to_file = os.path.join(filename['folder_path'], 'cam_calibration',
                                    'camera_calib.pickle')

    # Load the file
    with open(path_to_file, 'rb') as f:
        calibration_data = pickle.load(f)

    cam_names = calibration_data['camera_names']
    cam_serials = calibration_data['camera_serials']
    camera_matrices = calibration_data['camera_matrices']
    distortion_coefficients = calibration_data['distortion_coefficients']
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
            l_cam_distortion_coeffs.append(distortion_coefficients[cam_imp_id])
            l_reprojection_errors.append(reprojection_errors[cam_imp_id])
        cam_serials = current_cam_serials
        cam_names = l_cam_names
        camera_matrices = l_camera_matrices
        distortion_coefficients = l_cam_distortion_coeffs
        reprojection_errors = l_reprojection_errors

    return (reprojection_errors, cam_names, camera_matrices, distortion_coefficients)
