"""
NCams Toolbox
Copyright 2020 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Example script for setting up a system. Goes through file formatting and both intrinsic and
extrinsic calibration. This also includes measuring how accurate the calibration steps are.
The following examples are given:
    1. Creating an ncams_config dictionary
    2. Creation of a charucoboard to use for calibration
    3. Camera intrinsic calibration and inspection
    4. Camera extrinsic estimation and inspection
        4a. One-shot multi PnP
        4b. Sequential-stereo
    5. Loading an existing setup
    6. Calibrating an individual camera
    
For more details on the camera data structures and dicts, see help(ncams.camera_tools).
"""

import ncams
import os
import time
import numpy as np
import cv2

# Replace this with your working directory
BASE_DIR = r'C:\Repositories\NCamsExampleDataExternal\IntrinsicExtrinsicCalibration'

#%% 1. Create the ncams_config
ncams_config = {
    # Camera information
    'serials': [19194005,19194008,19194009,19194013,19335177,19340298,19340300,19340396], #
    'reference_camera_serial': 19335177,  # This is linked to how the cameras are hardwired
    'image_size': (1080, 1440),  # height x width 
    # Board information
    'board_type': 'charuco',  # charuco (preferred) or checkerboard,
    'board_dictionary': None, # Will use the default dictionary but a custom one may be specified
    'board_dim': [5, 7],  # If this is incorrect it will cause analyses to freeze
    'check_size': 30, # Size of the checks in mm, essential for accurate 3D reconstructions
    'world_units': 'mm', # Determines how to scale the world ('m', 'dm', 'cm', 'mm')
    # Path information
    'setup_path': BASE_DIR, # Where to store this configuration
    'setup_filename': 'ncams_config.yaml', # Desired filename for the configuration
    'intrinsic_path': 'intrinsic', # Name of the subdirectory for the intrinsic calibration data
    'intrinsic_filename': 'intrinsic_calib.pickle', # Desired filename for the intrinsics
    'extrinsic_path': 'extrinsic', # Name of the subdirectory for the extrinsic calibration data
    'extrinsic_filename': 'extrinsic_calib.pickle'}

ncams.camera_io.config_to_yaml(ncams_config) # Export the config

#%% 2. Create a board for calibration purposes
ncams.camera_tools.create_board(ncams_config, output=True, output_format='jpg', plotting=True)
# Note that this function will create a checkerboard or charucoboard depending on the ncams_config
''' The charucoboard may be printed on a piece of paper though the scaling for this is not 
guaranteed to be accurate. If this is desired then set the output format to pdf. Otherwise,
we recommend printing the image on metal with a matte coating (satin) to prevent glare. We do not
specifically endorse any company but have used bay photo (https://www.bayphoto.com/) with great
success. We also recommend testing different sized checks depending on the setup. With our setup
and the distance between the board and the cameras 40 mm checks were optimal.'''

#%% 3. Intrinsic calibration
'''
The below text describes how the folder structure should be formatted so that NCams knows where to 
look for the relevant images. The exact subfolder names are defined in the ncams_config above. These
can be changed but must still be in the same directory:
Setup path
    intrinsic_path
        cam1_folder
            cam1_image1
            cam1_image2
        cam2_folder
        ...
        camN_folder
        
The 'multi_camera_intrinsic_calibration' function takes the config, assumes the above folder structure
and will attempt to determine the intrinsic calibration of each camera. It is imperative to inspect
the calibration to determine if the steps were successful as without it subsequent extrinsic calibration
and triangulation efforts will fail.

The override parameter will force calibration to occur even if a pre-existing calibration file is
detected.
'''

intrinsics_config = ncams.camera_calibration.multi_camera_intrinsic_calibration(
    ncams_config, override=False, inspect=False, export_full=True, verbose=True)
# Equivalent of inspect=True
ncams.camera_calibration.inspect_intrinsics(ncams_config, intrinsics_config)

#%% 4a. Extrinsic calibration: one_shot_multi_PnP
'''
If all cameras have an overlapping view then it is possible to determine their relative locations
from a single synchronized image. If this is not the case one must use 'sequential_stereo' extrinsic
calibration instead (4b). When this function is called it assumes that the extrinsic path has no sub-
directories and only has one image for each camera:
Setup path
    extrinsic_path
        cam1_image
        cam2_image
        ...
        camN_image
'''

ncams_config['extrinsic_path'] = 'Extrinsic_OS'

extrinsics_config, extrinsics_info = ncams.camera_pose.one_shot_multi_PnP(
    ncams_config, intrinsics_config, export_full=True, show_extrinsics=False, inspect=False,
    ch_ids_to_ignore=None)
# Equivalent of inspect=True and show_extrinsics=True
ncams.camera_pose.plot_extrinsics(extrinsics_config, ncams_config)
ncams.camera_pose.inspect_extrinsics(ncams_config, intrinsics_config, extrinsics_config,
                                     extrinsics_info, error_threshold=0.1, world_points=None)

ncams.camera_io.config_to_yaml(ncams_config) # Export the config with Extrinsic_OS as extrinsic_path

#%% 4b. Extrinsic calibration: sequential-stereo
'''
When all cameras do not have a shared view of a planar board - for example they are facing each 
other - their relative poses cannot be computed in a single step without a 3D calibration object.
Instead, if multiple adjacent cameras have a shared view (each camera shares a view with 2 others) 
then the relative positions between each pair of cameras can be combined (daisy chained). This, 
however, comes with the risk of accumulating errors in the relative position. Consequently, 
the fewer links needed to  determine the relative position between the reference camera and all
other cameras the better. Therefore it is ideal to use a central camera as the reference camera.
If 'daisy_chain' is not enabled then instead every camera will be stereo-calibrated with the 
reference camera.

The format of the image directory must be either:
Setup path
    extrinsic_path
        cam1_image1
        cam1_image2
        cam1_imageN
        cam2_image1
        ...
        camN_imageN
        
OR
Setup path
    extrinsic_path
        cam1
            cam1_image1
            cam1_image2
            cam1_imageN
        cam2
            cam2_image1
        ...
        camN
            camN_imageN
            
It is imperative that the image number for each camera reflects the time point that each image was 
taken at. That is to say that cam1_image10 and cam3_image10 must have been taken at the same time.
'''
ncams_config['extrinsic_path'] = 'Extrinsic_SS'

cam_image_points, cam_charuco_ids = ncams.camera_pose.charuco_board_detector(ncams_config)

extrinsics_config = ncams.camera_pose.sequential_stereo_estimation(
    ncams_config, intrinsics_config, cam_image_points, cam_charuco_ids, daisy_chain=True,
    max_links=3, matching_threshold=1250, export_full=True, show_extrinsics=False)
ncams.camera_pose.plot_extrinsics(extrinsics_config, ncams_config)


#%% 5. Loading an existing setup
'''
Once calibration has been completed the values can be revisited for inspection/sanity checking.
These are also the steps necessary for other examples (included there too) for triangulation,
video creation, inverse kinematics, etc.
'''

# Path of the ncams dictionary as described/created in (1)
path_to_ncams_config = os.path.join(BASE_DIR, 'ncams_config.yaml')
# Load the dictionary which contains all necessary info for the system
ncams_config = ncams.camera_io.yaml_to_config(path_to_ncams_config)
# Import existing calibrations - note 
intrinsics_config, extrinsics_config = ncams.camera_io.load_calibrations(ncams_config)
# See if the imported calibrations are sensible
ncams.camera_calibration.inspect_intrinsics(ncams_config, intrinsics_config)
ncams.camera_pose.plot_extrinsics(extrinsics_config, ncams_config)


#%% 6. Calibrating an individual camera
'''
If one wishes to merely calculate the camera_matrix and/or distortion_coefficients without 
keeping values in an 'intrinsics_config' then the values can be calculated as shown below.

Note: the 'inspect_intrinsics' function cannot be called here as no 'intrinsics_config' has been
created. The reprojection error and the exported marked images are the appropriate way to determine
if an individual camera has been calibrated properly.
'''

# Path of the ncams dictionary as described/created in (1)
path_to_ncams_config = os.path.join(BASE_DIR, 'ncams_config.yaml')
# Load the dictionary which contains all necessary info for the system
ncams_config = ncams.camera_io.yaml_to_config(path_to_ncams_config)
# Create a board using the config info
charuco_dict, charuco_board, _ = ncams.camera_tools.create_board(ncams_config, plotting=False)
# Declare the path of the calibration images and get their paths
calibration_image_path = os.path.join(BASE_DIR, 'intrinsic','cam19194009')
cam_image_list = ncams.utils.get_image_list(calibration_image_path)
# Calibrate with those images
reprojection_error, camera_matrix, distortion_coefficients, detected_points = ncams.camera_calibration.charuco_calibration(
    cam_image_list, charuco_dict, charuco_board, export_marked_images=False, verbose=True)
# Evaluate calibration
ncams.camera_calibration.inspect_intrinsics_single(ncams_config, cam_image_list, camera_matrix,
                              distortion_coefficients, detected_points)
print('Reprojection error = ' + str(np.around(reprojection_error[0], 3)) + ' ' + ncams_config['world_units'])

