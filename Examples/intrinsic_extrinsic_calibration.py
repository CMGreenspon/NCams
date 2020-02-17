# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:57:53 2020

@author: somlab
"""


"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Example script for setting up a system. Goes through file formatting and both intrinsic and
extrinsic calibration. This also includes measuring how accurate the calibration steps are.
The steps taken are as follows:
    1. Create a ncams_config dictionary
    2. Creation of a charucoboard to use for calibration
    3. Camera intrinsic calibration and inspection
    4. Camera pose estimation and inspection
    
For more details on the camera data structures and dicts, see help(ncams.camera_tools).
"""

import ncams
import os
import time

# Replace this with your working directory
#BASE_DIR = os.path.join('C:\\', 'GitHub', 'NCams', 'PublicExample') 
BASE_DIR = r'C:\Users\somlab\Desktop\FLIRTesting'

#%% 1. Create the ncams_config
ncams_config = {
    # Camera information
    'serials': [19194005,19194008,19194009,19194013,19335177,19340298,19340300,19340396],
    'reference_camera_serial': 19194009,  # This is linked to how the cameras are hardwired
    'image_size': (1080, 1440),  # height x width 
    # Board information
    'board_type': 'charuco',  # charuco (preferred) or checkerboard
    'board_dim': [7, 10],  # If this is incorrect it will cause analyses to freeze
    'check_size': 30, # Size of the checks in mm, essential for accurate 3D reconstructions
    'world_units': 'mm', # Determines how to scale the world ('m', 'dm', 'cm', 'mm')
    # Path information
    'setup_path': BASE_DIR, # Where to store this configuration
    'setup_filename': 'ncams_config.yaml', # Desired filename for the configuration
    'intrinsic_path': 'calibration', # Name of the subdirectory for the intrinsic calibration data
    'intrinsic_filename': 'intrinsic_calib.pickle', # # Desired filename for the intrinsics
    'extrinsic_path': 'pose_estimation', # Name of the subdirectory for the extrinsic calibration data
    'extrinsic_filename': 'pose_estimate.pickle',
}

ncams.camera_io.config_to_yaml(ncams_config)

#%% 2. Create a board for calibration purposes
ncams.camera_tools.create_board(ncams_config, output=True, output_format='jpg', plotting=True)
''' The charucoboard may be printed on a piece of paper though the scaling for this is not 
guaranteed to be accurate. If this is desired then set the output format to pdf. Otherwise,
we recommend printing the image on metal with a matte coating to prevent glare. We do not
specifically endorse any company but have used bay photo (https://www.bayphoto.com/) with great
success. We also recommend testing different sized checks depending on the setup. With our setup
and the distance between the board and the cameras 40 mm checks were optimal.'''

#%% 3. Intrinsic calibration
'''
The below text describes how the folder structure should be formatted so that NCams knows where to 
look for the relevant images. The exact subfolder names are defined in the ncams_config above. These
can be changed but must still be in the same directory.

Setup path
    intrinsic_path
        cam1_folder
            synced_image1
            synced_image2
        cam2_folder
        camN_folder
'''

intrinsics_config = ncams.camera_calibration.multi_camera_intrinsic_calibration(
    ncams_config, override=True, inspect=False, export_full=True, verbose=True)
ncams.camera_calibration.inspect_intrinsics(ncams_config, intrinsics_config)
#%% 4. Extrinsic calibration - one_shot_multi_PnP

'''

Setup path
    extrinsic_path
        cam1_image
        cam2_image
        camN_image
'''

extrinsics_config, extrinsics_info = ncams.camera_pose.one_shot_multi_PnP(
    ncams_config, intrinsics_config, export_full=True, show_poses=False, inspect=False,
    ch_ids_to_ignore=None)
