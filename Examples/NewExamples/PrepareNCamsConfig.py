# NCams Config Script

import ncams
import os
import time
import numpy as np
import cv2

BASE_DIR = r'\\BENSMAIA-LAB\LabSharing\Stereognosis\DeepLabCut\CameraConfigs'
session_dir = r'21.10.15_calibration'
config_dir = os.path.join(BASE_DIR, session_dir)

#%% Create the ncams_config
ncams_config = {
    # Camera information
    'serials': [19340300, 19340298, 19194008, 19335177, 19340396, 20050811, 19194005, 19194009], #
    'reference_camera_serial': 19335177,  # This is linked to how the cameras are hardwired
    'image_size': (1080, 1440),  # height x width 
    # Board information
    'board_type': 'charuco',  # charuco (preferred) or checkerboard
    'board_dim': [6, 8],  # If this is incorrect it will cause analyses to freeze
    'check_size': 40, # Size of the checks in mm, essential for accurate 3D reconstructions
    'world_units': 'mm', # Determines how to scale the world ('m', 'dm', 'cm', 'mm')
    # Path information
    'setup_path': config_dir, # Where to store this configuration
    'setup_filename': 'ncams_config.yaml', # Desired filename for the configuration
    'intrinsic_path': 'intrinsic', # Name of the subdirectory for the intrinsic calibration data
    'intrinsic_filename': 'intrinsic_calib.pickle', # Desired filename for the intrinsics
    'extrinsic_path': 'extrinsic', # Name of the subdirectory for the extrinsic calibration data
    'extrinsic_filename': 'extrinsic_calib.pickle'}

#%% Prepare folders
if os.path.exists(config_dir) is False:
    os.mkdir(config_dir)

ncams.camera_io.config_to_yaml(ncams_config)

if os.path.exists(os.path.join(config_dir, ncams_config['extrinsic_path'])) is False:
    os.mkdir(os.path.join(config_dir, ncams_config['extrinsic_path']))
    
if os.path.exists(os.path.join(config_dir, ncams_config['intrinsic_path'])) is False:
    os.mkdir(os.path.join(config_dir, ncams_config['intrinsic_path']))
    
    for s in ncams_config['serials']:
        os.mkdir(os.path.join(config_dir, ncams_config['intrinsic_path'], str(s)))
        
