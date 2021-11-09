import ncams
import os
import time
import numpy as np
import cv2

BASE_DIR = r'\\bensmaia-lab\LabSharing\Stereognosis\DeepLabCut\CameraConfigs'
session_dir = r'21.10.15_calibration'
ncams_config_path = os.path.join(BASE_DIR, session_dir, 'ncams_config.yaml')
ncams_config = ncams.camera_io.yaml_to_config(ncams_config_path)

#%% Intrinsic calibration
intrinsics_config = ncams.camera_calibration.multi_camera_intrinsic_calibration(
    ncams_config, override=False, inspect=True, export_full=True, verbose=True)

#%% Extrinsic calibration (One Shot)
extrinsics_config, extrinsics_info = ncams.camera_pose.one_shot_multi_PnP(
    ncams_config, intrinsics_config, export_full=True, show_extrinsics=True,
    inspect=True)

