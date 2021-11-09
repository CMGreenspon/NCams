import os, ncams
import numpy as np

BASE_DIR = r'\\bensmaia-lab\LabSharing\Stereognosis\DeepLabCut\CameraConfigs'
session_dir = '21.10.28_calibration'
ncams_config_path = os.path.join(BASE_DIR, session_dir, 'ncams_config.yaml')
ncams_config = ncams.camera_io.yaml_to_config(ncams_config_path)
charuco_dict, charuco_board, _ = ncams.camera_tools.create_board(ncams_config, plotting=False)

#%% Individual camera calibration
serial = 19340396

cam_dir = os.path.join(BASE_DIR, session_dir, 'intrinsic', str(serial))

cam_image_list = ncams.utils.get_image_list(cam_dir)
reprojection_error, camera_matrix, distortion_coefficients, detected_points = ncams.camera_calibration.charuco_calibration(
    cam_image_list, charuco_dict, charuco_board, export_marked_images=False, verbose=True)

ncams.camera_calibration.inspect_intrinsics_single(ncams_config, cam_image_list, camera_matrix,
                              distortion_coefficients, detected_points)
print('Reprojection error = ' + str(np.around(reprojection_error[0], 3)) + ' ' + ncams_config['world_units'])

# Export if calibration is good
if reprojection_error[0] < 1:
    cam_calib_filename = os.path.join(cam_dir, 'cam' + str(serial) + '_calib.yaml')
    
    camera_calib_dict = {
         'serial': serial,
         'distortion_coefficients': distortion_coefficients,
         'camera_matrix': camera_matrix,
         'reprojection_error': reprojection_error,
         'calibration_images': cam_image_list,
         'detected_markers': detected_points,
     }
    ncams.camera_io.intrinsic_to_yaml(cam_calib_filename, camera_calib_dict)

