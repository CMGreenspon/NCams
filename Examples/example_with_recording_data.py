#!python36
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Script for running an experimental protocol.

Intended to be used in an interactive environment (e.g. Spyder).
Has following steps:
0. Import modules
1-2. Sets up cameras, directories, configurations.
3. Calibration of camera lenses.
4. Estimation of relative position of cameras.
5. Instead of Steps 1-4 load the setup, calibration and pose estimation.
6. Setup an experimental recording session.
7. Load a config for the session from a file (instead of setting up).
8. Records pictures from multiple cameras and stores them on the disk.
9. Release cameras.
10. Transform pictures into videos.
11. Undistort videos using camera calibration.

Is continued in analysis.py

For more details on the camera data structures and dicts, see help(ncams.camera_tools).
"""

# %% 0 Imports
import os
import time
import math

import ncams
import ncams.spinnaker_tools


BASE_DIR = os.path.join('C:\\', 'FLIR_cameras', 'PublicExample')

# %% 1 Initialize setup and working directories (if intialized before, go to Step 5)
# If you wish to work with example data, proceed to Step 5
# Rerun if want refresh the data storage
cdatetime = time.strftime('%Y.%m.%d_%H.%M.%S', time.localtime())
camera_config_dir = os.path.join(BASE_DIR, 'camconf_'+cdatetime)


# %% 2 Initialize cameras
system, cam_serials, cam_dicts = ncams.spinnaker_tools.get_system()

# Then lets make a dictionary containing the relevant info for cameras
# (see help(ncams.camera_tools))
camera_config = {
    'datetime': cdatetime,
    'serials': cam_serials,  # We want to keep track of these in case the order gets altered
    # keys are serials, values are dictionaries with valuable information on each camera:
    'dicts': cam_dicts,
    'reference_camera_serial': 19194009,  # is set up within hardware
    'system': system,

    'image_size': (1080, 1440),  # width x height
    'board_type': 'charuco',  # charuco or checkerboard
    'board_dim': [7, 10],  # If this is incorrect it will cause analyses to freeze:
    'check_size': 30,  # Size of the checks in mm, essential for accurate 3D reconstructions

    'setup_path': camera_config_dir,
    'setup_filename': 'config.yaml',
    'calibration_path': 'calibration',
    'calibration_filename': 'camera_calib.pickle',
    'pose_estimation_path': 'pose_estimation',
    'pose_estimation_filename': 'pose_estimate.pickle',
}

# make sure that all required directories exist
for p in (BASE_DIR, camera_config['setup_path'],
          os.path.join(camera_config['setup_path'], camera_config['calibration_path']),
          os.path.join(camera_config['setup_path'], camera_config['pose_estimation_path'])):
    if not os.path.isdir(p):
        print('Making dir {}'.format(p))
        os.mkdir(p)

# Take a sample picture from each camera so we know which camera is which
ncams.spinnaker_tools.test_system_capture(camera_config)

# Export config to disk
ncams.config_to_yaml(camera_config)


# %% 3.0 Calibrate lenses on a camera
# (command for loading from file is in Step 5)

# We need a calibration object.
# This function will create one that you can print
ncams.camera_tools.create_board(camera_config, output=True, output_format='jpg', plotting=True)


# %% 3.1
# Now we'll take pictures now with each of the cameras. While the pictures are taken the
# experimenter is supposed to move the board within the field of view of the camera so that the
# whole field of view is covered. It is important to pay attention to corners.
for icam, serial in enumerate(camera_config['serials']):
    cam_dict = camera_config['dicts'][serial]
    # First lets initalize with some appropriats settings
    ncams.spinnaker_tools.set_cam_settings(cam_dict['obj'], default=True)
    ncams.spinnaker_tools.set_cam_settings(cam_dict['obj'], frame_rate=2)

    cam_calib_path = os.path.join(camera_config['setup_path'], camera_config['calibration_path'],
                                  cam_dict['name'])
    if not os.path.isdir(cam_calib_path):
        os.mkdir(cam_calib_path)

    # pause to get ready for the camera calibration
    print('Camera #{}/{} serial {} name {} is ready for lense calibration.'.format(
        icam+1, len(camera_config['serials']), serial, cam_dict['name']))
    while not input('Are you ready for the capture? (y)\n').lower() == 'y':
        pass

    ncams.spinnaker_tools.capture_sequence_gui(
        cam_dict['obj'], num_images=50, output_path=cam_calib_path,
        file_prefix=cam_dict['name']+'_')


# %% 3.2
# Run the multi-calibration on all of them
calibration_config = ncams.multi_camera_calibration(camera_config, inspect=True)

# export to disk
ncams.export_calibration(calibration_config)


# %% 4.1 Relative pose estimation of cameras
# (command for loading from file is in Step 5)
# Now we will perform a one-shot pose estimation using the synced_capture function
# Capture the images
ncams.spinnaker_tools.init_sync_settings(camera_config)

ncams.spinnaker_tools.synced_capture_sequence(
    camera_config, 1,
    output_folder=os.path.join(camera_config['setup_path'], camera_config['pose_estimation_path']),
    separate_folders=False)


# %% 4.2
# Do the pose estimation
pose_estimation_config = ncams.camera_pose.one_shot_multi_PnP(
    camera_config, calibration_config)

# Does it look okay?
ncams.camera_pose.plot_poses(pose_estimation_config)

# If so lets export it
ncams.export_pose_estimation(pose_estimation_config)


# %% 5 Load camera_config, calibration and pose estimation data from files
# Works if calibration and pose estimation has been done before and saved
cdatetime = '2019.12.19_10.38.38'
camera_config_dir = os.path.join(BASE_DIR, 'camconf_'+cdatetime)
camera_config = ncams.yaml_to_config(os.path.join(camera_config_dir, 'config.yaml'))

calibration_config, pose_estimation_config = ncams.load_camera_config(camera_config)

# Does it look okay?
ncams.camera_pose.plot_poses(pose_estimation_config)


# %% 6 Set up experiment (you can load the setup in Step 7)
session_time_length_sec = 30
frame_rate = 50

session_number_frames = math.ceil(session_time_length_sec * frame_rate)
print('Going to capture {} frames over {} seconds'.format(
    session_number_frames, session_time_length_sec))

session_number = 1  # if multiple sessions
session_user = 'AS'  # The person conducting the recordings
session_subject = 'CMG'  # Subject of the recording
session_datetime = time.strftime('%Y.%m.%d_%H.%M.%S', time.localtime())
session_path = os.path.join(BASE_DIR, 'exp_session_{}_{}_{}_{}'.format(
    session_datetime, session_user, session_subject, session_number))
if not os.path.isdir(session_path):
    print('Making session directory: {}'.format(session_path))
    os.mkdir(session_path)

session_cam_dicts = {}
for serial in camera_config['serials']:
    session_cam_dicts[serial] = {
        'serial': serial,
        'name': camera_config['dicts'][serial]['name']
    }

session_config = {
    'user': session_user,
    'subject': session_subject,
    'datetime': session_datetime,
    'session_number': session_number,
    'session_path': session_path,  # where the config and the data get stored
    'session_filename': 'session_config.yaml',
    'frame_rate': frame_rate,  # to set fr of cameras during acquisition
    'number_frames': session_number_frames,  # number of frames captured from each camera during
    'camera_config_path': camera_config['setup_path'],  # to know how cameras were configured
    'camera_config_filename': camera_config['setup_filename'],
    'cam_dicts': session_cam_dicts
}

# save the config to disk
ncams.export_session_config(session_config)


# %% 7 Load a session config from a file
session_full_filename = os.path.join(BASE_DIR, 'exp_session_2019.12.09_16.40.45_AS_CMG_2',
                                     'session_config.yaml')
session_config = ncams.import_session_config(session_full_filename)

# %% 8 Run experiment
# useful for some low-level functions:
cam_list = [camera_config['dicts'][serial]['obj'] for serial in camera_config['serials']]

ncams.spinnaker_tools.reset_cams(cam_list)
for cam in cam_list:
    ncams.spinnaker_tools.set_cam_settings(cam, default=True)
print('Cameras were reset.')

ncams.spinnaker_tools.init_sync_settings(camera_config,
                                         frame_rate=session_config['frame_rate'], num_images=None)
print('Cameras sync and init done.')

print('Starting capture')
ncams.spinnaker_tools.synced_capture_sequence(
    camera_config, session_config['number_frames'],
    output_folder=session_config['session_path'], separate_folders=True)

print('Capture done')
ncams.spinnaker_tools.reset_cams(cam_list)


# %% 9 Release cameras
ncams.spinnaker_tools.release_system(system, camera_config['dicts'])


# %% 10 Make images into videos
session_config['video_path'] = 'videos'
session_config['ud_video_path'] = 'undistorted_videos'

for p in (os.path.join(session_config['session_path'], session_config['video_path']),
          os.path.join(session_config['session_path'], session_config['ud_video_path'])):
    if not os.path.isdir(p):
        print('Making dir {}'.format(p))
        os.mkdir(p)

for serial in camera_config['serials']:
    session_config['cam_dicts'][serial]['pic_dir'] = session_config['cam_dicts'][serial]['name']
    session_config['cam_dicts'][serial]['video'] = os.path.join(
        session_config['video_path'], session_config['cam_dicts'][serial]['name']+'.mp4')
    session_config['cam_dicts'][serial]['ud_video'] = os.path.join(
        session_config['ud_video_path'], session_config['cam_dicts'][serial]['name']+'.mp4')

for cam_dict in session_config['cam_dicts'].values():
    image_list = ncams.utils.get_image_list(
        sort=True, path=os.path.join(session_config['session_path'], cam_dict['pic_dir']))
    print('Making a video for camera {} from {} images.'.format(
        cam_dict['name'], len(image_list)))
    ncams.images_to_video(
        image_list, cam_dict['video'], fps=session_config['frame_rate'],
        output_folder=session_config['session_path'])

ncams.export_session_config(session_config)

# %% 11 Undistort the videos
for icam, serial in enumerate(camera_config['serials']):
    cam_dict = session_config['cam_dicts'][serial]
    ncams.undistort_video(
        cam_dict['video'], calibration_config['dicts'][serial],
        crop_and_resize=False,
        output_filename=os.path.join(session_config['session_path'], cam_dict['ud_video']))
    print('Camera {} video undistorted.'.format(cam_dict['name']))
