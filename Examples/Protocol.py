#!python36
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Script for running an experimental protocol.

Intended to be used in an interactive environment (e.g. Spyder).
Has following steps:
1-2. Sets up cameras, directories, configurations.
3-5. Calibration and relative pose estimations for cameras.
6-8. Records pictures from multiple cameras and stores them on the disk.
9. Transforms pictures into videos.

Is continued in Analysis.py

author(s): Charles M. Greenspon
           Anton Sobinov
lab: Sliman Bensmaia
"""

# %% 0 Imports
import os
import time
import datetime
import math

# Spyder starts wherever
os.chdir(os.path.join('C:/', 'Repositories', 'Stereognosis',
                      'AnimalTrackingToolbox'))

import CameraTools
import ImageTools
import SpinnakerTools


# %% 1 Init working directories
# Rerun if want refresh the data storage
BASE_DIR = os.path.join('C:/', 'FLIR_cameras')
cdatetime = time.strftime('%Y.%m.%d_%H.%M.%S', time.localtime())
working_dir = os.path.join(BASE_DIR, 'exp'+cdatetime)
# if you want to work with older data:
working_dir = 'C:/FLIR_cameras/exp2019.11.22_10.00.09'

for p in (BASE_DIR, working_dir):
    if not os.path.isdir(p):
        os.mkdir(p)

# %% 2 Initialize cameras
reference_cam_serial = 19194009
system, cam_list, cam_serials, cam_dicts = SpinnakerTools.get_system()
reference_cam_i = cam_serials.index(reference_cam_serial)

# Take a sample picture from each camera so we know which camera is which
SpinnakerTools.test_system_capture(cam_list)

# Then lets make a dictionary containing the relevant info for calibration
camera_config = dict(
    # whatever string is used for individual camera folders:
    camera_names=[cam_dicts[s]['name'] for s in cam_serials],
    # We want to keep track of these in case the order gets altered
    camera_serials=cam_serials,
    reference_camera=reference_cam_i,  # zero indexed from above list
    reference_camera_serial=reference_cam_serial,
    image_size=(1080, 1440),  # width x height
    board_type='charuco',  # charuco or checkerboard
    # If this is incorrect it will cause analyses to freeze:
    board_dim=[7, 10],
    # Size of the checks in mm, essential for accurate 3D reconstructions:
    check_size=30,
    folder_path=working_dir)

CameraTools.config_to_yaml(camera_config)  # Export it

# %% 3.0 Calibrate lenses on a camera
# (command for loading from file is further in a script)
# We need a calibration object.
# This function will create one that you can print
CameraTools.create_board(camera_config, output=True, output_format='jpg',
                         plotting=True)
# %% 3.1
# Now we'll take pictures now with each of the cameras
calib_dir = os.path.join(camera_config['folder_path'], 'calibration')
if not os.path.isdir(calib_dir):
    os.mkdir(calib_dir)
for icam, (cam_serial, cam_dict) in enumerate(cam_dicts.items()):
    # First lets initalize with some appropriats settings
    SpinnakerTools.set_cam_settings(cam_dict['obj'], default=True)
    SpinnakerTools.set_cam_settings(cam_dict['obj'], frame_rate=2)

    cam_calib_path = os.path.join(calib_dir, cam_dict['name'])
    if not os.path.isdir(cam_calib_path):
        os.mkdir(cam_calib_path)

    # pause to get ready for the next camera calibration
    valid_input = False
    print('Camera #{}/{} serial {} name {} is ready for lense '
          'calibration.'.format(icam+1, len(cam_dicts), cam_serial,
                                cam_dict['name']))
    while valid_input is False:
        user_input = input('Are you ready for the capture? (y/n)\n').lower()
        if user_input == 'y':
            valid_input = True

    SpinnakerTools.capture_sequence_GUI(cam_dict['obj'],
                                        num_images=50,
                                        output_path=cam_calib_path,
                                        file_prefix=cam_dict['name']+'_')
# %% 3.2
# Run the multi-calibration on all of them
(reprojection_errors, camera_matrices, distortion_coefficients
 ) = CameraTools.multi_camera_calibration(
     camera_config, inspect=True, calib_dir=calib_dir)

# %% 4.1 Relative pose estimation of cameras
# (command for loading from file is further in a script)
# Now lets try to do a one-shot pose estimation using the synced_capture
# function
pose_estimation_path = os.path.join(camera_config['folder_path'],
                                    'pose_estimation')
if not os.path.isdir(pose_estimation_path):
    os.mkdir(pose_estimation_path)

# Capture the images
SpinnakerTools.init_sync_settings_serials(
    cam_dicts, reference_cam_serial, frame_rate=30, num_images=None)

SpinnakerTools.synced_capture_sequence_serials(
    cam_dicts, reference_cam_serial, 1,
    output_folder=pose_estimation_path, separate_folders=False)

# %% 4.2
# Do the pose estimation
world_locations, world_orientations = CameraTools.one_shot_multi_PnP(
    camera_config, camera_matrices, distortion_coefficients)

# Does it look okay?
CameraTools.plot_poses(world_locations, world_orientations, scale_factor=1)

# If so lets export it
CameraTools.export_pose_estimation(
    pose_estimation_path, camera_config['camera_serials'],
    world_locations, world_orientations)


# %% 5 Load calibration and pose estimation data from a file if calibration and
# pose estimation has been done before and saved
# if the variables do not exist in the local space:
# if 'camera_matrices' not in locals() or 'world_locations' not in locals():
(camera_matrices, distortion_coefficients, reprojection_errors,
 world_locations, world_orientations) = CameraTools.load_camera_config(
     camera_config)

# Does it look okay?
CameraTools.plot_poses(world_locations, world_orientations, scale_factor=1)

# %% 6 Set up experiment
pre_touchpad_early_start = 1000  # msec before you press Start Protocol
# msec additional recording after protocol is finished:
trailing_waiting_period = 1000
# from ProtocolParameters.cpp:
total_trial_time = 3000  # msec maxWaitTime
intertrial_time = 100  # msec intertrialTime
number_trials_desired = 100  # nTrialsDesired, first ten for sync
#number_trials_desired = 4  # nTrialsDesired, first ten for sync
# from Protocol.cpp:
pretouch_wait = 1500  # msec PRETOUCH_WAIT

camera_frame_rate = 50

session_time_length_sec = (
    pre_touchpad_early_start +
    number_trials_desired*(total_trial_time+intertrial_time) +
    trailing_waiting_period) / 1e3
session_number_frames = math.ceil(session_time_length_sec * camera_frame_rate)
print('Going to capture {} frames over {} seconds'.format(
    session_number_frames, session_time_length_sec))

session_number = 2
session_user = 'AS'
session_subject = 'CMG'
session_path = os.path.join(working_dir, 'exp_session_{}_{}_{}'.format(
    session_user, session_subject, session_number))
if not os.path.isdir(session_path):
    os.mkdir(session_path)
session_info = dict(
    user=session_user,  # The person present during the recordings
    subject=session_subject,  # Subject of the recording
    date=datetime.date.today(),  # Date of recording
    session_number=1,  # If multiple sessions
    # In case the session config is saved separately from the data:
    session_path=session_path,
    # To set fr of cameras during acquisition
    camera_frame_rate=camera_frame_rate)

# %% 7 Run experiment
SpinnakerTools.reset_cams(cam_list)
for cam in cam_list:
    SpinnakerTools.set_cam_settings(cam, default=True)
print('Cameras set up done.')

SpinnakerTools.init_sync_settings_serials(
    cam_dicts, reference_cam_serial,
    frame_rate=session_info['camera_frame_rate'], num_images=None)
print('Cameras sync done.')

print('Starting capture')
SpinnakerTools.synced_capture_sequence_serials(
    cam_dicts, reference_cam_serial, session_number_frames,
    output_folder=session_info['session_path'], separate_folders=True)

print('Capture done')
SpinnakerTools.reset_cams(cam_list)

# %% 8 Release cameras
SpinnakerTools.release_system(system, cam_list)

# %% 9 Make images into videos
video_path = os.path.join(session_path, 'videos')
if not os.path.isdir(video_path):
    os.mkdir(video_path)
session_info['video_path'] = video_path

for cam_serial in cam_serials:
    cam_dicts[cam_serial]['pic_dir'] = os.path.join(session_info['session_path'],
                                                    cam_dicts[cam_serial]['name'])
    cam_dicts[cam_serial]['video'] = os.path.join(session_info['video_path'],
                                                  cam_dicts[cam_serial]['name']+'.mp4')

for cam_dict in cam_dicts.values():
    cam_dir = os.path.join(session_info['session_path'], cam_dict['name'])
    image_list = ImageTools.get_image_list(sort=True, path=cam_dict['video_dir'])
    ImageTools.images_to_video(image_list, cam_dict['name']+'.mp4',
                               fps=session_info['camera_frame_rate'],
                               output_folder=session_info['video_path'])

# %% 10 Undistort the videos
ud_video_path = os.path.join(session_path, 'undistorted_videos')
if not os.path.isdir(ud_video_path):
    os.mkdir(ud_video_path)
session_info['ud_video_path'] = ud_video_path

for icam, cam_serial in enumerate(cam_serials):
    cam_dicts[cam_serial]['ud_video'] = os.path.join(
        session_info['ud_video_path'], cam_dicts[cam_serial]['name']+'.mp4')
    cam_dict = cam_dicts[cam_serial]

    ImageTools.undistort_video(cam_dict['video'], camera_matrices[icam],
                               distortion_coefficients[icam], crop_and_resize=False,
                               output_path=session_info['ud_video_path'])
    print('Camera {} video undistorted.'.format(
        cam_dict['name']))
