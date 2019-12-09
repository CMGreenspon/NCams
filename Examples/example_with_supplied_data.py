#!python36
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Script for running functions on supplied example data. Useful for changing into own data structures.
Basically an abridged version of 'Examples/example_with_recording_data.py' that does not require
FLIR cameras, Spinnaker, etc.

The images are available here:
??????
and BASE_DIR has to point at that data.

For more details on the camera data structures and dicts, see help(ncams.camera_t).

Intended to be used in an interactive environment (e.g. Spyder).
Has following steps:
0. Import modules
1. Load camera configuration
2. Calibration of camera lenses.
3. Estimation of relative position of cameras.
4. Instead of Steps 1-3 you can load the setup, calibration and pose estimation from files.
5. Load a config for the session from a file (instead of setting up).
6. Transform pictures into videos.
7. Undistort videos using camera calibration.

Is continued in analysis.py

author(s): Charles M. Greenspon
           Anton Sobinov
lab: Sliman Bensmaia
"""

# %% 0 Imports
import os

import ncams


BASE_DIR = os.path.join('C:/', 'FLIR_cameras', 'PublicExample')

# %% 1 Load camera_config
# Works if calibration and pose estimation has been done before and saved
cdatetime = '2019.12.09_16.23.02'
camera_config_dir = os.path.join(BASE_DIR, 'camconf_'+cdatetime)
camera_config = ncams.camera_io.yaml_to_config(os.path.join(camera_config_dir, 'config.yaml'))


# %% 2 Run the multi-calibration on all cameras
calibration_config = ncams.camera_calibration.multi_camera_calibration(camera_config, inspect=True)

# export to disk
ncams.camera_io.export_calibration(calibration_config)


# %% 3 Do the pose estimation
pose_estimation_config = ncams.camera_positions.one_shot_multi_PnP(
    camera_config, calibration_config)

# Does it look okay?
ncams.camera_positions.plot_poses(pose_estimation_config)

# If so lets export it
ncams.camera_io.export_pose_estimation(pose_estimation_config)


# %% 4 Load camera_config, calibration and pose estimation data from files
# Works if calibration and pose estimation has been done before and saved
cdatetime = '2019.12.09_16.23.02'
camera_config_dir = os.path.join(BASE_DIR, 'camconf_'+cdatetime)
camera_config = ncams.camera_io.yaml_to_config(os.path.join(camera_config_dir, 'config.yaml'))

calibration_config, pose_estimation_config = ncams.camera_io.load_camera_config(camera_config)

# Does it look okay?
ncams.camera_positions.plot_poses(pose_estimation_config)


# %% 5 Load a session config from a file
session_full_filename = os.path.join(BASE_DIR, 'exp_session_2019.12.09_16.40.45_AS_CMG_2',
                                     'session_config.yaml')
session_config = ncams.utils.import_session_config(session_full_filename)


# %% 6 Make images into videos
video_path = os.path.join(session_config['session_path'], 'videos')
ud_video_path = os.path.join(session_config['session_path'], 'undistorted_videos')
session_config['video_path'] = video_path
session_config['ud_video_path'] = ud_video_path

for p in (session_config['video_path'], session_config['ud_video_path']):
    if not os.path.isdir(p):
        print('Making dir {}'.format(p))
        os.mkdir(p)

for serial in camera_config['serials']:
    session_config['cam_dicts'][serial]['pic_dir'] = os.path.join(
        session_config['session_path'], session_config['cam_dicts'][serial]['name'])
    session_config['cam_dicts'][serial]['video'] = os.path.join(
        session_config['video_path'], session_config['cam_dicts'][serial]['name']+'.mp4')
    session_config['cam_dicts'][serial]['ud_video'] = os.path.join(
        session_config['ud_video_path'], session_config['cam_dicts'][serial]['name']+'.mp4')

for cam_dict in session_config['cam_dicts'].values():
    image_list = ncams.utils.get_image_list(sort=True, path=cam_dict['pic_dir'])
    print('Making a video for camera {} from {} images.'.format(
        cam_dict['name'], len(image_list)))
    ncams.image_t.images_to_video(image_list, cam_dict['video'],
                                  fps=session_config['frame_rate'],
                                  output_folder=session_config['video_path'])

ncams.utils.export_session_config(session_config)


# %% 7 Undistort the videos
for icam, serial in enumerate(camera_config['serials']):
    cam_dict = session_config['cam_dicts'][serial]
    ncams.image_t.undistort_video(cam_dict['video'],
                                  calibration_config['dicts'][serial],
                                  crop_and_resize=False,
                                  output_filename=cam_dict['ud_video'])
    print('Camera {} video undistorted.'.format(cam_dict['name']))
