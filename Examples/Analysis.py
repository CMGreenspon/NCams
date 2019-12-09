#!python36
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Script for running an analysis of the recordings from multiple cameras.

Intended to be used in an interactive environment (e.g. Spyder).
Has following steps:
0. Import modules

For more details on the camera data structures and dicts, see help(ncams.camera_t).
"""

# %% 0 Imports
import os

import deeplabcut

import ncams.camera_io
import ncams.reconstruction_t


BASE_DIR = os.path.join('C:/', 'FLIR_cameras', 'PublicExample')

# %% 1 Load configurations
cdatetime = '2019.12.09_16.23.02'
camera_config_dir = os.path.join(BASE_DIR, 'camconf_'+cdatetime)
camera_config = ncams.camera_io.yaml_to_config(os.path.join(camera_config_dir, 'config.yaml'))

calibration_config, pose_estimation_config = ncams.camera_io.load_camera_config(camera_config)

#  Load a session config from a file
session_full_filename = os.path.join(BASE_DIR, 'exp_session_2019.12.09_16.40.45_AS_CMG_2',
                                     'session_config.yaml')
session_config = ncams.utils.import_session_config(session_full_filename)

# which videos do you want to train on?
training_videos = [session_config['cam_dicts'][cs]['video'] for cs in camera_config['serials']]


# %% 2a Make a new DLC project (this or 2b)
dlc_prj_name = 'CMGPretrainedNetwork'
scorer = 'CMG'
prj_date = '2019-12-03'
config_path = deeplabcut.create_new_project(
    dlc_prj_name, scorer, training_videos,
    working_directory=session_config['session_path'], copy_videos=False)
dlc_proj_name = '-'.join([dlc_prj_name, scorer, prj_date])

proj_path = os.path.join(BASE_DIR, dlc_proj_name)
labeled_csv_path = os.path.join(proj_path, 'labeled_videos')
if not os.path.isdir(labeled_csv_path):
    os.mkdir(labeled_csv_path)

print('New config_path: "{}"'.format(config_path))

# Edit the config file to represent your tracking

# DLC Cropping
deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', crop=True,
                          userfeedback=False)

deeplabcut.label_frames(config_path)

deeplabcut.check_labels(config_path)

deeplabcut.create_training_dataset(config_path)

# 0 is the GPU number, see in nvidia-smi
deeplabcut.train_network(config_path, gputouse=0, saveiters=25000, maxiters=250000)

deeplabcut.evaluate_network(config_path, plotting=False)

# DLC estimation
deeplabcut.analyze_videos(config_path, training_videos,
                          gputouse=0, save_as_csv=True, destfolder=labeled_csv_path)

deeplabcut.create_labeled_video(config_path, training_videos, destfolder=labeled_csv_path,
                                draw_skeleton=True)

# %% 2b Load existing labeled frames
dlc_prj_name = 'CMGPretrainedNetwork'
scorer = 'CMG'
prj_date = '2019-12-03'
dlc_proj_name = '-'.join([dlc_prj_name, scorer, prj_date])
proj_path = os.path.join(BASE_DIR, dlc_proj_name)
config_path = os.path.join(proj_path, 'config.yaml')

print('Existing config_path: "{}"'.format(config_path))

labeled_csv_path = os.path.join(proj_path, 'labeled_videos')
if not os.path.isdir(labeled_csv_path):
    os.mkdir(labeled_csv_path)

# %% 3 Triangulation from multiple cameras
triangulated_path = os.path.join(proj_path, 'triangulated')
if not os.path.exists(triangulated_path):
    os.mkdir(triangulated_path)

method = 'full_rank'
triangulated_csv = os.path.join(triangulated_path, 'triangulated_points_'+method+'.csv')
threshold = 0.9
ncams.reconstruction_t.triangulate(
    camera_config, session_config, calibration_config, pose_estimation_config, labeled_csv_path,
    threshold=threshold, method=method, output_csv=triangulated_csv)

# %% 4 Make markered videos
# In big videos it takes awhile, try running with 'parallel' keyword outside of interactive Python.
ncams.reconstruction_t.make_triangulation_videos(
    camera_config, session_config, calibration_config, pose_estimation_config, triangulated_csv,
    triangulated_path=triangulated_path, overwrite_temp=True)

# %% 5 Interactive demonstration with a slider
# This sometimes breaks in Spyder, try running as an executable, commenting out parts of
# 'analysis.py' that are not needed.
ncams.reconstruction_t.interactive_3d_plot(
    camera_config['serials'][0], camera_config, session_config, triangulated_csv,
    num_frames_limit=None)
