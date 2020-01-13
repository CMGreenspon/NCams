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
1. Load configurations
2. Either 'a' or 'b':
    a. Make a new DLC project (this or 2b)
    b. Load existing labeled frames
3. Triangulation from multiple cameras
4. Make markered videos
5. Interactive demonstration with a time slider

For more details on the camera data structures and dicts, see help(ncams.camera_tools).
"""

# %% 0 Imports
import os
import shutil
from glob import glob

import deeplabcut

import ncams


BASE_DIR = os.path.join('C:\\', 'FLIR_cameras', 'PublicExample')
os.environ['DLC_PER_PROCESS_GPU_MEMORY_FRACTION'] = '0.8'


# %% 1 Load configurations
cdatetime = '2019.12.19_10.38.38'  # calibrations and camera settings
camera_config_dir = os.path.join(BASE_DIR, 'camconf_'+cdatetime)
camera_config = ncams.yaml_to_config(os.path.join(camera_config_dir, 'config.yaml'))

calibration_config, pose_estimation_config = ncams.load_camera_config(camera_config)

#  Load a session config from a file
# select which sessions you want to train on
cdate = '2019.12.20'
session_ids = (4, 5)
sessions = [os.path.join(BASE_DIR, 'exp_session_{}_*_AS_CMG_{}'.format(cdate, i)) for i in session_ids]
training_videos = []
videos_collection_dir = os.path.join(BASE_DIR, 'exp_session_'+cdate+'_videos')
if not os.path.isdir(videos_collection_dir):
    os.mkdir(videos_collection_dir)
for session_id, session in zip(session_ids, sessions):
    session_dirname = glob(session)[0]
    session_filename = os.path.join(session_dirname, 'session_config.yaml')
    session_config = ncams.import_session_config(session_filename)
    for cs in camera_config['serials']:
        tv = os.path.join(session_config['session_path'],
                          session_config['cam_dicts'][cs]['ud_video'])
        tv2 = os.path.join(videos_collection_dir, '{}_{}'.format(session_id, os.path.split(tv)[-1]))
        #shutil.copyfile(tv, tv2)
        training_videos.append(tv2)


# %% 2a Make a new DLC project (this or 2b)
dlc_prj_name = '2019.12.20_8camsNoMarkers'
scorer = 'AS'
prj_date = '2019-12-23'
config_path = deeplabcut.create_new_project(
    dlc_prj_name, scorer, training_videos,
    working_directory=BASE_DIR, copy_videos=False)
dlc_proj_name = '-'.join([dlc_prj_name, scorer, prj_date])

proj_path = os.path.join(BASE_DIR, dlc_proj_name)
if config_path is None:
    config_path = os.path.join(proj_path, 'config.yaml')
labeled_csv_path = os.path.join(proj_path, 'labeled_videos')
if not os.path.isdir(labeled_csv_path):
    os.mkdir(labeled_csv_path)

print('New config_path: "{}"'.format(config_path))

# Edit the config file to represent your tracking

# DLC Cropping
deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', crop=False,
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
dlc_prj_name = '2019.12.20_8camsNoMarkers'
scorer = 'AS'
prj_date = '2019-12-23'
dlc_proj_name = '-'.join([dlc_prj_name, scorer, prj_date])
proj_path = os.path.join(BASE_DIR, dlc_proj_name)
config_path = os.path.join(proj_path, 'config.yaml')

print('Existing config_path: "{}"'.format(config_path))

labeled_csv_path = os.path.join(proj_path, 'labeled_videos')
if not os.path.isdir(labeled_csv_path):
    os.mkdir(labeled_csv_path)

analyzed_training_videos = []
for serial in camera_config['serials']:
    analyzed_training_videos.append(os.path.join(
        proj_path, 'labeled_videos',
        'cam{}DLC_resnet50_CMGPretrainedNetworkDec3shuffle1_250000_labeled.mp4'.format(serial)))
analyzed_training_videos_dir = [os.path.join(proj_path, 'labeled_videos')]

# %% Refinement?
deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', crop=False,
                          userfeedback=False)

deeplabcut.label_frames(config_path)

deeplabcut.merge_datasets(config_path)

deeplabcut.create_training_dataset(config_path)

deeplabcut.train_network(config_path, gputouse=0, saveiters=25000, maxiters=250000)

# %% 3 Triangulation from multiple cameras
method = 'full_rank'
threshold = 0.9
triangulated_path = os.path.join(proj_path, 'triangulated_{}_{}'.format(method, threshold))
if not os.path.exists(triangulated_path):
    os.mkdir(triangulated_path)
triangulated_csv = os.path.join(triangulated_path, 'triangulated_points.csv')

ncams.triangulate(
    camera_config, triangulated_csv, calibration_config, pose_estimation_config, labeled_csv_path,
    threshold=threshold, method=method, undistorted_data=True)

# filter the triangulated points in 3D space
triangulated_csv_p = os.path.join(triangulated_path, 'triangulated_points_smoothed.csv')
ncams.process_triangulated_data(triangulated_csv, output_csv=triangulated_csv_p)


# %% 4 Make markered videos
serial = 19335177
video_path = camera_config['dicts'][serial]['ud_video']
ncams.make_triangulation_video(video_path, triangulated_csv_p, skeleton_config=config_path)

# %% 5 Interactive demonstration with a slider
# This sometimes breaks in Spyder, try running as an executable, commenting out parts of
# 'analysis.py' that are not needed.
ncams.reconstruction.interactive_3d_plot(video_path, triangulated_csv_p, skeleton_path=config_path)
