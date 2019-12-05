#!python36
# -*- coding: utf-8 -*-
"""
Script for running a protocol. Not intended to be run in its entirety.

author(s): Charles M. Greenspon
           Anton Sobinov
lab: Sliman Bensmaia
"""

# %% 0 Imports
import os

import deeplabcut

# Spyder starts wherever
os.chdir(os.path.join('C:/', 'Repositories', 'Stereognosis',
                      'AnimalTrackingToolbox'))

import CameraTools
import ImageTools
import SpinnakerTools
import ReconstructionTools


# %% 1 Load configurations
working_dir = 'C:/FLIR_cameras/exp2019.11.22_10.00.09'
session_path = os.path.join(working_dir, 'exp_session_AS_CMG_2')
video_path = os.path.join(session_path, 'videos')
ud_video_path = os.path.join(session_path, 'undistorted_videos')
camera_config = CameraTools.yaml_to_config(os.path.join(working_dir, '2019-11-22_config.yaml'))
cam_dicts = {}
for cam_serial, cam_name in zip(camera_config['camera_serials'],
                                camera_config['camera_names']):
    cam_dicts[cam_serial] = {}
    cam_dicts[cam_serial]['name'] = cam_name
    cam_dicts[cam_serial]['video'] = os.path.join(video_path, cam_name+'.mp4')
    cam_dicts[cam_serial]['ud_video'] = os.path.join(ud_video_path, cam_name+'_undistorted.mp4')
cam_serials = sorted(camera_config['camera_serials'])

(camera_matrices, distortion_coefficients, reprojection_errors,
 world_locations, world_orientations) = CameraTools.load_camera_config(
     camera_config)

training_videos = [cam_dicts[cs]['video'] for cs in cam_serials]

# %% 2 Make DLC project
dlc_prj_name = 'CMGPretrainedNetwork'
scorer = 'CMG'
prj_date = '2019-12-03'
config_path = deeplabcut.create_new_project(
    dlc_prj_name, scorer, training_videos,
    working_directory=session_path, copy_videos=False)
dlc_proj_name = '-'.join([dlc_prj_name, scorer, prj_date])

proj_path = os.path.join(session_path, dlc_proj_name)
labeled_video_path = os.path.join(proj_path, 'labeled_videos')
if not os.path.isdir(labeled_video_path):
    os.mkdir(labeled_video_path)

# if project exists, return to it - could not find an "open project" function
if config_path is None:
    config_path = os.path.join(proj_path, 'config.yaml')
else:
    print('New config_path: "{}"'.format(config_path))

# %% 3 Edit the config file to represent your tracking
# or copy the provided one in the repository

# %% DLC Cropping
deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', crop=True,
                          userfeedback=False)

# %%
deeplabcut.label_frames(config_path)

# %%
deeplabcut.check_labels(config_path)

# %%
deeplabcut.create_training_dataset(config_path)

# %%
# 0 is the GPU number, see in nvidia-smi
deeplabcut.train_network(config_path, gputouse=0, saveiters=25000, maxiters=250000)

# %%
deeplabcut.evaluate_network(config_path, plotting=False)

# %% DLC estimation
deeplabcut.analyze_videos(config_path, training_videos,
                          gputouse=0, save_as_csv=True, destfolder=labeled_video_path)

# %%
deeplabcut.create_labeled_video(config_path, training_videos, destfolder=labeled_video_path,
                                draw_skeleton=True)


# %% Triangulation from multiple cameras
images_3d_path = os.path.join(proj_path, 'rec_3d')
method = 'best_pair'
triangulated_csv = os.path.join(images_3d_path, 'triangulated_points_'+method+'.csv')
threshold = 0.9
ReconstructionTools.triangulate(
    cam_dicts, camera_config, session_path, labeled_video_path,
    threshold=threshold, images_3d_path=images_3d_path,
    method=method, output_csv=triangulated_csv)

# %%
ReconstructionTools.make_triangulation_videos(
    camera_config, cam_dicts, session_path, triangulated_csv,
    images_3d_path=images_3d_path, overwrite_temp=True, fps=camera_frame_rate)

# %% 3D validation
