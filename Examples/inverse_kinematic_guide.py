#!python36
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Guidelines and NCams code to run to perform an inverse kinematic analysis of the triangulated data.
Requires OpenSim 4 https://simtk.org/frs/index.php?group_id=91

Intended to be used in an interactive environment (e.g. Spyder).
Has following steps:
0. Import modules
1. Translate the triangulated data into OSim trc format
2. Scale the model
3. Run the inverse kinematic tool
4. Record the video of IKs and put it alongside the camera video
"""
# %% 0 Imports
import os
import ncams


# %% 1 Translate the triangulated data into OSim trc format
BASE_DIR = os.path.join('C://', 'FLIR_cameras', 'PublicExample')
proj_path = os.path.join(BASE_DIR, '2019.12.20_8camsNoMarkers-AS-2019-12-23')
triangulated_path = os.path.join(proj_path, 'triangulated_full_rank_0.9', 'session4')
ik_dir = os.path.join(proj_path, 'inverse_kinematics')
if not os.path.isdir(ik_dir):
    os.mkdir(ik_dir)
config_path = os.path.join(proj_path, 'config.yaml')


# load a csv file into a dictionary by specified column names
marker_name_dict = ncams.utils.dic_from_csv('marker_meta.csv', 'sDlcMarker', 'sOpenSimMarker')

triangulated_csv = os.path.join(triangulated_path, 'triangulated_points_4_smoothed.csv')
trc_file = os.path.join(ik_dir, 'triangulated_4_marshmallow.trc')
frame_range = (260, 360)
ncams.inverse_kinematics.triangulated_to_trc(
    triangulated_csv, trc_file, marker_name_dict,
    data_unit_convert=lambda x: x*100,  # dm to mm
    rate=50, zero_marker='scapula_anterior', frame_range=frame_range)

# %% 2 Scale the model
# Select a data subset for which you know the approximate joint angles
# In OpenSim 4 with a desired model loaded use Tools->Scale model
# Guides: https://simtk-confluence.stanford.edu/display/OpenSim/Scaling
# There are also screencasts available on youtube:
# https://www.youtube.com/user/OpenSimVideos/videos
# Manual scaling of each segment is also an option.


# %% 3 Run inverse kinematics
# In OpenSim 4 with a desired model loaded use Tools->Inverse kinematics
# Load the IK settings generated during import of data. TODO(AS)
# Guides: https://simtk-confluence.stanford.edu/display/OpenSim/Inverse+Kinematics


# %% 4 Make videos
# Load the motion generated during inverse kinematics and play it.
# To record a video, press a camera button in the top right corner of the viewer. To stop recording,
# press the button again. Save the video path to 'ik_video_path'.
video_path = os.path.join(BASE_DIR, 'exp_session_2019.12.20_videos', '4_cam19335177.mp4')
ik_video_path = os.path.join(ik_dir, 'marshmallow.webm')
output_path = os.path.join(triangulated_path, 'marshmallow_19335177_4.mp4')
ncams.make_triangulation_video(
    video_path, triangulated_csv, skeleton_config=config_path,
    frame_range=frame_range, output_path=output_path,
    thrd_video_path=ik_video_path, thrd_video_frame_offset=0,  # if the IK movement starts later
    third_video_crop_hw=[slice(50, -100), slice(350, -700)],  # crops the IK video
    figure_dpi=300,
    ranges=((-0.33, 3), (-2, 2), (-1.33, 6.74)))  # manually set ranges for 3D plot
