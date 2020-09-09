# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:32:30 2020

@author: somlab
"""

"""
NCams Toolbox
Copyright 2020 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Example script for using the various triangulation/reconstruction functions.
    1. Triangulation from multiple CSVs
    2. Smoothing & filtering in 2D/3D
    3. Interactive 3D plotting
    4. Exporting triangulated videos
    5. Triangulation of individual points (not streamlined)
"""

import ncams
import os
import time
import numpy as np
import cv2

# Replace this with your working directory
#BASE_DIR = r'*\NCamsCalibrationExampleData'
BASE_DIR = r'C:\Users\somlab\Desktop\NCamsCalibrationExampleData'
# Load calibrations (see intrinsic_extrinsic_calibration.py, section 5)
path_to_ncams_config = os.path.join(BASE_DIR, 'ncams_config.yaml')
ncams_config = ncams.camera_io.yaml_to_config(path_to_ncams_config)
intrinsics_config, extrinsics_config = ncams.camera_io.load_calibrations(ncams_config)

#%% 1. Triangulation from multiple CSVs
''' The most common usage for the CSV based triangulation is to direct NCams towards a folder
containing a CSV (in the format of DeepLabCut) for each camera where each CSV has time matched rows 
for each camera (note that the serial number of the camera is used to identify which CSV belongs
to which camera). The following file structure is recommended:
    
- Project/Animal/Session Folder
    - Trial_N
        - Trial_N_cam12345678_DLC.csv
        - Trial_N_cam23456789_DLC.csv
        
By collecting a list of trial directories each can be passed to the "triangulate_csv" function in a 
for loop. The following example shows how to deal with an individual trial.'''

# Declare the folder containing the CSVs for each camera and the desired name
labeled_csv_path = os.path.join(BASE_DIR, 'trial10')

ncams.reconstruction.triangulate_csv(ncams_config, labeled_csv_path, intrinsics_config,
                                     extrinsics_config, output_csv_fname=None, threshold=0.9,
                                     method='full_rank', best_n=2, centroid_threshold=2.5,
                                     iteration=None, undistorted_data=False, file_prefix='',
                                     filter_2D=False, filter_3D=False)

#%% 5. Triangulation of individual points
''' This example serves to demonstrate how a given point can be triangulated by itself if desired
for whatever reason. This is not an efficient or particularly useful thing to do but through this
several underlying functions are demonstrated.'''

# Create projection matrices for each camera
projection_matrices = []
for icam in range(len(ncams_config['serials'])):
    projection_matrices.append(ncams.camera_tools.make_projection_matrix(
        intrinsics_config['camera_matrices'][icam],
        extrinsics_config['world_orientations'][icam],
        extrinsics_config['world_locations'][icam]))
    
