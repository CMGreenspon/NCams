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

import matplotlib
import matplotlib.pyplot as mpl_pp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Replace this with your working directory
#BASE_DIR = r'*\NCamsCalibrationExampleData\TriangulationPlotting'
BASE_DIR = r'C:\Repositories\NCamsExampleDataExternal\TriangulationPlotting'
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

output_csv_fname = os.path.join(labeled_csv_path, 'triangulated_raw.csv')
ncams.reconstruction.triangulate_csv(ncams_config, labeled_csv_path, intrinsics_config,
                                     extrinsics_config, output_csv_fname=output_csv_fname,
                                     filter_2D=False, filter_3D=False)

#%% 2. Smoothing and filtering in 2D/3D
''' Due to the inherent noise in the keypoint estimation, both the 2D and 3D data can be erratic.
On the assumption that the noise across cameras is uncorrelated, all cameras can be filtered/smoothed
before triangulation occurs and then the subsequent data can again be filtered/smoothed. Both stages 
are optional. Given that these are options in the "triangulate_csv" function, trying each of these out
is trivial. To save time a quick comparison of the combinations of filtering is given below.'''

# Choose a bodypart number, the camera to view and plot the results in 2D & 3D
ibp = 13
serial_oi = 19340300

# Make a CSV for each possible combination of the triangulated output
output_csv_fname = os.path.join(labeled_csv_path, 'triangulated_2D.csv')
ncams.reconstruction.triangulate_csv(ncams_config, labeled_csv_path, intrinsics_config,
                                     extrinsics_config, output_csv_fname=output_csv_fname,
                                     filter_2D=True, filter_3D=False)

output_csv_fname = os.path.join(labeled_csv_path, 'triangulated_3D.csv')
ncams.reconstruction.triangulate_csv(ncams_config, labeled_csv_path, intrinsics_config,
                                     extrinsics_config, output_csv_fname=output_csv_fname,
                                     filter_2D=False, filter_3D=True)

output_csv_fname = os.path.join(labeled_csv_path, 'triangulated_2D3D.csv')
ncams.reconstruction.triangulate_csv(ncams_config, labeled_csv_path, intrinsics_config,
                                     extrinsics_config, output_csv_fname=output_csv_fname,
                                     filter_2D=True, filter_3D=True)


# Load the 2D CSV, get filtered & unfiltered points
csv_list = ncams.utils.get_file_list('.csv', path=labeled_csv_path)
csv_oi = [s for s in csv_list if str(serial_oi) in s][0]
(formatted_2d_points,_) = ncams.reconstruction.process_points(csv_oi, '2D', filtering=False)
unfiltered_2d_points = np.squeeze(formatted_2d_points[:,:,ibp])
(formatted_2d_points,_) = ncams.reconstruction.process_points(csv_oi, '2D', filtering=True)
filtered_2d_points = np.squeeze(formatted_2d_points[:,:,ibp])

# Load the 3D CSVs
csv_oi = [s for s in csv_list if 'triangulated' in s]
legend_3d, points_to_plot = [],[]
for fname in csv_oi: # Get the filenames for the legend
    short_path = os.path.split(fname)[1]
    legend_3d.append(short_path[13:-4])
    
    formatted_3d_points = ncams.reconstruction.process_points(fname, '3D', filtering=False)
    points_to_plot.append(np.squeeze(formatted_3d_points[:,:,ibp]))


# Plot it all    
fig = mpl_pp.figure()

ax1 = fig.add_subplot(121)
ax1.plot(unfiltered_2d_points[:,0], unfiltered_2d_points[:,1])
ax1.plot(filtered_2d_points[:,0], filtered_2d_points[:,1])
ax1.set_xlim([0, ncams_config['image_size'][0]])
ax1.set_ylim([0, ncams_config['image_size'][1]])
ax1.legend(('Unfiltered', 'Filtered'))

ax2 = fig.add_subplot(122, projection='3d')
for p in range(len(points_to_plot)):
    ax2.plot(points_to_plot[p][:,0], points_to_plot[p][:,1],points_to_plot[p][:,2])
    
ax2.legend(legend_3d)

#%% 3. Interactive 3D plotting
''' In order to make inspecting the outputs of the triangulation easy it is we found it best to 
view it alongside a video. The interactive tool allows scrubbing of individual frames, ideal for
cropping and rotating the projection, as well as inspecting the movement of individual markers.
'''
vid_name = r'trial10_cam19340300DLC_resnet50_SR_2020.08.07Aug7shuffle1_250000_labeled.mp4'
vid_path = os.path.join(labeled_csv_path, vid_name)
dlc_config_path = os.path.join(BASE_DIR, 'DLC_config.yaml') # For the skeleton

ncams.reconstruction.interactive_3d_plot(vid_path, output_csv_fname, skeleton_path=dlc_config_path)


#%% 4. Exporting triangulated videos
''' If one wishes to save the full or cropped version of the video with reconstruction the following
function may be used. There are many options such as adding a third panel for skeletal reconstruction
or an alternate view.
'''

ncams.reconstruction.make_triangulation_video(vid_path, output_csv_fname,
                                              skeleton_config=dlc_config_path, view=(90, 120))
#%% 5. Triangulation of individual points
raise Warning('Not complete.')
''' This example serves to demonstrate how a given point can be triangulated by itself if desired
for whatever reason. This is not an efficient or particularly useful thing to do but through this
several underlying functions are demonstrated. From the below framework any variation on the supplied
triangulation should be easy.'''

# Create projection matrices for each camera
projection_matrices = []
for icam in range(len(ncams_config['serials'])):
    projection_matrices.append(ncams.camera_tools.make_projection_matrix(
        intrinsics_config['camera_matrices'][icam],
        extrinsics_config['world_orientations'][icam],
        extrinsics_config['world_locations'][icam]))
    
