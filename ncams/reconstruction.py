#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019-2020 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Functions related to triangulation of marker positions from multiple cameras.

For more details on the camera data structures and dicts, see help(ncams.camera_tools).
"""
import os
import re
import csv
import shutil
import math
import multiprocessing
import functools
import ntpath
import yaml
import cv2
import warnings

from glob import glob
import numpy as np
from tqdm import tqdm
from scipy.signal import medfilt
from scipy.spatial.distance import euclidean
from scipy.stats import norm
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
from itertools import combinations

import matplotlib
import matplotlib.pyplot as mpl_pp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from . import utils
from . import camera_tools


FIG = None
FIGNUM = None
AXS = None
SLIDER = None

def triangulate(image_coordinates, projection_matrices):
    '''
    The base triangulation function for NCams. Takes image coordinates and projection matrices from
    2+ cameras and will produce a triangulated point with the desired approach.

    Arguments:
        image_coordinates {array or list of} -- the x,y coordinates of a given marker for multiple
            cameras. The points must be in the format (1,2) if in a list or (n,2) if an array.
        projection_matrices {list} -- the projection matrices for the cameras corresponding
        to each image points input.

    Keyword Arguments:
        mode {str} -- the triangulation method to use:
            full_rank - performs SVD to find the point with the least squares error between all
                projection lines. If a threshold is given along with confidence values then only
                points above the threshold will be used.
            best_n - uses the n number of cameras with the highest confidence values for the
                triangulation. If a threshold is given then only points above the threshold will
                be considered.
            cluster - [in development] performs all combinations of triangulations and checks for
                outlying points suggesting erroneous image coordinates from one or more cameras.
                After removing the camera(s) that produce out of cluser points it then performs the
                full_rank triangulation.
        confidence_values {list or array} -- the confidence values for the points given by the
            marking system (e.g. DeepLabCut)
        threshold {float} -- the minimum confidence value to accept for triangulation.

    Output:
        u_3d {(1,3) np.array} -- the triangulated point produced.

    '''
    u_3d = np.zeros((1,3))
    u_3d.fill(np.nan)

    # Check if image coordinates are formatted properly
    if isinstance(image_coordinates, list):
        if len(image_coordinates) > 1:
            image_coordinates = np.vstack(image_coordinates)
        else:
            return u_3d

    if not np.shape(image_coordinates)[1] == 2:
        raise ValueError('ncams.reconstruction.triangulate only accepts numpy.ndarrays or lists of' +
                         'in the format (camera, [x,y])')

    num_cameras = np.shape(image_coordinates)[0]
    if num_cameras < 2: # return NaNs if insufficient points to triangulate
        return u_3d

    if num_cameras != len(projection_matrices):
        raise ValueError('Different number of coordinate pairs and projection matrices given.')

    decomp_matrix = np.empty((num_cameras*2, 4))
    for decomp_idx in range(num_cameras):
        point_mat = image_coordinates[decomp_idx]
        projection_mat = projection_matrices[decomp_idx]

        temp_decomp = np.vstack([
            [point_mat[0] * projection_mat[2, :] - projection_mat[0, :]],
            [point_mat[1] * projection_mat[2, :] - projection_mat[1, :]]])

        decomp_matrix[decomp_idx*2:decomp_idx*2 + 2, :] = temp_decomp

    Q = decomp_matrix.T.dot(decomp_matrix)
    u, _, _ = np.linalg.svd(Q)
    u = u[:, -1, np.newaxis]
    u_3d = np.transpose((u/u[-1, :])[0:-1, :])

    return u_3d

def triangulate_csv_OLD(ncams_config, output_csv, intrinsics_config, extrinsics_config,
                labeled_csv_path, threshold=0.5, method='full_rank',
                best_n=2, num_frames_limit=None, iteration=None, undistorted_data=False,
                file_prefix=''):
    '''Triangulates points from multiple cameras and exports them into a csv.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). This function uses following keys:
            serials {list of numbers} -- list of camera serials.
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
        output_csv {str} -- file to save the triangulated points into.
        intrinsics_config {dict} -- see help(ncams.camera_tools).
        extrinsics_config {dict} -- see help(ncams.camera_tools).
        labeled_csv_path {str} -- locations of csv's with marked points.
    Keyword Arguments:
        threshold {number 0-1} -- only points with confidence (likelihood) above the threshold will
            be used for triangulation. (default: 0.9)
        method {'full_rank' or 'best_pair'} -- method for triangulation.
            full_rank: uses all available cameras
            best_n: uses best n cameras to locate the point.
            (default: 'full_rank')
        best_n {number} -- how many cameras to use when best_n method is used. (default: 2)
        num_frames_limit {number or None} -- limit to the number of frames used for analysis. Useful
            for testing. If None, then all frames will be analyzed. (default: None)
        iteration {int} -- look for csv's with this iteration number. (default: {None})
        undistorted_data {bool} -- if the marker data was made on undistorted videos. (default:
            {False})
        file_prefix {string} -- prefix of the csv file to search for in the folder. (default: {''})
    Output:
        output_csv {csv file} -- csv containing all triangulated points.
    '''

    cam_serials = ncams_config['serials']

    camera_matrices = intrinsics_config['camera_matrices']
    if not undistorted_data:
        distortion_coefficients = intrinsics_config['distortion_coefficients']

    world_locations = extrinsics_config['world_locations']
    world_orientations = extrinsics_config['world_orientations']

    if not method in ('full_rank', 'best_n', 'centroid'):
        raise ValueError('{} is not an accepted method. '
                         'Please use "full_rank", "best_n", or "centroid".'.format('"'+method+'"'))

    # Get data files
    list_of_csvs = get_list_labelled_csvs(ncams_config, labeled_csv_path,
                                          file_prefix=file_prefix, iteration=iteration)

    # Load them
    csv_arrays = [[] for _ in list_of_csvs]
    for ifile, csvfname in enumerate(list_of_csvs):
        with open(csvfname) as csvfile:
            reader_object = csv.reader(csvfile, delimiter=',')
            for row in reader_object:
                csv_arrays[ifile].append(row)

    # Get the list of bodyparts - this way doesn't require loading in a yaml though that might be
    # better for skeleton stuff
    temp_bodyparts = csv_arrays[0][1]
    bodypart_idx = np.arange(1, len(temp_bodyparts)-2, 3)
    bodyparts = []
    for idx in bodypart_idx:
        bodyparts.append(temp_bodyparts[idx])

    # Format the data
    num_cameras = len(csv_arrays)
    num_bodyparts = len(bodyparts)
    num_frames = len(csv_arrays[0])-3
    if num_frames_limit is not None and num_frames > num_frames_limit:
        num_frames = num_frames_limit

    image_coordinates, thresholds = [], []
    for icam in range(num_cameras):
        # Get the numerical data
        csv_array = np.vstack(csv_arrays[icam][3:])
        csv_array = csv_array[:num_frames, 1:]
        # Get coordinate and confidence idxs
        if icam == 0:
            confidence_idx = np.arange(2, np.shape(csv_array)[1], 3)
            coordinate_idx = []
            for idx in range(np.shape(csv_array)[1]):
                if not np.any(confidence_idx == idx):
                    coordinate_idx.append(idx)

        # Separate arrays
        coordinate_array = csv_array[:, coordinate_idx]
        threshold_array = csv_array[:, confidence_idx]

        # Format the coordinate array
        formatted_coordinate_array = np.empty((num_frames, 2, num_bodyparts))
        for ibp in range(num_bodyparts):
            formatted_coordinate_array[:, :, ibp] = coordinate_array[:, [ibp*2, ibp*2+1]]

        # Append to output lists
        image_coordinates.append(formatted_coordinate_array)
        thresholds.append(threshold_array)

    # Undistort the points and then threshold
    output_coordinates_filtered = []
    for icam in range(num_cameras):
        # output_array = np.empty((num_frames, 2, num_bodyparts))
        filtered_output_array = np.empty((num_frames, 2, num_bodyparts))
        # The filtered one needs NaN points so we know which to ignore
        filtered_output_array.fill(np.nan)

        # Get the sufficiently confident values for each bodypart
        for bodypart in range(num_bodyparts):
            # Get the distorted points
            distorted_points = image_coordinates[icam][:, :, bodypart]
            if undistorted_data:
                undistorted_points = distorted_points.reshape(np.shape(distorted_points)[0], 1, 2)
            else: # Undistort them
                undistorted_points = cv2.undistortPoints(
                    distorted_points, camera_matrices[icam],
                    distortion_coefficients[icam], None, P=camera_matrices[icam])

            # Get threshold filter
            bp_thresh = thresholds[icam][:, bodypart].astype(np.float32) > threshold
            thresh_idx = np.where(bp_thresh == 1)[0]
            # Put them into the output array
            for idx in thresh_idx:
                filtered_output_array[idx, :, bodypart] = undistorted_points[idx, 0, :]

        # output_coordinates.append(output_array)
        output_coordinates_filtered.append(filtered_output_array)

    # Triangulation
    # Make the projection matrices
    projection_matrices = []
    for icam in range(num_cameras):
        projection_matrices.append(camera_tools.make_projection_matrix(
            camera_matrices[icam], world_orientations[icam], world_locations[icam]))

    # Triangulate the points
    triangulated_points = np.empty((num_frames, 3, num_bodyparts))
    triangulated_points.fill(np.nan)

    for iframe in range(num_frames):
        for bodypart in range(num_bodyparts):
            # Get points for each camera
            cam_image_points = np.empty((2, num_cameras))
            cam_image_points.fill(np.nan)
            if method == 'full_rank':
                for icam in range(num_cameras):
                    cam_image_points[:, icam] = output_coordinates_filtered[icam][iframe, :, bodypart]
            elif method == 'best_n':
                # decorate-sort-undecorate sort to find the icams for the highest likelihood
                best_likelh = [b[0] for b in sorted(
                    zip(range(num_cameras),
                        [thresholds[icam][iframe, bodypart].astype(np.float64)
                         for icam in range(num_cameras)]),
                    key=lambda x: x[1], reverse=True)][:best_n]
                for icam in [icam for icam in range(num_cameras) if icam in best_likelh]:
                    cam_image_points[:, icam] = output_coordinates_filtered[icam][iframe, :, bodypart]

            # Check how many cameras detected the bodypart in that frame
            cams_detecting = ~np.isnan(cam_image_points[0, :])
            cam_idx = np.where(cams_detecting)[0]
            if np.sum(cams_detecting) < 2:
                continue

            # Create the image point and projection matrices
            tri_projection_mats, tri_image_points = [], []
            for cam in cam_idx:
                tri_image_points.append(cam_image_points[:, cam])
                tri_projection_mats.append(projection_matrices[cam])

            triangulated_points[iframe, :, bodypart] = triangulate(tri_image_points, tri_projection_mats)

    with open(output_csv, 'w', newline='') as f:
        triagwriter = csv.writer(f)
        bps_line = ['bodyparts']
        for bp in bodyparts:
            bps_line += [bp]*3
        triagwriter.writerow(bps_line)
        triagwriter.writerow(['coords'] + ['x', 'y', 'z']*num_bodyparts)
        for iframe in range(num_frames):
            rw = [iframe]
            for ibp in range(num_bodyparts):
                rw += [triangulated_points[iframe, 0, ibp],
                       triangulated_points[iframe, 1, ibp],
                       triangulated_points[iframe, 2, ibp]]
            triagwriter.writerow(rw)
    return output_csv


def get_list_labelled_csvs(ncams_config, labeled_csv_path, file_prefix='', iteration=None):
    '''Returns a list of ML-labelled CSV files from individual cameras in a directory.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). This function uses following keys:
            serials {list of numbers} -- list of camera serials.
        labeled_csv_path {str} -- locations of csv's with marked points. TODO: accept a list of
            files.
        file_prefix {string} -- prefix of the csv file to search for in the folder. (default: {''})
        iteration {int} -- look for csv's with this iteration number. (default: {None})
    Output:
        list_of_csvs {list of str} -- list of filenames matching the structure of ML-labelled 2D
            marker positions.
    '''
    cam_serials = ncams_config['serials']
    list_of_csvs = []
    for cam_serial in cam_serials:
        if iteration is None:
            sstr = '*.csv'
        else:
            sstr = '*_{}.csv'.format(iteration)
        list_of_csvs += glob(os.path.join(
            labeled_csv_path, file_prefix+'*'+ str(cam_serial) + sstr))

    if len(list_of_csvs) == 0:
        raise ValueError('No CSVs found in provided path.')
    elif not len(list_of_csvs) == len(cam_serials):
        if iteration is not None:
            raise ValueError('Detected {} csvs in {} with iteration #{} while was provided with {}'
                  ' serials.'.format(
                len(list_of_csvs), labeled_csv_path, iteration, len(cam_serials)))
        iterations = set()
        for csv_f in list_of_csvs:
            iterations.add(int(re.search('_[0-9]+.csv$', csv_f)[0][1:-4]))
        print('Detected {} csvs in {} while was provided with {} serials.'
              ' Found iterations: {}'.format(
            len(list_of_csvs), labeled_csv_path, len(cam_serials), sorted(iterations)))

        uinput_string = ('Provide iteration number to use: ')
        uinput = input(uinput_string)
        list_of_csvs = [i for i in list_of_csvs if re.fullmatch('.*_{}.csv'.format(uinput), i)]
        if len(list_of_csvs) > len(cam_serials):
            raise ValueError('Detected {} CSVs in {} with iteration #{} while was provided with {}'
                  ' serials.'.format(
                len(list_of_csvs), labeled_csv_path, uinput, len(cam_serials)))
        elif len(list_of_csvs) < len(cam_serials):
            raise ValueError('Fewer CSVs than cameras detected.')
    return list_of_csvs


def load_labelled_csvs(list_of_csvs, threshold=0.9, filtering=False, only_bodyparts=None,
                       skip_bodyparts=[]):
    '''Load the data from labelled 2D csvs.

    Arguments:
        list_of_csvs {list of str} -- list of filenames matching the structure of ML-labelled 2D
            marker positions.
    Keyword Arguments:
        threshold {number 0-1} -- only points with confidence (likelihood) above the threshold will
            be loaded. (default: 0.9)
        filtering {bool} -- if true, will filter the imported data. (default: False)
        only_bodyparts {list} -- TODO(might need changes to process_points and architecture change) only displays information about these bodyparts. If None, display
            all. (default: {None})
        skip_bodyparts {list} -- TODO(might need changes to process_points and architecture change) ignores these bodyparts. Runs after the bodyparts from
            only_bodyparts list are selected. (default: {[]})
    Outputs a tuple consisting of:
        bodyparts {list of str} -- names of all loaded markers.
        num_frames {int} -- number of frames in each file. Truncated to smallest.
        image_coordinates {list [num csvs]ndarray([num frame,num axes,num bodypart])}
        ic_confidences {list [num csvs]ndarray([num frame,num bodypart])}
    '''
    frame_count = []
    csv_arrays = [[] for _ in list_of_csvs]
    for ifile, csvfname in enumerate(list_of_csvs):
        with open(csvfname) as csvfile:
            reader_object = csv.reader(csvfile, delimiter=',')
            for row in reader_object:
                csv_arrays[ifile].append(row)

            frame_count.append(int(len(csv_arrays[ifile])-3))

    # Check frame count
    frame_count_match = all(x==frame_count[0] for x in frame_count)
    if frame_count_match:
        num_frames = frame_count[0]
    else:
        num_frames = min(frame_count)
        print('Warning: Not all CSVs have the same number of rows. Truncating to shortest.')

    # Get the list of bodyparts
    temp_bodyparts = csv_arrays[0][1]
    bodypart_idx = np.arange(1, len(temp_bodyparts)-2, 3)
    bodyparts = []
    for idx in bodypart_idx:
        bodyparts.append(temp_bodyparts[idx])

    # trim the csv_arrays

    # Format the data
    image_coordinates, ic_confidences = [], []
    for icam in range(len(csv_arrays)):
        csv_array = np.vstack(csv_arrays[icam][3:])[:,1:] # Remove header rows, and first column
        if not frame_count_match:
            csv_array = csv_array[:num_frames,:]
        # Reshape and threshold the data
        ic_array, ic_confidence = process_points(csv_array, '2D', threshold=threshold,
                                   filtering=filtering)
        image_coordinates.append(ic_array)
        ic_confidences.append(ic_confidence)

    return bodyparts, num_frames, image_coordinates, ic_confidences


def undistort_point(distorted_points, camera_matrix, distortion_coefficient):
    return np.squeeze(cv2.undistortPoints(
        distorted_points, camera_matrix, distortion_coefficient, None,
        P=camera_matrix))


def triangulate_point(method, num_cameras, uics, iccs, projection_matrices, best_n=2,
                      centroid_threshold=2.5):
    cam_image_points = np.empty((2, num_cameras))
    cam_image_points.fill(np.nan)

    if method == 'full_rank' or method == 'centroid':
        for icam in range(num_cameras):
            cam_image_points[:, icam] = uics[icam]
    elif method == 'best_n':
        # decorate-sort-undecorate sort to find the icams for the highest likelihood
        best_likelh = [b[0] for b in sorted(
            zip(range(num_cameras), [iccs[icam].astype(np.float64) for icam in range(num_cameras)]),
            key=lambda x: x[1], reverse=True)][:best_n]
        for icam in [icam for icam in range(num_cameras) if icam in best_likelh]:
            cam_image_points[:, icam] = uics[icam]

    # Check how many cameras detected the bodypart in that frame
    cams_detecting = ~np.isnan(cam_image_points[0, :])
    cam_idx = np.where(cams_detecting)[0]
    if np.sum(cams_detecting) < 2:
        return None

    if method == 'full_rank' or method == 'best_n':
        # Create the image point and projection matrices
        tri_projection_mats, tri_image_points = [], []
        for cam in cam_idx:
            tri_image_points.append(cam_image_points[:, cam])
            tri_projection_mats.append(projection_matrices[cam])

        return triangulate(tri_image_points, tri_projection_mats)

    elif method == 'centroid':
        cam_comb_list = list(combinations(cam_idx, 2))
        num_combs = len(cam_comb_list)
        t_points = np.zeros((num_combs, 3))
        for c in range(num_combs):
            tri_projection_mats, tri_image_points = [], []
            for cam in cam_comb_list[c]:
                tri_image_points.append(cam_image_points[:, cam])
                tri_projection_mats.append(projection_matrices[cam])

            t_points[c, :] = triangulate(tri_image_points, tri_projection_mats)
        # Take the centroid of the points
        t_centroid = np.mean(t_points, axis=0)

        # Check for outliers if there are sufficient points to do so
        if num_combs > 3:
            t_cent_dist = []
            for c in range(num_combs):
                t_cent_dist.append(euclidean(t_centroid, t_points[c, :]))
            t_cent_dist = np.vstack(t_cent_dist)
            # Get z-scores for the distances from the centroid
            euclid_sd = np.std(t_cent_dist)
            euclid_threshold = euclid_sd * centroid_threshold
            dist_bool = t_cent_dist < euclid_threshold

            if np.sum(dist_bool) < num_combs:  # Recalculate the centroid
                cent_idx = np.where(dist_bool)[0]
                t_points_filt = t_points[cent_idx, :]
                t_centroid = np.mean(t_points_filt, axis=0)

        return t_centroid


def triangulate_points(
        ncams_config, intrinsics_config, extrinsics_config,
        bodyparts, num_frames, image_coordinates, ic_confidences,
        threshold=0.9, method='full_rank', best_n=2,
        centroid_threshold=2.5, undistorted_data=False,
        filter_3D=False, custom_3D_filter=None):
    # check if configs are not None
    if intrinsics_config is None:
        raise ValueError('No intrinsic configuration provided.')
    if extrinsics_config is None:
        raise ValueError('No extrinsics configuration provided.')

    camera_matrices = intrinsics_config['camera_matrices']
    if not undistorted_data:
        distortion_coefficients = intrinsics_config['distortion_coefficients']

    world_locations = extrinsics_config['world_locations']
    world_orientations = extrinsics_config['world_orientations']

    if method not in ('full_rank', 'best_n', 'centroid'):
        raise ValueError('"{}" is not an accepted method. '
                         'Please use "full_rank", "best_n", or "centroid".'.format(method))

    num_cameras = len(ncams_config['serials'])
    num_bodyparts = len(bodyparts)

    if not undistorted_data:  # Undistort points
        undistorted_image_coordinates = []
        # for each camera
        for cam_image_coordinates, camera_matrix, distortion_coefficient in zip(
                image_coordinates, camera_matrices, distortion_coefficients):
            undistorted_csv_array = np.empty(cam_image_coordinates.shape)
            undistorted_csv_array.fill(np.nan)
            for bp in range(num_bodyparts):
                undistorted_csv_array[:, :, bp] = undistort_point(
                    cam_image_coordinates[:, :, bp],
                    camera_matrix, distortion_coefficient)

            undistorted_image_coordinates.append(undistorted_csv_array)

    else:
        undistorted_image_coordinates = image_coordinates

    # Triangulation
    # Make the projection matrices
    projection_matrices = []
    for icam in range(num_cameras):
        projection_matrices.append(camera_tools.make_projection_matrix(
            camera_matrices[icam], world_orientations[icam], world_locations[icam]))

    # Triangulate the points
    triangulated_points = np.empty((num_frames, 3, len(bodyparts)))
    triangulated_points.fill(np.nan)

    for iframe in range(num_frames):
        for bodypart in range(len(bodyparts)):
            triangulated_point = triangulate_point(
                method, num_cameras,
                [undistorted_image_coordinates[icam][iframe, :, bodypart]
                 for icam in range(num_cameras)],
                [ic_confidences[icam][iframe, bodypart] for icam in range(num_cameras)],
                projection_matrices,
                best_n=best_n, centroid_threshold=centroid_threshold)

            if triangulated_point is not None:
                triangulated_points[iframe, :, bodypart] = triangulated_point

    if filter_3D:
        triangulated_points = process_points(triangulated_points, '3D', threshold=threshold)

    if custom_3D_filter is not None:
        triangulated_points = custom_3D_filter(bodyparts, triangulated_points)

    return triangulated_points


def triangulate_csv(ncams_config, labeled_csv_path, intrinsics_config, extrinsics_config,
                    output_csv_fname=None, threshold=0.9, method='full_rank', best_n=2,
                    centroid_threshold=2.5, iteration=None, undistorted_data=False, file_prefix='',
                    filter_2D=False, filter_3D=False, custom_3D_filter=None):

    '''Triangulates points from multiple cameras and exports them into a csv.

    TODO: Calculate a confidence metric for a 3D point based on the confidence of 2D points.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). This function uses following keys:
            serials {list of numbers} -- list of camera serials.
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
        output_csv {str} -- file to save the triangulated points into.
        intrinsics_config {dict} -- see help(ncams.camera_tools).
        extrinsics_config {dict} -- see help(ncams.camera_tools).
        labeled_csv_path {str} -- locations of csv's with marked points. TODO: accept a list of
            files.
    Keyword Arguments:
        threshold {number 0-1} -- only points with confidence (likelihood) above the threshold will
            be used for triangulation. (default: 0.9)
        method {'full_rank' or 'best_pair'} -- method for triangulation.
            full_rank: uses all available cameras
            best_n: uses best n cameras to locate the point.
            TODO: add centroid/cluster description
            (default: 'full_rank')
        best_n {number} -- how many cameras to use when best_n method is used. (default: 2)
        iteration {int} -- look for csv's with this iteration number. (default: {None})
        undistorted_data {bool} -- if the marker data was made on undistorted videos. (default:
            {False})
        file_prefix {string} -- prefix of the csv file to search for in the folder. (default: {''})
        filter_2D {bool} -- filter the imported 2D data. (default: False)
        filter_3D {bool} -- filter the produced 3D data. (default: False)
        custom_3D_filter {None or callable} -- optional processing the 3D data after filter_3D. For
            example, remove outlier points. Has to accept (bodyparts, triangulated_points) and
            return triangulate_points. (default: None)
    Output:
        output_csv {csv file} -- csv containing all triangulated points.
        output_csv_fname {string} -- returns the filename of the produced file.
    '''

    # Check if the source CSV path exists
    if not os.path.exists(labeled_csv_path):
        raise ValueError('Provided path for CSVs does not exist.')

    # Get data files
    list_of_csvs = get_list_labelled_csvs(ncams_config, labeled_csv_path,
                                          file_prefix=file_prefix, iteration=iteration)

    # Load them
    bodyparts, num_frames, image_coordinates, ic_confidences = load_labelled_csvs(
        list_of_csvs, threshold=threshold, filtering=filter_2D)

    triangulated_points = triangulate_points(
        ncams_config, intrinsics_config, extrinsics_config,
        bodyparts, num_frames, image_coordinates, ic_confidences,
        threshold=threshold, method=method, best_n=best_n,
        centroid_threshold=centroid_threshold, undistorted_data=undistorted_data,
        filter_3D=filter_3D, custom_3D_filter=custom_3D_filter)

    if output_csv_fname is None:
        _, dir_name = os.path.split(labeled_csv_path)
        output_csv_fname = os.path.join(labeled_csv_path, dir_name + '_triangulated.csv')
    else:  # check if correct delimiter
        if not os.path.split(output_csv_fname)[1][-4:] == '.csv':
            output_csv_fname = output_csv_fname + '.csv'

    # TODO use io_tools
    with open(output_csv_fname, 'w', newline='') as f:
        triagwriter = csv.writer(f)
        bps_line = ['bodyparts']
        for bp in bodyparts:
            bps_line += [bp]*3
        triagwriter.writerow(bps_line)
        triagwriter.writerow(['coords'] + ['x', 'y', 'z']*len(bodyparts))
        for iframe in range(num_frames):
            rw = [iframe]
            for ibp in range(len(bodyparts)):
                rw += [triangulated_points[iframe, 0, ibp],
                       triangulated_points[iframe, 1, ibp],
                       triangulated_points[iframe, 2, ibp]]
            triagwriter.writerow(rw)

    return output_csv_fname


def process_points(path_or_array, csv_type, filt_width=5, threshold=0.9, filtering=True):
    '''Formats and processes CSVs or numpy arrays as necessary for further usage.
       Uses median and gaussian filters to both smooth and interpolate points.
       Will only interpolate when fewer missing values are present than the gaussian width.

    Arguments:
        path_or_array {str} -- path of the triangulated csv or a numpy array (2 or 3D).
        csv_type {str} -- indicator of whether or not the array is '2D' or '3D'
    Keyword Arguments:
        filt_width {int} -- how wide the filters should be. (default: 5)
        threshold {float} -- confidence threshold to filter 2D DLC data by (default: 0.9).
        filtering {bool} -- whether or not to perform median and gaussian filters (default: True).
    Outputs if csv_type == '2D' is a tuple:
        processed_point_array {ndarray([num frame,num axes,num bodypart])}
        formatted_confidence_values {ndarray([num frame,num bodypart])}
    Output if csv_type == '3D':
        processed_point_array {ndarray([num frame,num axes,num bodypart])}
    '''
    # Check if the input is an array or a path
    if type(path_or_array) == str: # Assume it's DLC output or CSV output from triangulation
        # Load in the CSV
        with open(path_or_array, 'r') as f:
            csv_reader = csv.reader(f)
            # Check if the csv is a DLC output or a triangulation output
            csv_row = next(csv_reader)
            if csv_type == '2D':
                csv_row = next(csv_reader)

            # Get the names of the bodyparts for storage
            bodyparts = []
            for i, bp in enumerate(csv_row):
                if (i-1)%3 == 0:
                    bodyparts.append(bp)
            num_bodyparts = len(bodyparts)

            next(csv_reader) # Skip the 'xyz/xyc title row'
            point_array = []
            for row in csv_reader:
                point_array.append([[] for _ in range(3)])
                for ibp in range(num_bodyparts):
                    point_array[-1][0].append(float(row[1+ibp*3]))
                    point_array[-1][1].append(float(row[2+ibp*3]))
                    point_array[-1][2].append(float(row[3+ibp*3]))

        point_array = np.array(point_array)

    elif type(path_or_array) == np.ndarray: # Assume it's ncam working array
        if len(path_or_array.shape) == 2: # Flat CSV format
            num_bodyparts = int(path_or_array.shape[1]/3)
            n_frames = int(path_or_array.shape[0])
            point_array = []
            for f in range(n_frames):
                row = path_or_array[f,:]
                point_array.append([[] for _ in range(3)])
                for ibp in range(num_bodyparts):
                    point_array[-1][0].append(float(row[ibp*3]))
                    point_array[-1][1].append(float(row[1+ibp*3]))
                    point_array[-1][2].append(float(row[2+ibp*3]))

            point_array = np.array(point_array)

        elif len(path_or_array.shape) == 3: # Already formatted
            point_array = path_or_array
            num_bodyparts = int(path_or_array.shape[2])
            n_frames = int(path_or_array.shape[0])

    else:
        raise ValueError('Incompatible type given to "path_or_array". Must be "str" or "ndarray".')

    # Threshold filtering
    if csv_type == '2D':
        thresholded_point_array = np.empty((point_array.shape[0], 2, point_array.shape[2]))
        thresholded_point_array.fill(np.nan)
        formatted_confidence_values = np.empty((point_array.shape[0], point_array.shape[2]))
        formatted_confidence_values.fill(np.nan)
        for ibp in range(num_bodyparts):
            c_vals = np.squeeze(point_array[:,2,ibp])
            c_idx = np.where(c_vals > threshold)[0]
            ibp_vals = np.squeeze(point_array[:,:2,ibp])
            thresholded_point_array[c_idx,:,ibp] = ibp_vals[c_idx,:]
            formatted_confidence_values[c_idx,ibp] = c_vals[c_idx]

        point_array = thresholded_point_array

    # gaussian and median filtering
    if filtering:
        num_axes = point_array.shape[1]
        # Smooth each bodypart along each axis
        processed_point_array = np.empty(point_array.shape)
        processed_point_array.fill(np.nan)
        gauss_filt = Gaussian1DKernel(stddev=filt_width/10)
        for ibp in range(num_bodyparts):
            for a in range(num_axes):
                ibp_a = np.squeeze(point_array[:,a,ibp])
                # Apply median filter
                ibp_a = _nanmedianfilt(ibp_a, filt_width)
                # Apply gaussian filter
                ibp_a_gauss = convolve(ibp_a, gauss_filt, boundary='extend', nan_treatment='interpolate')
                processed_point_array[:,a,ibp] = ibp_a_gauss
    else:
        processed_point_array = point_array

    # return
    if csv_type == '2D':
        return processed_point_array, formatted_confidence_values
    else:
        return processed_point_array


def _nanmedianfilt(input_vector, kernel_width):
    '''Median filter that ignores nan values'''
    if kernel_width % 2 == 0:
        kernel_width = kernel_width + 1

    kernel_offset = int((kernel_width-1)/2)

    output_vector = np.empty(input_vector.shape)
    output_vector.fill(np.nan)

    init_idx = int(np.ceil(kernel_width/2))
    term_idx = int(len(input_vector) - np.ceil(kernel_width/2))

    output_vector[:init_idx] = input_vector[:init_idx]
    output_vector[term_idx:] = input_vector[term_idx:]
    for idx in np.arange(init_idx, term_idx):
        vals_to_filt = input_vector[idx-kernel_offset:idx+kernel_offset+1]
        num_nans = sum(np.isnan(vals_to_filt))
        if num_nans < kernel_offset:
            output_vector[idx] = np.nanmedian(vals_to_filt)

    return output_vector


def process_triangulated_data(csv_path, filt_width=5, outlier_sd_threshold=5, output_csv=None):
    '''Uses median and gaussian filters to both smooth and interpolate points.
       Will only interpolate when fewer missing values are present than the gaussian width.
       Arguments:
        csv_path {str} -- path of the triangulated csv.
    Keyword Arguments:
        filt_width {int} -- how wide the filters should be. (default: 5)
        outlier_sd_threshold {int} -- How many standard deviations before a point should be
        considered an outlier. (default: 5)
        output_csv {str} -- filename for the output smoothed csv. (default: {csv_path +
            _smoothed.csv})
    '''
    print('Warning: This function has been depreciated. Use process_points instead.')

    # Load in the CSV
    with open(csv_path, 'r') as f:
        triagreader = csv.reader(f)
        l = next(triagreader)
        bodyparts = []
        for i, bp in enumerate(l):
            if (i-1)%3 == 0:
                bodyparts.append(bp)
        num_bodyparts = len(bodyparts)
        next(triagreader)
        triangulated_points = []
        for row in triagreader:
            triangulated_points.append([[] for _ in range(3)])
            for ibp in range(num_bodyparts):
                triangulated_points[-1][0].append(float(row[1+ibp*3]))
                triangulated_points[-1][1].append(float(row[2+ibp*3]))
                triangulated_points[-1][2].append(float(row[3+ibp*3]))

    processed_array = np.array(triangulated_points)

    # Smooth each bodypart along each axis
    gauss_filt = Gaussian1DKernel(stddev=filt_width/10)

    for ibp in range(num_bodyparts):
        for a in range(3):
            ibp_a = np.squeeze(processed_array[:,a,ibp])
            # Apply median filter
            ibp_a = medfilt(ibp_a, kernel_size=filt_width)
            # Outlier detection
            mean_val = np.nanmean(ibp_a)
            std_val = np.nanstd(ibp_a)
            ut = mean_val + std_val*outlier_sd_threshold
            lt = mean_val - std_val*outlier_sd_threshold
            ibp_a = [np.nan if e > ut or e < lt else e for e in ibp_a]
            # Apply gaussian smoothing filter
            processed_array[:,a,ibp] = convolve(ibp_a, gauss_filt,boundary='extend')

    if output_csv is None:
        output_csv = csv_path[:-4] + '_smoothed.csv'

    with open(output_csv, 'w', newline='') as f:
        triagwriter = csv.writer(f)
        bps_line = ['bodyparts']
        for bp in bodyparts:
            bps_line += [bp]*3
        triagwriter.writerow(bps_line)
        triagwriter.writerow(['coords'] + ['x', 'y', 'z']*num_bodyparts)
        for iframe in range(np.shape(processed_array)[0]):
            rw = [iframe]
            for ibp in range(num_bodyparts):
                rw += [processed_array[iframe, 0, ibp],
                       processed_array[iframe, 1, ibp],
                       processed_array[iframe, 2, ibp]]
            triagwriter.writerow(rw)

    return output_csv


def make_triangulation_video(video_path, triangulated_csv_path, skeleton_config=None,
                             output_path=None, frame_range=None, view=(90, 120), figure_size=(9, 5),
                             figure_dpi=150, marker_size=5, skeleton_thickness=1,
                             frame_count=False, frame_rate=None, thrd_video_path=None,
                             thrd_video_frame_offset=0, third_video_crop_hw=None, ranges=None,
                             plot_markers=True, horizontal_subplots=True):

    '''Makes a video based on triangulated marker positions.

    Arguments:
        video_path {str} -- Full file path of video.
        triangulated_csv_path {str} -- Full file path of csv with triangulated points.
    Keyword Arguments:
        skeleton_config {str} -- Path to yaml file with both 'bodyparts' and 'skeleton' as shown in
            the example config. (default: None)
        output_path {str} -- Path to place the video. Will accept full file name.
            (default: None = same as triangulated_csv)
        frame_range {tuple or None} --  part of video and points to create a video for. If a tuple
            then indicates the start and end frame number, including both as an interval. If None
            then all frames will be used. (default: None)
        view {tuple} -- The desired (elevation, azimuth) required for the 3d plot. (default:
            (90, 90))
        figure_size {tuple} -- desired (width, height) of the figure. (default:(9, 5))
        figure_dpi {int} -- DPI of the video. (default: 150)
        marker_size {int} -- size of the markers in the 3d plot. (default: 5)
        skeleton_thickness {int} -- thickness of the connecting lines in the 3d plot. (default: 1)
        thrd_video_path {str} -- add another video from this path to the side. (default: None)
        thrd_video_frame_offset {int} -- start the added vide from this frame. (default: 0)
        third_video_crop_hw {list of 2 slices} -- crop the third video using slices.
            (default: None)
        ranges {list of 2-lists} -- overwrites xrange, yrange and zrange for the 3d plot. Individual
            elements can be None. (default: None)
        plot_markers {bool} -- plot 3D view of the markers. Having it False with no third_video_path
            set can lead to unexpected behavior. (default: True)
        horizontal_subplots {bool} -- makes subplots horizontal, otherwise vertical. (default: True)
    '''
    if skeleton_config is not None:
        with open(skeleton_config, 'r') as yaml_file:
            dic = yaml.safe_load(yaml_file)
            bp_list = dic['bodyparts']
            bp_connections = dic['skeleton']
        skeleton = True
    else:
        skeleton = False

    with open(triangulated_csv_path, 'r') as f:
        triagreader = csv.reader(f)
        l = next(triagreader)
        bodyparts = []
        for i, bp in enumerate(l):
            if (i-1)%3 == 0:
                bodyparts.append(bp)
        num_bodyparts = len(bodyparts)
        next(triagreader)
        triangulated_points = []
        for row in triagreader:
            triangulated_points.append([[] for _ in range(3)])
            for ibp in range(num_bodyparts):
                triangulated_points[-1][0].append(float(row[1+ibp*3]))
                triangulated_points[-1][1].append(float(row[2+ibp*3]))
                triangulated_points[-1][2].append(float(row[3+ibp*3]))

    triangulated_points = np.array(triangulated_points)

    cmap = matplotlib.cm.get_cmap('jet')
    color_idx = np.linspace(0, 1, num_bodyparts)
    bp_cmap = cmap(color_idx)
    # Limits in space of the markers + 10%
    margin_min = 0.7
    margin_max = 1.3
    pcntl = 0.1
    if ranges is None or ranges[0] is None:
        x_range = (np.nanpercentile(triangulated_points[:, 0, :], pcntl) * margin_min,
                   np.nanpercentile(triangulated_points[:, 0, :], 100-pcntl) * margin_max)
    else:
        x_range = ranges[0]
    if ranges is None or ranges[1] is None:
        y_range = (np.nanpercentile(triangulated_points[:, 1, :], pcntl) * margin_min,
                   np.nanpercentile(triangulated_points[:, 1, :], 100-pcntl) * margin_max)
    else:
        y_range = ranges[1]
    if ranges is None or ranges[2] is None:
        z_range = (np.nanpercentile(triangulated_points[:, 2, :], pcntl) * margin_min,
                   np.nanpercentile(triangulated_points[:, 2, :], 100-pcntl) * margin_max)
    else:
        z_range = ranges[2]

    # Inspect the video
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None: # Use the default file name
        csv_path_head = os.path.splitext(triangulated_csv_path)[0]
        video_path_body = os.path.splitext(os.path.split(video_path)[1])[0]
        output_filename = (csv_path_head + '_' +  video_path_body + '_triangulated.mp4')
    else:
        if os.path.isdir(output_path): # Check if an existing directory
            csv_path_body = os.path.splitext(os.path.split(triangulated_csv_path)[1])[0]
            video_path_body = os.path.splitext(os.path.split(video_path)[1])[0]
            output_filename = os.path.join(output_path, csv_path_body + '_' +  video_path_body +
                                           '_triangulated.mp4')
        else:
            output_filename = output_path

    output_filename = utils.iterative_filename(output_filename)
    print('File path: {}'.format(output_filename))

    # Check the frame range
    if frame_range is not None:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_range[0]) # Set the start position
        if frame_range[1] >= num_frames:
            print('Too many frames requested, the video will be truncated appropriately.\n')
            frame_range = (frame_range[0], num_frames-1)

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_range[0]) # Set the start position
        # # If the above method does not work with MPEG/FFMPEG, see
        # # @https://stackoverflow.com/questions/19404245/opencv-videocapture-set-cv-cap-prop-pos-frames-not-working
        # and try:
        # for i in range(frame_range[0]):
        #     video.read()
    else:
        frame_range = (0, num_frames-1)

    # load the third video
    if thrd_video_path is not None:
        thrd_video = cv2.VideoCapture(thrd_video_path)
        # thrd_video.set(cv2.CAP_PROP_POS_FRAMES, thrd_video_frame_offset) # Set the start position
        thrd_fe, thrd_frame = thrd_video.read()
        thrd_video_fps = int(thrd_video.get(cv2.CAP_PROP_FPS))

    # Create the figure
    fig = mpl_pp.figure(figsize=figure_size, dpi=figure_dpi)
    fw, fh = fig.get_size_inches() * fig.get_dpi()
    canvas = FigureCanvas(fig)
    # Make a new video keeping the old properties - need to know figure size first
    if frame_rate is None:
        frame_rate = fps

    if horizontal_subplots:
        xn_sbps = lambda num_subplots: 1
        yn_sbps = lambda num_subplots: num_subplots
    else:
        xn_sbps = lambda num_subplots: num_subplots
        yn_sbps = lambda num_subplots: 1


    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    output_video = cv2.VideoWriter(output_filename, fourcc, frame_rate, (int(fw), int(fh)))
    # Create the axes
    if thrd_video_path is None and plot_markers:
        num_subplots = 2
        ax_video = fig.add_subplot(xn_sbps(num_subplots), yn_sbps(num_subplots), 1)
        ax_3d = fig.add_subplot(xn_sbps(num_subplots), yn_sbps(num_subplots), 2, projection='3d')
    elif thrd_video_path is not None:
        num_subplots = 2
        ax_video = fig.add_subplot(xn_sbps(num_subplots), yn_sbps(num_subplots), 1)
        ax_third = fig.add_subplot(xn_sbps(num_subplots), yn_sbps(num_subplots), 2)
    else:
        num_subplots = 3
        ax_video = fig.add_subplot(xn_sbps(num_subplots), yn_sbps(num_subplots), 1)
        ax_3d = fig.add_subplot(xn_sbps(num_subplots), yn_sbps(num_subplots), 2, projection='3d')
        ax_third = fig.add_subplot(xn_sbps(num_subplots), yn_sbps(num_subplots), 3)

    if plot_markers:
        ax_3d.view_init(elev=view[0], azim=view[1])

    for f_idx in tqdm(range(frame_range[0], frame_range[1]+1), desc='Writing frame'):
        fe, frame = video.read() # Read the next frame
        if fe is False:
            print('Could not read the frame. Aborting and saving.')
            break

        frame_rgb = frame[..., ::-1].copy()
        # Clear axis 1
        ax_video.cla()
        ax_video.imshow(frame_rgb)
        ax_video.set_xticks([])
        ax_video.set_yticks([])
        # Clear axis 2
        if plot_markers:
            ax_3d.cla()
            ax_3d.set_xlim(x_range)
            ax_3d.set_ylim(y_range)
            ax_3d.set_zlim(z_range)
            if frame_count:
                ax_3d.set_title('Frame: ' + str(f_idx))

        # Handle video 3
        if thrd_video_path is not None:
            frame_from_start = f_idx - frame_range[0]
            thrd_frame_from_start = (thrd_video_frame_offset +
                                     int(frame_from_start/fps*thrd_video_fps))
            # tv_cf = thrd_video.get(cv2.CAP_PROP_POS_FRAMES)
            while thrd_video.get(cv2.CAP_PROP_POS_FRAMES) < thrd_frame_from_start:
                thrd_fe, thrd_frame = thrd_video.read()
            # thrd_video.set(cv2.CAP_PROP_POS_FRAMES, thrd_frame_from_start)
            # thrd_fe, thrd_frame = thrd_video.read()  # Read the next frame
            if thrd_fe is False:
                print('Could not read the third video frame. Frame in the raw {},'
                      ' frame_from_start: {} thrd_frame_from_start: {}'.format(
                          f_idx, frame_from_start, thrd_frame_from_start))
            else:
                thrd_frame_rgb = thrd_frame[..., ::-1].copy()
                if third_video_crop_hw is not None:
                    if third_video_crop_hw[0] is not None:
                        thrd_frame_rgb = thrd_frame_rgb[third_video_crop_hw[0], :]
                    if third_video_crop_hw[1] is not None:
                        thrd_frame_rgb = thrd_frame_rgb[:, third_video_crop_hw[1]]
                ax_third.cla()
                ax_third.imshow(thrd_frame_rgb)
                ax_third.set_xticks([])
                ax_third.set_yticks([])

        # Underlying skeleton
        if plot_markers and skeleton:
            for bpc in bp_connections:
                ibp1 = bp_list.index(bpc[0])
                ibp2 = bp_list.index(bpc[1])

                t_point1 = triangulated_points[f_idx, :, ibp1]
                t_point2 = triangulated_points[f_idx, :, ibp2]

                if any(np.isnan(t_point1)) or any(np.isnan(t_point1)):
                    continue
                ax_3d.plot([t_point1[0], t_point2[0]],
                           [t_point1[1], t_point2[1]],
                           [t_point1[2], t_point2[2]],
                           color='k', linewidth=skeleton_thickness)

        # Bodypart markers
        if plot_markers:
            for ibp in range(np.size(triangulated_points, 2)):
                # Markers
                ax_3d.scatter(triangulated_points[f_idx, 0, ibp],
                              triangulated_points[f_idx, 1, ibp],
                              triangulated_points[f_idx, 2, ibp],
                              color=bp_cmap[ibp, :], s=marker_size)

        # Pull matplotlib data to a variable and format for writing
        canvas.draw()
        temp_frame = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(
            int(fh), int(fw), 3)
        temp_frame = temp_frame[..., ::-1].copy()
        output_video.write(temp_frame)

    # Release objects
    mpl_pp.close(fig)
    video.release()
    output_video.release()

    print('*  Video saved to:\n\t' + output_filename)


def interactive_3d_plot(vid_path, triangulated_csv_path, skeleton_path=None, figure_size=(9, 5),
                        marker_size=5, skeleton_thickness=1):
    """Makes an interactive 3D plot with video and a slider to control the frame number.

    Arguments:
        vid_path {str} -- location of the video to observe.
        triangulated_csv_path {str} -- location of csv with triangulated points corresponding to the
            video given in vid_path.
    """
    global FIG, FIGNUM, AXS, SLIDER

    # Import the triangulated CSV
    with open(triangulated_csv_path, 'r') as f:
        triagreader = csv.reader(f)
        l = next(triagreader)
        bodyparts = []
        for i, bp in enumerate(l):
            if (i-1)%3 == 0:
                bodyparts.append(bp)
        num_bodyparts = len(bodyparts)
        next(triagreader)
        triangulated_points = []
        for row in triagreader:
            triangulated_points.append([[] for _ in range(3)])
            for ibp in range(num_bodyparts):
                triangulated_points[-1][0].append(float(row[1+ibp*3]))
                triangulated_points[-1][1].append(float(row[2+ibp*3]))
                triangulated_points[-1][2].append(float(row[3+ibp*3]))
    triangulated_points = np.array(triangulated_points)

    # Get the video
    if not os.path.exists(vid_path):
        raise ValueError('Provided video file does not exist.')
    video = cv2.VideoCapture(vid_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if skeleton_path is not None:
        with open(skeleton_path, 'r') as yaml_file:
            dic = yaml.safe_load(yaml_file)
            bp_list = dic['bodyparts']
            bp_connections = dic['skeleton']
        skeleton = True
    else:
        skeleton = False

    # Check the number of frames vs number of rows in csv
    if num_frames != np.shape(triangulated_points)[0]:
        warnings.warn('Number of frames in video ({}) and rows in CSV ({}) are not equal.'
                      ' Check that the paths given are correct.'.format(
            num_frames, np.shape(triangulated_points)[0]))

    # Initalize the plots
    cmap = matplotlib.cm.get_cmap('jet')
    color_idx = np.linspace(0, 1, num_bodyparts)
    bp_cmap = cmap(color_idx)
    # Limits in space of the markers + 10%
    margin = 1.3
    pcntl = 2
    x_range = (np.nanpercentile(triangulated_points[:, 0, :], pcntl) * margin,
               np.nanpercentile(triangulated_points[:, 0, :], 100-pcntl) * margin)
    y_range = (np.nanpercentile(triangulated_points[:, 1, :], pcntl) * margin,
               np.nanpercentile(triangulated_points[:, 1, :], 100-pcntl) * margin)
    z_range = (np.nanpercentile(triangulated_points[:, 2, :], pcntl) * margin,
               np.nanpercentile(triangulated_points[:, 2, :], 100-pcntl) * margin)

    FIG = mpl_pp.figure(figsize=figure_size)
    FIGNUM = mpl_pp.gcf().number
    AXS = []
    AXS.append(FIG.add_subplot(1, 2, 1))
    AXS.append(FIG.add_subplot(1, 2, 2, projection='3d'))
    AXS[1].view_init(elev=90, azim=90)

    def update(iframe):
        iframe = int(iframe)
        mpl_pp.figure(FIGNUM)
        video.set(cv2.CAP_PROP_POS_FRAMES, iframe) # Set the frame to get
        fe, frame = video.read() # Read the frame
        if fe:
            frame_rgb = frame[..., ::-1].copy()

            AXS[0].cla()
            AXS[0].imshow(frame_rgb)

            AXS[1].cla()
            AXS[1].set_xlim(x_range)
            AXS[1].set_ylim(y_range)
            AXS[1].set_zlim(z_range)

            # Underlying skeleton
            if skeleton:
                for bpc in bp_connections:
                    ibp1 = bp_list.index(bpc[0])
                    ibp2 = bp_list.index(bpc[1])

                    t_point1 = triangulated_points[iframe, :, ibp1]
                    t_point2 = triangulated_points[iframe, :, ibp2]

                    if any(np.isnan(t_point1)) or any(np.isnan(t_point1)):
                        continue
                    AXS[1].plot([t_point1[0], t_point2[0]],
                                [t_point1[1], t_point2[1]],
                                [t_point1[2], t_point2[2]],
                                color='k', linewidth=skeleton_thickness)

            # Bodypart markers
            for ibp in range(np.size(triangulated_points, 2)):
                # Markers
                AXS[1].scatter(triangulated_points[iframe, 0, ibp],
                               triangulated_points[iframe, 1, ibp],
                               triangulated_points[iframe, 2, ibp],
                               color=bp_cmap[ibp, :], s=marker_size)


    def arrow_key_image_control(event):
        if event.key == 'left' and SLIDER.val > 0:
            SLIDER.set_val(SLIDER.val - 1)
        elif event.key == 'right' and SLIDER.val < num_frames-1:
            SLIDER.set_val(SLIDER.val + 1)
        else:
            pass

    update(0)

    axcolor = 'lightgoldenrodyellow'
    ax_ind = mpl_pp.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    SLIDER = mpl_pp.Slider(ax_ind, 'Frame', 0, num_frames-1, valinit=0, valstep=1, valfmt='%u')
    SLIDER.on_changed(update)

    cid = FIG.canvas.mpl_connect('key_press_event', arrow_key_image_control)

    mpl_pp.show()


def display_confidence_statistics(ncams_config, labeled_csv_paths, iteration=None, file_prefix='',
                                  num_bins=100, title='', per_camera=True, pooled=True,
                                  only_bodyparts=None, skip_bodyparts=[], threshold=0,
                                  pool_across_trials=False):
    '''Triangulates points from multiple cameras and exports them into a csv.

    Call matplotlib.pyplot.show() or pylab.show() to display the figures after the function runs.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). This function uses following keys:
            serials {list of numbers} -- list of camera serials.
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
        labeled_csv_paths {list of str} -- locations of csv's with marked points.
    Keyword Arguments:
        iteration {int} -- look for csv's with this iteration number. (default: {None})
        file_prefix {string} -- prefix of the csv file to search for in the folder. (default: {''})
        num_bins {int} -- number of bins in each histogram. (default: {100})
        title {str or list of str} -- a string to append to the figure title. (default: {''}).
        per_camera {bool} -- make per_camera plots. (default: {True})
        pooled {bool} -- make pooled plots. (default: {True})
        only_bodyparts {list} -- only displays information about these bodyparts. If None, display
            all. (default: {None})
        skip_bodyparts {list} -- ignores these bodyparts. Runs after the bodyparts from
            only_bodyparts list are selected. (default: {[]})
        threshold {float} -- threshold for confidence for which the stats to display. If 0, no
            relevant stats are calcualted. (default: {0})
        pool_across_trials {bool} -- data from individual trials will be concatenated.
            (default: {False})
    '''
    cam_serials = ncams_config['serials']

    # assuming bodyparts are the same
    ic_confidencess = []
    bodyparts_old = None
    for labeled_csv_path in labeled_csv_paths:
        # Check if the source CSV path exists
        if not os.path.exists(labeled_csv_path):
            raise ValueError('Provided path for CSVs does not exist.')

        # Get data files
        list_of_csvs = get_list_labelled_csvs(ncams_config, labeled_csv_path,
                                              file_prefix=file_prefix, iteration=iteration)
        list_of_csvs.sort()

        # Load them
        bodyparts, _, _, ic_confidences = load_labelled_csvs(
            list_of_csvs, threshold=0, filtering=False)
        num_cameras = len(list_of_csvs)

        # Remove unneeded bodyparts
        for ibp in reversed(range(len(bodyparts))):
            if ((only_bodyparts is not None and bodyparts[ibp] not in only_bodyparts) or
                    bodyparts[ibp] in skip_bodyparts):
                del bodyparts[ibp]
                for icam in range(num_cameras):
                    ic_confidences[icam] = np.delete(ic_confidences[icam], ibp, axis=1)

        if bodyparts_old is None:
            bodyparts_old = bodyparts
        elif not all([bpo == bp for bpo, bp in zip(bodyparts_old, bodyparts)]):
            raise ValueError('Loaded files have different bodyparts.')

        num_bodyparts = len(bodyparts)
        if pool_across_trials:
            if len(ic_confidencess) == 0:
                ic_confidencess.append(ic_confidences)
            else:
                # append to the end
                for icam in range(num_cameras):
                    ic_confidencess[0][icam] = np.concatenate(
                        (ic_confidencess[0][icam], ic_confidences[icam]), axis=0)
        else:
            ic_confidencess.append(ic_confidences)

    bins = np.linspace(0, 1, num_bins)
    xn_sbps = int(math.ceil(math.sqrt(num_bodyparts)))
    yn_sbps = int(math.ceil(num_bodyparts/xn_sbps))

    if per_camera:
        for itrial, ic_confidences in enumerate(ic_confidencess):
            mpl_pp.figure()
            if isinstance(title, str):
                mpl_pp.suptitle('Per camera colorcoded. ' + title)
            else:
                mpl_pp.suptitle('Per camera colorcoded. ' + title[itrial])
            for ibp in range(num_bodyparts):
                if ibp == 0:
                    ax1 = mpl_pp.subplot(xn_sbps, yn_sbps, ibp+1)
                else:
                    mpl_pp.subplot(xn_sbps, yn_sbps, ibp+1, sharey=ax1)
                for icam in range(num_cameras):
                    mpl_pp.hist(ic_confidences[icam][:,ibp], bins=bins, alpha=0.4, color='k')
                mpl_pp.xlim([0, 1])
                mpl_pp.yscale('log')
                mpl_pp.title(bodyparts[ibp])
                if ibp % yn_sbps == 0:
                    mpl_pp.ylabel('Number of time points.')
                if math.floor(ibp / yn_sbps) == xn_sbps - 1:
                    mpl_pp.xlabel('Estimated confidence, nu')
                else:
                    mpl_pp.xticks([])

    if pooled:
        if threshold > 0:  # doing stats
            above_theshold = [0]*num_bodyparts
            num_points = [0]*num_bodyparts
        mpl_pp.figure()
        mpl_pp.suptitle('Pooled from all cameras. {}'.format(title))
        # make subplots axes
        axs = []
        for ibp in range(num_bodyparts):
            if ibp == 0:
                axs.append(mpl_pp.subplot(xn_sbps, yn_sbps, ibp+1))
            else:
                axs.append(mpl_pp.subplot(xn_sbps, yn_sbps, ibp+1, sharey=axs[0]))
        # plot
        for itrial, ic_confidences in enumerate(ic_confidencess):
            for ibp in range(num_bodyparts):
                y = []
                for icam in range(num_cameras):
                    y += np.squeeze(ic_confidences[icam][:,ibp]).tolist()
                if len(ic_confidencess) == 1:
                    axs[ibp].hist(y, bins=bins, color='k')
                else:
                    axs[ibp].hist(y, bins=bins, alpha=0.4)
                axs[ibp].set_xlim([0, 1])
                axs[ibp].set_yscale('log')
                axs[ibp].set_title(bodyparts[ibp])
                if ibp % yn_sbps == 0:
                    axs[ibp].set_ylabel('Number of time points.')
                if math.floor(ibp / yn_sbps) == xn_sbps - 1:
                    axs[ibp].set_xlabel('Estimated confidence, nu')
                else:
                    axs[ibp].set_xticks([])

                if threshold > 0:
                    above_theshold[ibp] += sum([int(v > threshold) for v in y])
                    num_points[ibp] += len(y)
        if threshold > 0:
            ylims = axs[0].get_ylim()
            for ibp in range(num_bodyparts):
                axs[ibp].vlines(threshold, ylims[0], ylims[1], colors='r', linestyles='dashed')
                axs[ibp].annotate(
                    '{:.2f}% above threshold'.format(above_theshold[ibp]/num_points[ibp]*100),
                    (threshold, 0.85),  xycoords='axes fraction')
            axs[0].set_ylim(ylims)


'''
The below is the core for triangulation procedures.
decomp_matrix = np.empty((np.sum(cams_detecting)*2, 4))
for decomp_idx, cam in enumerate(cam_idx):
    point_mat = cam_image_points[:, cam]
    projection_mat = projection_matrices[cam]

    temp_decomp = np.vstack([
        [point_mat[0] * projection_mat[2, :] - projection_mat[0, :]],
        [point_mat[1] * projection_mat[2, :] - projection_mat[1, :]]])

    decomp_matrix[decomp_idx*2:decomp_idx*2 + 2, :] = temp_decomp

Q = decomp_matrix.T.dot(decomp_matrix)
u, _, _ = np.linalg.svd(Q)
u = u[:, -1, np.newaxis]
u_euclid = (u/u[-1, :])[0:-1, :]
triangulated_points[iframe, :, bodypart] = np.transpose(u_euclid)
'''
