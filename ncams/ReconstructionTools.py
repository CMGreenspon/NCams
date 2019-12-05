'''
author(s): Charles M. Greenspon
           Anton Sobinov
lab: Sliman Bensmaia
'''
import os
import csv
import shutil
import multiprocessing
import functools

import glob
import numpy as np
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as mpl_pp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import pylab

import cv2

import CameraTools
import ImageTools


FIG = None
FIGNUM = None
AXS = None
SLIDER = None


def triangulate(cam_dicts, camera_config, session_path, labeled_video_path,
                threshold=0.9, images_3d_path=None, method='full_rank', best_pair_n=2,
                num_frames_limit=None, output_csv=None):
    '''Triangulates points from multiple cameras

    Input:
        labeled_video_path: locations of csv's with marked points
        threshold: only points with confidence above the threshold will be used for triangulation.
            (default: 0.9)
        images_3d_path: where you would like the 3d video and images to be stored.
            (default: os.path.join(session_path, 'rec_3d'))
        method: method for triangulation.
            full_rank: uses all available cameras
            best_pair: uses best 'best_pair_n' cameras to locate the point
            (default: 'full_rank')
        best_pair_n: how many cameras to use when best_pair method is used. (default: 2)
        num_frames_limit: limit to the number of frames used for analysis. Useful for testing. If
            None, then all frames will be analyzed. (default: None)
        output_csv: file to save the triangulated points into.
            (default: os.path.join(images_3d_path, 'triangulated_points_<method>.csv'))
    '''
    cam_serials = camera_config['camera_serials']
    (camera_matrices, distortion_coefficients, _, world_locations, world_orientations
     ) = CameraTools.load_camera_config(camera_config)

    # Get files
    list_of_csvs = []
    for cam_serial in cam_serials:
        list_of_csvs += glob.glob(os.path.join(
            labeled_video_path, cam_dicts[cam_serial]['name']+'*.csv'))
    if not len(list_of_csvs) == len(cam_serials):
        print('Detected {} csvs while was provided with {} serials. Quitting'.format(
            len(list_of_csvs), len(cam_serials)))
        return

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
    output_coordinates, output_coordinates_filtered = [], []
    for icam in range(num_cameras):
        # Get the optimal camera matrix
        optimal_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrices[icam],
            distortion_coefficients[icam],
            (camera_config['image_size'][1], camera_config['image_size'][0]),
            1,
            (camera_config['image_size'][1], camera_config['image_size'][0]))

        output_array = np.empty((num_frames, 2, num_bodyparts))
        filtered_output_array = np.empty((num_frames, 2, num_bodyparts))
        # The filtered one needs NaN points so we know which to ignore
        filtered_output_array.fill(np.nan)

        # Get the sufficiently confident values for each bodypart
        for bodypart in range(num_bodyparts):
            # Get the distorted points
            distorted_points = image_coordinates[icam][:, :, bodypart]
            # Undistort them
            undistorted_points = cv2.undistortPoints(
                distorted_points, camera_matrices[icam],
                distortion_coefficients[icam], P=optimal_matrix)
            # Add to the unfiltered array
            output_array[:, :, bodypart] = undistorted_points[:, 0, :]
            # Get threshold filter
            bp_thresh = thresholds[icam][:, bodypart].astype(np.float32) > threshold
            thresh_idx = np.where(bp_thresh == 1)[0]
            # Put them into the output array
            for idx in thresh_idx:
                filtered_output_array[idx, :, bodypart] = undistorted_points[idx, 0, :]

        output_coordinates.append(output_array)
        output_coordinates_filtered.append(filtered_output_array)

    # Triangulation
    # Make the projection matrices
    projection_matrices = []
    for icam in range(num_cameras):
        projection_matrices.append(CameraTools.make_projection_matrix(
            camera_matrices[icam], world_orientations[icam], world_locations[icam]))

    # Triangulate the points
    triangulated_points = np.empty((num_frames, 3, num_bodyparts))
    triangulated_points.fill(np.nan)

    for iframe in range(num_frames):
        for bodypart in range(num_bodyparts):
            # Get points for each camera
            cam_image_points = np.empty((2, num_cameras))
            cam_image_points.fill(np.nan)
            if method == 'full_rank' or (method == 'best_pair' and num_cameras <= best_pair_n):
                for icam in range(num_cameras):
                    cam_image_points[:, icam] = output_coordinates_filtered[icam][iframe, :, bodypart]
            elif method == 'best_pair':
                # decorate-sort-undecorate sort to find the icams for the highest likelihood
                best_likelh = [b[0] for b in sorted(
                    zip(range(num_cameras),
                        [thresholds[icam][iframe, bodypart].astype(np.float64)
                         for icam in range(num_cameras)]),
                    key=lambda x: x[1], reverse=True)][:best_pair_n]
                for icam in [icam for icam in range(num_cameras) if icam in best_likelh]:
                    cam_image_points[:, icam] = output_coordinates_filtered[icam][iframe, :, bodypart]

            # Check how many cameras detected the bodypart in that frame
            cams_detecting = ~np.isnan(cam_image_points[0, :])
            cam_idx = np.where(cams_detecting)[0]
            if np.sum(cams_detecting) < 2:
                continue

            # Perform the triangulation - adapted from DeepFly3D
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

    # Make a side by side video
    if images_3d_path is None:
        images_3d_path = os.path.join(session_path, 'rec_3d')
    if not os.path.isdir(images_3d_path):
        os.mkdir(images_3d_path)

    if output_csv is None:
        output_csv = os.path.join(images_3d_path, 'triangulated_points.csv')
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


def make_triangulation_videos(camera_config, cam_dicts, session_path, triangulated_csv,
                              images_3d_path=None, overwrite_temp=False, fps=30,
                              num_frames_limit=None, parallel=None):
    """
    Input:
        images_3d_path: where you would like the 3d video and images to be stored.
            (default: os.path.join(session_path, 'rec_3d'))
        fps: for making movies. (default: 30)
        overwrite_temp: flag for automatically overwrite folder for holding images.
            (default: False)
        output_csv: file to save the triangulated points into.
            (default: os.path.join(images_3d_path, 'triangulated_points.csv'))
        num_frames_limit: limit to the number of frames used for analysis. Useful for testing. If
            None, then all frames will be analyzed. (default: None)
        parallel: parallelize the image creation. If integer, create that many processes. If None,
            do not parallelize
    """
    matplotlib.use('Agg')

    cam_serials = camera_config['camera_serials']
    (camera_matrices, distortion_coefficients, _, world_locations, world_orientations
     ) = CameraTools.load_camera_config(camera_config)

    with open(triangulated_csv, 'r') as f:
        triagreader = csv.reader(f)
        l = next(triagreader)
        bodyparts = []
        for i, bp in enumerate(l):
            if (i-1)%3 == 0:
                bodyparts.append(bp)
        num_bodyparts = len(bodyparts)
        next(triagreader)
        triangulated_points = []
        num_frames = 0
        for row in triagreader:
            triangulated_points.append([[] for _ in range(3)])
            for ibp in range(num_bodyparts):
                triangulated_points[-1][0].append(float(row[1+ibp*3]))
                triangulated_points[-1][1].append(float(row[2+ibp*3]))
                triangulated_points[-1][2].append(float(row[3+ibp*3]))
            num_frames += 1
            if num_frames_limit is not None and num_frames >= num_frames_limit:
                break
    triangulated_points = np.array(triangulated_points)

    for cam_serial in cam_serials:
        print('Making images for {}'.format(cam_dicts[cam_serial]['name']))
        image_list = ImageTools.get_image_list(path=os.path.join(
            session_path, cam_dicts[cam_serial]['name']))
        if not os.path.isdir(images_3d_path):
            os.mkdir(images_3d_path)
        output_path = os.path.join(images_3d_path,
                                   cam_dicts[cam_serial]['name'])
        if os.path.isdir(output_path):
            if overwrite_temp:
                shutil.rmtree(output_path, ignore_errors=True)
                os.mkdir(output_path)
            else:
                uinput_string = (
                    "Directory {} used for image temporary hold exists. Would you like to wipe it?"
                    "(Yes/No/Abort/Help')."
                    "\nHaving residual files in the folder can lead to erroneus videos.\n".format(
                        output_path))
                while True:
                    user_input = input(uinput_string).lower()
                    if user_input in ('no', 'n'):
                        print('- Proceeding...')
                        break
                    elif user_input in ('yes', 'y'):
                        print('- Wiping folder...')
                        shutil.rmtree(output_path, ignore_errors=True)
                        os.mkdir(output_path)
                        break
                    elif user_input == 'abort':
                        return
                    elif user_input == 'help':
                        print('- Yes: delete the folder and all its contents, and make a new one.\n'
                              '- No: use the folder as is.\n'
                              '- Abort: exit the function without returning anything.\n')
                    else:
                        print("Invalid response given.\n")
        else:
            os.mkdir(output_path)

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

        if parallel is None or parallel < 2:
            fig = mpl_pp.figure(figsize=(9, 5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.view_init(elev=90, azim=90)
            # Create the figure
            for frame in tqdm(range(num_frames)):
                ax1.cla()
                image_path = image_list[frame]
                ax1.imshow(mpl_pp.imread(image_path))

                ax2.cla()
                ax2.set_xlim(x_range)
                ax2.set_ylim(y_range)
                ax2.set_zlim(z_range)

                for ibp in range(np.size(triangulated_points, 2)):
                    ax2.scatter(triangulated_points[frame, 0, ibp],
                                triangulated_points[frame, 1, ibp],
                                triangulated_points[frame, 2, ibp],
                                color=bp_cmap[ibp, :])

                mpl_pp.savefig(os.path.join(output_path, 'frame' + str(frame)))
        else:
            print('Using parallel pool to create images')
            mip = functools.partial(make_image,
                                    ranges=[x_range, y_range, z_range],
                                    output_path=output_path, bp_cmap=bp_cmap)
            with multiprocessing.Pool(parallel) as pool:
                pool.map(mip, zip(range(num_frames), image_list[:num_frames], triangulated_points))

        # Make a video of it
        print('Making a video for {}'.format(cam_dicts[cam_serial]['name']))
        output_image_list = ImageTools.get_image_list(path=output_path)
        ImageTools.images_to_video(
            output_image_list,
            os.path.join(images_3d_path, cam_dicts[cam_serial]['name'] + '.mp4'), fps=fps)


def make_image(args, ranges=None, output_path=None, bp_cmap=None):
    iframe, image_path, triangulated_points = args
    if output_path is None:
        output_path = os.getcwd()

    fig = mpl_pp.figure(figsize=(9, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    # Create the figure
    ax1.imshow(mpl_pp.imread(image_path))

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.view_init(elev=90, azim=90)
    if ranges is not None:
        ax2.set_xlim(ranges[0])
        ax2.set_ylim(ranges[1])
        ax2.set_zlim(ranges[2])

    if bp_cmap is None:
        for ibp in range(np.size(triangulated_points, 1)):
            ax2.scatter(triangulated_points[0, ibp],
                        triangulated_points[1, ibp],
                        triangulated_points[2, ibp])
    else:
        for ibp in range(np.size(triangulated_points, 1)):
            ax2.scatter(triangulated_points[0, ibp],
                        triangulated_points[1, ibp],
                        triangulated_points[2, ibp],
                        color=bp_cmap[ibp, :])

    mpl_pp.savefig(os.path.join(output_path, 'frame' + str(iframe)))
    mpl_pp.close(fig)


def interactive_3d_plot(cam_serial, camera_config, cam_dicts, session_path, triangulated_csv,
                        num_frames_limit=None):
    """
    Input:
        images_3d_path: where you would like the 3d video and images to be stored.
            (default: os.path.join(session_path, 'rec_3d'))
        fps: for making movies. (default: 30)
        num_frames_limit: limit to the number of frames used for analysis. Useful for testing. If
            None, then all frames will be analyzed. (default: None)
    """
    (camera_matrices, distortion_coefficients, _, world_locations, world_orientations
     ) = CameraTools.load_camera_config(camera_config)

    with open(triangulated_csv, 'r') as f:
        triagreader = csv.reader(f)
        l = next(triagreader)
        bodyparts = []
        for i, bp in enumerate(l):
            if (i-1)%3 == 0:
                bodyparts.append(bp)
        num_bodyparts = len(bodyparts)
        next(triagreader)
        triangulated_points = []
        num_frames = 0
        for row in triagreader:
            triangulated_points.append([[] for _ in range(3)])
            for ibp in range(num_bodyparts):
                triangulated_points[-1][0].append(float(row[1+ibp*3]))
                triangulated_points[-1][1].append(float(row[2+ibp*3]))
                triangulated_points[-1][2].append(float(row[3+ibp*3]))
            num_frames += 1
            if num_frames_limit is not None and num_frames >= num_frames_limit:
                break
    triangulated_points = np.array(triangulated_points)

    image_list = ImageTools.get_image_list(path=os.path.join(
        session_path, cam_dicts[cam_serial]['name']))

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

    global FIG, FIGNUM, AXS, SLIDER

    FIG = mpl_pp.figure(figsize=(9, 5))
    FIGNUM = mpl_pp.gcf().number
    AXS = []
    AXS.append(FIG.add_subplot(1, 2, 1))
    AXS.append(FIG.add_subplot(1, 2, 2, projection='3d'))
    AXS[1].view_init(elev=90, azim=90)

    def update(iframe):
        mpl_pp.figure(FIGNUM)
        AXS[0].cla()
        image_path = image_list[int(iframe)]
        AXS[0].imshow(mpl_pp.imread(image_path))

        AXS[1].cla()
        AXS[1].set_xlim(x_range)
        AXS[1].set_ylim(y_range)
        AXS[1].set_zlim(z_range)

        for ibp in range(np.size(triangulated_points, 2)):
            AXS[1].scatter(triangulated_points[int(iframe), 0, ibp],
                           triangulated_points[int(iframe), 1, ibp],
                           triangulated_points[int(iframe), 2, ibp],
                           color=bp_cmap[ibp, :])
    update(0)

    axcolor = 'lightgoldenrodyellow'
    ax_ind = mpl_pp.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    SLIDER = mpl_pp.Slider(ax_ind, 'Ind', 0, num_frames-1, valinit=0)
    SLIDER.on_changed(update)

    mpl_pp.show()

