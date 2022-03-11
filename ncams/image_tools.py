#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019-2020 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Functions for working with images and making videos.

For more details on the camera data structures and dicts, see help(ncams.camera_tools).
"""
import os
import numpy
import cv2

from tqdm import tqdm
import moviepy # A very easy way of using FFMPEG
import moviepy.editor
# current release version of moviepy (1.0.3) redirects all warnings to logging. fix:
from packaging import version
if version.parse(moviepy.__version__) <= version.parse("1.0.3"):
    import logging
    logging.captureWarnings(False)

import matplotlib.pyplot as mpl_pp


def undistort_video(video_filename, camera_calib_dict, output_filename=None):
    '''Undistorts every frame in a video based on camera calibration parameters.

    Iterates through every frame in a video and undistorts it as appropriate based on the given
    intrinsics and distortion coefficients.

    Arguments:
        video_filename {string} -- filename of the video.
        camera_calib_dict {dict} -- see help(ncams.camera_tools). Sould have following keys:
            distortion_coefficients {np.array} -- distortion coefficients for the camera.
            camera_matrix {np.array} -- camera calibration matrifor the camera.
    Keyword Arguments:
        output_filename {string} -- output video filename. (default: {same folder as video_filename
            with '_undistorted' added})
    '''
    if output_filename is None:
        # replaces the extention with '_undistorted.mp4'
        output_filename = video_filename[:video_filename.rfind('.')] + '_undistorted.mp4'

    # Inspect the video - maintain all properties of the video
    video = cv2.VideoCapture(video_filename)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')

    # Create the new video
    video_undistorted = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    frame_exists, frame = video.read()
    while frame_exists:
        # Undistort and write
        undistorted_frame = undistort_image(frame, camera_calib_dict)
        video_undistorted.write(undistorted_frame)

        frame_exists, frame = video.read() # Read the next frame if it exists

    video.release()
    video_undistorted.release()


def undistort_image(image, camera_calib_dict):
    '''Undistorts an individual frame or image based on the camera calibration.

    Undistorts an individual frame or image based on the camera matrix and distortion coefficients.

    Arguments:
        image {np.array XxYxcolor} --  image array.
        camera_calib_dict {dict} -- see help(ncams.camera_tools). Sould have following keys:
            distortion_coefficients {np.array} -- distortion coefficients for the camera.
            camera_matrix {np.array} -- camera calibration matrifor the camera.
    Output:
        undistorted_image {np.array X Y Color} --  undistorted image array.
    '''
    undistorted_image = cv2.undistort(
        image, camera_calib_dict['camera_matrix'], camera_calib_dict['distortion_coefficients'],
        None, camera_calib_dict['camera_matrix'])

    return undistorted_image


def images_to_video(image_filenames, video_filename, fps=30, output_folder=None, logger='bar'):
    '''Combines a list of images into a video.

    Arguments:
        image_filenames {list} -- list of strings, these must either be complete paths or
            filenames depending on cwd.
        video_filename {string} -- output file name. e.g. 'new_video.mp4'. If 'output_folder' is
            provided, then 'video_filename' should not be a full path.
    Keyword Arguments:
        fps {integer} -- frame rate. (default: 30)
        output_folder {string} -- where to save the video. (default: {export to the current working
            directory})
        logger {logger} -- either "bar" for progress bar or None or any Proglog logger. (default:
            {'bar', same as moviepy.video.VideoClip.write_videofile default})
    '''
    if output_folder is None:
        output_folder = os.path.split(image_filenames[0])[0]

    output_name = os.path.join(output_folder, video_filename)
    output_filetype = os.path.splitext(output_name)[1]

    clip = moviepy.editor.ImageSequenceClip(image_filenames, fps=fps)

    if output_filetype == '.gif':
        clip.write_gif(output_name, fps=fps, logger=logger)
    else:
        clip.write_videofile(output_name, fps=fps, logger=logger)


def video_to_images(list_of_videos, output_directory=None, output_format='jpeg',
                    frame_range=None, dlc_filename_format=False):
    '''Exports a video to a series of images.

    Arguments:
        list_of_videos {list} -- list of strings, these must either be complete paths or
            filenames depending on cwd.
    Keyword Arguments:
        output_directory {string} -- where to save the images. (default: {export to the current working
            directory as a subfolder with the same name as the video})
        frame_range {2-list of numbers or None} -- frame range to export from the file. If a tuple
            then indicates the start and end frame number, including both as an interval. If None
            then all frames will be used. If frame_range[1] is None, continue until the last frame.
            (default: None)
    '''
    if isinstance(list_of_videos, str):
        list_of_videos = [list_of_videos]

    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        od = output_directory
    else:
        od = os.getcwd()

    for vid_path in list_of_videos:
        # Get the video
        video = cv2.VideoCapture(vid_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check the frame range
        if frame_range is not None:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_range[0]) # Set the start position
            if frame_range[1] >= num_frames:
                print('Too many frames requested, the video will be truncated appropriately.\n')
                frame_range = (frame_range[0], num_frames-1)

            video.set(cv2.CAP_PROP_POS_FRAMES, frame_range[0]) # Set the start position
        else:
            frame_range = (0, num_frames-1)

        # Check the name
        vid_name = os.path.splitext(os.path.split(vid_path)[1])[0]
        image_dir = os.path.join(od, vid_name)
        print('Exporting images to: {}'.format(image_dir))
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        for f_idx in tqdm(range(frame_range[0], frame_range[1]+1), desc='Exporting frame'):
            frame_exists, frame = video.read() # Read the next frame if it exists
            if not frame_exists:
                break

            if dlc_filename_format:
                fname = os.path.join(image_dir, 'img{:04d}.{}'.format(f_idx, output_format))
            else:
                fname = os.path.join(image_dir, vid_name + str(f_idx) + '.' + output_format)
            cv2.imwrite(fname, frame)

        video.release()


def video_to_timeseries(video_filename, fig_filename, num_images=5, figure_size=(9, 5),
                        figure_dpi=150, crop_hw=None):
    '''Exports a number of images from a video to generally show the video in stills.

    Arguments:
        video_filename {str} -- filename of the video.
        fig_filename {str} -- output figure filename

    Keyword Arguments:
        num_images {number} -- number of frames in the timeseries (default: {5})
        figure_size {tuple} -- desired (width, height) of the figure. (default:(9, 5))
        figure_dpi {int} -- DPI of the video. (default: 150)
        crop_hw {list of 2 slices} -- crop the video frames using slices. (default: None)
    '''
    # make a figure
    fig = mpl_pp.figure(figsize=figure_size, dpi=figure_dpi)

    # Get the video
    video = cv2.VideoCapture(video_filename)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    f_idxs = [int(i) for i in numpy.linspace(0, num_frames-1, num_images)]

    for f_i, f_idx in enumerate(f_idxs):
        video.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        frame_exists, frame = video.read() # Read the next frame if it exists
        if not frame_exists:
            break
        t = video.get(cv2.CAP_PROP_POS_MSEC)
        frame_rgb = frame[..., ::-1].copy()
        if crop_hw is not None:
            if crop_hw[0] is not None:
                frame_rgb = frame_rgb[crop_hw[0], :]
            if crop_hw[1] is not None:
                frame_rgb = frame_rgb[:, crop_hw[1]]

        ax = fig.add_subplot(1, num_images, f_i+1)
        ax.imshow(frame_rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('{}: {} ms'.format(f_idx, int(t)))
        ax.patch.set_visible(False)

    fig.patch.set_visible(False)
    mpl_pp.tight_layout(pad=0, h_pad=0, w_pad=0)

    fig.savefig(fig_filename, transparent=True)

    video.release()
