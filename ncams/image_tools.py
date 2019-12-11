#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Functions for working with images and making videos.

For more details on the camera data structures and dicts, see help(ncams.camera_tools).
"""
import os

from moviepy import editor # A very easy way of using FFMPEG
import cv2


def undistort_video(video_filename, camera_calib_dict, crop_and_resize=False, output_filename=None):
    '''Undistorts every frame in a video based on camera calibration parameters.

    Iterates through every frame in a video and undistorts it as appropriate based on the given
    intrinsics and distortion coefficients.

    Arguments:
        video_filename {string} -- filename of the video.
        camera_calib_dict {dict} -- see help(ncams.camera_tools). Sould have following keys:
            distortion_coefficients {np.array} -- distortion coefficients for the camera.
            camera_matrix {np.array} -- camera calibration matrifor the camera.
    Keyword Arguments:
        crop_and_resize {boolean} -- if true the optimal undistorted region (from
            getOptimalNewCameraMatrix) will be selected for the output. (default: {False})
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
    fourcc = cv2.VideoWriter_fourcc(*'X264')

    # Create the new video
    video_undistorted = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    frame_exists, frame = video.read()
    while frame_exists:
        # Undistort and write
        undistorted_frame = undistort_image(
            frame, camera_calib_dict, crop_and_resize=crop_and_resize)
        video_undistorted.write(undistorted_frame)

        frame_exists, frame = video.read() # Read the next frame if it exists

    video.release()
    video_undistorted.release()


def undistort_image(image, camera_calib_dict, crop_and_resize=False):
    '''Undistorts an individual frame or image based on the camera calibration.

    Undistorts an individual frame or image based on the camera matrix and distortion coefficients.

    Arguments:
        image {np.array XxYxcolor} --  image array.
        camera_calib_dict {dict} -- see help(ncams.camera_tools). Sould have following keys:
            distortion_coefficients {np.array} -- distortion coefficients for the camera.
            camera_matrix {np.array} -- camera calibration matrifor the camera.
    Keyword Arguments:
        crop_and_resize {boolean} -- if true the optimal undistorted region (from
            getOptimalNewCameraMatrix) will be selected for the output. (default: {False})
    Output:
        undistorted_image {np.array X Y Color} --  undistorted image array.
    '''
    h, w = image.shape[:2]
    new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(
        camera_calib_dict['camera_matrix'], camera_calib_dict['distortion_coefficients'],
        (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(
        image, camera_calib_dict['camera_matrix'], camera_calib_dict['distortion_coefficients'],
        None, new_cam_mat)

    if crop_and_resize:
        x, y, w2, h2 = roi
        undistorted_image = undistorted_image[y:y+h2, x:x+w2]
        undistorted_image = cv2.resize(undistorted_image, (w, h))

    return undistorted_image


def images_to_video(image_filenames, video_filename, fps=30, output_folder=None):
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
    '''
    if output_folder is None:
        output_folder = os.getcwd()

    output_name = os.path.join(output_folder, video_filename)

    clip = editor.ImageSequenceClip(image_filenames, fps=fps)
    clip.write_videofile(output_name, fps=fps)
