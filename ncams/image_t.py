#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Please see AUTHORS for contributors.
https://github.com/CMGreenspon/NCams/blob/master/README.md
Licensed under the Apache License, Version 2.0
"""
import os

import moviepy.editor # A very easy way of using FFMPEG
import cv2


def undistort_video(file_path, camera_matrix, distortion_coefficients, crop_and_resize=False,
                    output_path=None):
    '''Undistorts every frame in a video based on camera calibration parameters.

    Iterates through every frame in a video and undistorts it as appropriate based on the given
    intrinsics and distortion coefficients.

    Arguments:
        file_path {string} --  string to video path.
        camera_matrix {np.array 3x3} --  camera intrinsics
        distortion_coefficients {np.array 1x5} --  distortion coefficients
    Keyword Arguments:
        crop_and_resize {boolean} -- if true the optimal undistorted region (from
            getOptimalNewCameraMatrix) will be selected for the output. (default: {False})
        output_path {string} -- output path of video. (default: {same folder as file_path})
    '''
    # Format the output name
    base_path, filename = os.path.split(file_path)
    filename = os.path.splitext(filename)[0]
    if output_path is None:
        output_name = os.path.join(base_path, filename) + '_undistorted.mp4'
    else:
        output_name = os.path.join(output_path, filename) + '_undistorted.mp4'

    # Inspect the video - maintain all properties of the video
    video = cv2.VideoCapture(file_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'X264')

    # Create the new video
    video_undistorted = cv2.VideoWriter(output_name, fourcc, fps, (w, h))

    frame_exists = True
    while frame_exists:
        frame_exists, frame = video.read() # Read the next frame if it exists
        if frame_exists: # Undistort and write
            undistorted_frame = undistort_image(frame, camera_matrix, distortion_coefficients,
                                                crop_and_resize)
            video_undistorted.write(undistorted_frame)

    video.release()
    video_undistorted.release()


def undistort_image(image, camera_matrix, distortion_coefficients, crop_and_resize=False):
    '''Undistorts an individual frame or image based on the camera calibration.

    Undistorts an individual frame or image based on the camera matrix and distortion coefficients.

    Arguments:
        image {np.array XxYxcolor} --  image array.
        camera_matrix {np.array 3x3} --  camera intrinsics
        distortion_coefficients {np.array 1x5} --  distortion coefficients
    Keyword Arguments:
        crop_and_resize {boolean} -- if true the optimal undistorted region (from
            getOptimalNewCameraMatrix) will be selected for the output. (default: {False})
    Output:
        undistorted_image {np.array XxYxcolor} --  undistorted image array.

    '''
    '''
    Output:
        undistorted image array: (x,y,color)
    '''
    h, w = image.shape[:2]
    new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients,
                                                     (w, h), 1, (w, h))
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None,
                                      new_cam_mat)

    if crop_and_resize:
        x, y, w2, h2 = roi
        undistorted_image = undistorted_image[y:y+h2, x:x+w2]
        undistorted_image = cv2.resize(undistorted_image, (w, h))

    return undistorted_image


def images_to_video(list_of_image_names, filename, fps=30, output_folder=None):
    '''Combines a list of images into a video.

    Arguments:
        list_of_image_names {list} -- list of strings, these must either be complete paths or
            filenames depending on cwd.
        filename {string} -- output file name. e.g. 'new_video.mp4'.
    Keyword Arguments:
        fps {integer} -- frame rate. (default: 30)
        output_folder {string} -- where to save the video. (default: {export to the current working
            directory})
    '''
    if output_folder is None:
        output_folder = os.getcwd()

    output_name = os.path.join(output_folder, filename)

    clip = moviepy.editor.ImageSequenceClip(list_of_image_names, fps=fps)
    clip.write_videofile(output_name, fps=fps)
