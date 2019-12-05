'''
author(s): Charles M. Greenspon
           Anton Sobinov
lab: Sliman Bensmaia
'''
import os
import re
import glob

import moviepy.editor # A very easy way of using FFMPEG
import cv2


def undistort_video(file_path, camera_matrix, distortion_coefficients, crop_and_resize=False,
                    output_path=None):
    '''
    Iterates through every frame in a video and undistorts it as appropriate based on the given
    intrinsics and distortion coefficients.

    If crop_and_resize is true the optimal undistorted region (from getOptimalNewCameraMatrix) will
    be selected for the output.
    Inputs:
        file_path: string to video path.
        camera_matrix: 3x3 array of camera intrinsics
        distortion_coefficients: 1x5 array of distortion coefficients
        crop_and_resize (optional): boolean
        output_path (optional): output path (string) of video if different
    '''
    # Format the output name
    base_path, filename = os.path.split(file_path)
    filename, filetype = os.path.splitext(filename)
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
    '''
    Undistorts an individual frame or image based on the camera matrix and distortion coefficients.
    If crop_and_resize is true the optimal undistorted region (from getOptimalNewCameraMatrix) will
    be selected for the output.
    Inputs:
        image: image array (y,x,color)
        camera_matrix: 3x3 array of camera intrinsics
        distortion_coefficients: 1x5 array of distortion coefficients
        crop_and_resize (optiona): boolean
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


def get_image_list(path=None, sort=True, img_file_extensions=None):
    '''Returns a list of all image filenames.

    Files with extensions .jpg, .jpeg, .png, .bmp are considered images.
    Searches shell style for <path>/*.<file extension>

    Keyword Arguments:
        path {string} -- directory to explore. (default: current directory)
        sort {bool} -- alphanumeric sort the output list (default: {True})
        img_file_extensions {list} -- file extensions to return, with or
            without the dot (default: {('jpg', 'jpeg', 'png', 'bmp')})

    Output:
        list of strings: all images contained in cd or path.
    '''
    if img_file_extensions is None:
        img_file_extensions = ('jpg', 'jpeg', 'png', 'bmp')
    else:
        img_file_extensions = [ifx.strip('.') for ifx in img_file_extensions]

    return get_file_list(img_file_extensions, path=path, sort=sort)


def get_file_list(file_extensions, path=None, sort=True):
    '''Returns a list of all filenames with a specific extension.

    Files with extensions .jpg, .jpeg, .png, .bmp are considered images.
    Searches shell style for <path>/*.<file extension>

    Keyword Arguments:
        path {string} -- directory to explore. (default: current directory)
        sort {bool} -- alphanumeric sort the output list (default: {True})
        file_extensions {list} -- file extensions to return, with or
            without the dot. If None or empty, returns all files with extensions

    Output:
        list of strings: all files w/ ext contained in cd or path.
    '''
    if path is None:
        path = os.getcwd()

    if file_extensions is None or len(file_extensions) == 0:
        file_extensions = ('*', )
    else:
        file_extensions = [ifx.strip('.') for ifx in file_extensions]

    image_list = []
    for img_file_extension in file_extensions:
        image_list += glob.glob(os.path.join(path, '*.' + img_file_extension))

    if sort:
        image_list = alpha_numeric_sort(image_list)

    return image_list


def images_to_video(list_of_image_names, filename, fps=30, output_folder=None):
    '''
    Accepts a sorted list and combines all of them into a single video.
    If no output_folder is given it will export to the current working directory.
    Inputs:
        list_of_image_names: list of strings, these must either be complete paths or filenames
            depending on cwd.
        filename: (string) output file name 'new_video.mp4'
        fps (optional): integer.
        output_folder (optional): string to folder.
    '''
    if output_folder is None:
        output_folder = os.getcwd()

    output_name = os.path.join(output_folder, filename)

    clip = moviepy.editor.ImageSequenceClip(list_of_image_names, fps=fps)
    clip.write_videofile(output_name, fps=fps)


def alpha_numeric_sort(list_of_strings):
    '''
    Literally just a generic AN sorter for a list of strings.
    Useful for sorting out very high framerate images.
    Useful for sorting by framerate, because '11'<'9', but 11>9
    Useful when recording for more than 999.9999 seconds (because then the generic sorted does not
    work).
    Input:
        list of strings
    Output:
        sorted list of strings
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(list_of_strings, key=alphanum_key)
