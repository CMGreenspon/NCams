#!python3
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

Recording from FLIR cameras.

For more details on the camera data structures and dicts, see help(ncams.camera_tools).
"""
import os
import threading
import logging
import multiprocessing

import numpy as np
import matplotlib.pyplot as mpl_pp

import cv2
import PySpin
import tqdm


CAMERA_NAME = 'cam{cam_serial}'
IMAGE_FILENAME = '{time}_image{frame}.jpeg'


# Logging
logging.basicConfig(level=logging.INFO)


# System init/denit & test
def get_system():
    '''Initializes communication with the camera system.

    Output:
        system {PySpin.System instance} -- PySpin System
        cam_serials {list of numbers} -- list of camera serials. Order is not essential.
        cam_dicts {dict of camera_dict's} -- keys are serials, values are 'camera_dict', see
            help(ncams.camera_tools).
    '''
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    print('{} cameras detected.'.format(len(cam_list)))

    cam_dicts = {}
    cam_serials = []
    for icam, cam in enumerate(cam_list):
        serial = int(cam.GetUniqueID())
        cam_serials.append(serial)
        cam_dicts[serial] = {
            'name': CAMERA_NAME.format(cam_serial=serial),
            'serial': serial,
            'obj': cam}

    cam_serials = sorted(cam_serials)
    for icam, serial in enumerate(cam_serials):
        print('- Camera #{} serial {} name {}'.format(
            icam+1, serial, cam_dicts[serial]['name']))

    return system, cam_serials, cam_dicts


def import_system_into_camera_config(camera_config, add_new_cameras=True):
    '''Puts camera objects of the connected cameras into the camera_config.

    Useful when restoring a recording session. Use this instead of recreating camera_config.

    Arguments:
        camera_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
    Keyword Arguments:
        add_new_cameras {bool} -- if new cameras are detected, they will be added to config.
            otherwise they will be skipped. (default: {True})

    Output:
        system {PySpin.System instance} -- PySpin System
        camera_config {dict} -- same as input with added camera objects for recording.
    '''
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    print('{} cameras detected.'.format(len(cam_list)))

    for icam, cam in enumerate(cam_list):
        serial = int(cam.GetUniqueID())
        if serial in camera_config['dicts'].keys():
            cam_dicts[serial]['obj'] = cam
        else:
            print('New camera detected!')
            if add_new_cameras:
                print('Adding co camera_config.')
                camera_config['serials'].append(serial)
                cam_dicts[serial] = {
                    'name': CAMERA_NAME.format(serial),
                    'serial': serial,
                    'obj': cam}

    return (system, camera_config)


def release_system(system, cam_list):
    '''Initializes communication with the camera system.

    Arguments:
        system {PySpin.System instance} -- PySpin System
        cam_list {list of camera objects} --
    '''
    if system.IsInUse() is False:
        print('No system to release.')
        return

    for cam in cam_list:
        if cam.IsStreaming():
            cam.EndAcquisision()
        if cam.IsInitialized():
            cam.DeInit()

    system.ReleaseInstance()
    print('System released.')


def get_image(cam, output=False, output_path=None):
    '''Captures a single image.

    Arguments:
        cam {PySpin Camera} -- camera from which to capture the image.
    Keyword Arguments:
        output {bool} -- save the image to disk? (default: {False})
        output_path {string} -- where to save the image if 'output' is True. (default: {current
            working directory})
    Output:
        rgb_array {pixel array XxYx3} -- image matrix
    '''
    set_cam_settings(cam, default=True)

    if not cam.IsInitialized():
        cam.Init()

    cam.BeginAcquisition()
    image = cam.GetNextImage()

    if output:
        filename = 'cam_test_image.jpeg'
        if output_path is not None:
            filename = os.path.join(output_path, filename)

        image.Save(filename)

    image_converted = image.Convert(
        PySpin.PixelFormat_BGR8, PySpin.DIRECTIONAL_FILTER)

    width = image_converted.GetWidth()
    height = image_converted.GetHeight()
    pixel_array = image_converted.GetData()
    pixel_array = pixel_array.reshape(height, width, 3)
    rgb_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB)

    image.Release()
    cam.EndAcquisition()
    return rgb_array


def test_system_capture(camera_config):
    '''Tries to get a picture (get_image) from every camera in the system.

    Arguments:
        camera_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
    '''
    serials = camera_config['serials']
    num_cameras = len(serials)

    if num_cameras == 1:
        rgb_array = get_image(camera_config['dicts'][serials[0]]['obj'])

        fig, ax = mpl_pp.subplots()
        ax.imshow(rgb_array)

    else:
        num_vert_plots = int(np.ceil(np.sqrt(num_cameras)))
        num_horz_plots = int(np.ceil(num_cameras/num_vert_plots))
        fig, axs = mpl_pp.subplots(num_vert_plots, num_horz_plots, squeeze=False)

        for icam, serial in enumerate(serials):
            rgb_array = get_image(camera_config['dicts'][serial]['obj'])

            vert_ind = int(np.floor(icam / num_horz_plots))
            horz_ind = icam - num_horz_plots * vert_ind

            axs[vert_ind, horz_ind].imshow(rgb_array)
            axs[vert_ind, horz_ind].set_xticks([])
            axs[vert_ind, horz_ind].set_yticks([])


# Camera settings
def reset_cams(cam_objs):
    '''Resets a camera or a list of cameras to default.

    Arguments:
        cam_objs {PySpin.CameraPtr or a list of PySpin.CameraPtr} -- camera(s) to reset.
    '''
    if isinstance(cam_objs, PySpin.CameraPtr):  # A single camera object
        cam_objs = [cam_objs]

    for cam in cam_objs:
        if not cam.IsInitialized():
            cam.Init()
        if cam.IsStreaming():
            cam.EndAcquisition()

        cam.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        cam.UserSetLoad()


def set_cam_settings(cam, default=False, frame_rate=None, exposure_time=None, gain=None,
                     trigger_mode=None, acquisition_mode=None, pixel_format=None):
    '''Function for setting most useful camera settings.

    Arguments:
        cam {PySpin.CameraPtr} -- a camera object that has been initalized.
    Keyword Arguments:
        default {bool} -- set parameters to default values. Overrides all other keywords provided.
            (defult: {False})
        frame_rate {number} -- desired frames per second. (default: {30})
        exposure_time {int} -- exposure time in microseconds. (default: {2000})
        gain {number or 'auto'} -- 'auto' for auto gain, number for manual control. (default: {13})
        trigger_mode {boolean} -- True means that it will not take images unless a trigger is
            sent from the reference camera. (default: {False})
        acquisition_mode {0, 1 or 2} -- mode of acquisition:
            0: continuous - essentially a video
            1: singleframe - a predefined number of images are taken
            2: multiframe - a single frame occurs before acquisition ends
            (default: {0})
        pixel_format {PySpin.PixelFormat_[opt]} -- pixel format. Note that not all options are
            available to all cameras. (default: {PySpin.PixelFormat_BayerRG8})

    '''
    # Just in case the camera is running when we try to change the settings
    if not cam.IsInitialized():
        cam.Init()
    if cam.IsStreaming():
        cam.EndAcquisition()

    # To avoid overwriting when not called the default parameter is instead
    # available which will overwrite everything
    if default:
        if any(param is not None for param in [
               frame_rate, exposure_time, gain, trigger_mode, acquisition_mode, pixel_format]):
            print('Warning: both the default flag and parameters have been passed.\n'
                  'Default parameters will be used.')
        frame_rate = 30
        exposure_time = 2000
        gain = 13
        trigger_mode = False
        acquisition_mode = 0
        pixel_format = PySpin.PixelFormat_BayerRG8

    # Frame rate
    if frame_rate is not None:
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRate.SetValue(frame_rate)

    # Exposure time
    if exposure_time is not None:
        if exposure_time == 'auto':
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_On)
        else:
            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            cam.ExposureTime.SetValue(exposure_time)

    # Gain
    if gain is not None:
        if gain == 'auto':
            cam.GainAuto.SetValue(2)
        else:
            cam.GainAuto.SetValue(0)
            gain_max = cam.Gain.GetMax()
            if gain_max < gain:
                gain = gain_max
                print('Maximum gain exceeded; gain set to maximum: {}'.format(round(gain, 3)))

            cam.Gain.SetValue(gain)

    # Trigger Mode
    if trigger_mode is not None:
        if trigger_mode:
            cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        else:
            cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

    # Acquisition mode
    if (acquisition_mode is not None and not cam.AcquisitionMode.GetValue() == acquisition_mode):
        cam.AcquisitionMode.SetValue(acquisition_mode)

    # Pixel format
    if (pixel_format is not None and not cam.PixelFormat.GetValue() == pixel_format):
        cam.PixelFormat.SetValue(pixel_format)


# Synchronized capture
def init_sync_settings(camera_config, frame_rate=30, num_images=None):
    '''Initializes all cameras for sequence capture.

    Arguments:
        camera_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
            reference_camera_serial {number} -- serial number of the reference camera.
    Keyword Arguments:
        frame_rate {number} -- fps to set the cameras to. (default: {30})
        num_images {number or None} -- number of images to set to capture. If None, captures
            indefinitely. (default: {None})
    '''
    cam_dicts = camera_config['dicts']
    reference_serial = camera_config['reference_camera_serial']

    # Settings for each camera
    nodemap = []
    for cam_dict in cam_dicts.values():
        # Check cam states
        if not cam_dict['obj'].IsInitialized():
            cam_dict['obj'].Init()
        if cam_dict['obj'].IsStreaming():
            cam_dict['obj'].EndAcquisition()

        # Set buffers for all cameras
        nodemap.append(cam_dict['obj'].GetTLStreamNodeMap())
        # Oldest first
        PySpin.CEnumerationPtr(
            nodemap[-1].GetNode('StreamBufferHandlingMode')).SetIntValue(0)

        # Acquisition mode
        if num_images is None:
            cam_dict['obj'].AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        elif isinstance(num_images, int):
            cam_dict['obj'].AcquisitionMode.SetValue(PySpin.AcquisitionMode_MultiFrame)
            cam_dict['obj'].AcquisitionFrameCount.SetValue(num_images)

    # Primary cam settings
    # Triggering
    cam_dicts[reference_serial]['obj'].LineSelector.SetValue(PySpin.LineSelector_Line2)
    cam_dicts[reference_serial]['obj'].LineMode.SetValue(PySpin.LineMode_Output)
    cam_dicts[reference_serial]['obj'].TriggerSource.SetValue(PySpin.TriggerSource_Software)
    # Frame rate
    cam_dicts[reference_serial]['obj'].AcquisitionFrameRateEnable.SetValue(True)
    cam_dicts[reference_serial]['obj'].AcquisitionFrameRate.SetValue(frame_rate)
    cam_dicts[reference_serial]['obj'].TriggerMode.SetValue(PySpin.TriggerMode_On)
    # Enable acquisition - won't start until trigger mode is turned off
    cam_dicts[reference_serial]['obj'].BeginAcquisition()

    # Secondary cam settings
    for cam_dict in [c for ic, c in cam_dicts.items()
                     if not ic == reference_serial]:
        cam_dict['obj'].TriggerSource.SetValue(PySpin.TriggerSource_Line3)
        cam_dict['obj'].TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
        cam_dict['obj'].TriggerMode.SetValue(PySpin.TriggerMode_On)
        # As is being controlled by the trigger
        cam_dict['obj'].AcquisitionFrameRateEnable.SetValue(False)
        # This should only prime the cameras:
        cam_dict['obj'].BeginAcquisition()


def synced_capture_sequence(camera_config, num_images, output_folder=None, separate_folders=True):
    '''Captures images from all cameras in synchronized manner.

    Allows up to and including 50 fps capturing.

    Arguments:
        camera_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
            reference_camera_serial {number} -- serial number of the reference camera.
        num_images {number} -- number of images to capture.
    Keyword Arguments:
        output_folder {string} -- where to store the images (default: {os.getcwd()}).
        separate_folders {} -- create individual directories for each camera (default: {True}).
    '''
    cam_dicts = camera_config['dicts']
    reference_serial = camera_config['reference_camera_serial']

    # Sort out file storage
    if output_folder is None:
        output_folder = os.getcwd()

    cam_paths = {}
    if separate_folders:
        for cam_serial, cam_dict in cam_dicts.items():
            cam_paths[cam_serial] = os.path.join(output_folder, cam_dict['name'])
            if not os.path.isdir(cam_paths[cam_serial]):
                os.mkdir(cam_paths[cam_serial])

    # Check if cams are ready to go
    for cam_dict in cam_dicts.values():
        if (not cam_dict['obj'].IsInitialized() or not cam_dict['obj'].IsStreaming()):
            print('Cameras not setup for synchronized capture.')
            return

    # Hit it
    init_time = {}
    thread_list = []
    cam_dicts[reference_serial]['obj'].TriggerMode.SetValue(PySpin.TriggerMode_Off)
    # We want to offload from images in order across cameras to reduce buffer load equally
    for i_image in tqdm.tqdm(range(num_images)):
        for cam_serial, cam_dict in cam_dicts.items():
            image = cam_dict['obj'].GetNextImage(500)
            f_id = image.GetFrameID()
            if i_image == 0:
                init_time[cam_serial] = image.GetTimeStamp()

            if separate_folders:
                dest_folder = cam_paths[cam_serial]
            else:
                dest_folder = output_folder

            thread_list.append(threading.Thread(
                target=save_image_thread,
                args=(image, dest_folder, f_id, init_time[cam_serial], cam_dict['name'])))
            thread_list[-1].start()

    for thread in thread_list:
        thread.join()

    for cam_dict in cam_dicts.values():
        cam_dict['obj'].EndAcquisition()


def synced_capture_sequence_p(camera_config, num_images, output_folder=None, separate_folders=True):
    '''Captures images from all cameras in synchronized manner, employs multiprocessing.
    Not Implemented - keeps breaking in children processes.

    Essentially the same as synced_capture_sequence, but tries to speed up saving by having a
    parallel process for saving images to the drive.

    Arguments:
        camera_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
            reference_camera_serial {number} -- serial number of the reference camera.
        num_images {number} -- number of images to capture.
    Keyword Arguments:
        output_folder {string} -- where to store the images (default: {os.getcwd()}).
        separate_folders {} -- create individual directories for each camera (default: {True}).
    '''
    raise NotImplementedError
    cam_dicts = camera_config['dicts']
    reference_serial = camera_config['reference_camera_serial']

    # Sort out file storage
    if output_folder is None:
        output_folder = os.getcwd()

    cam_paths = {}
    if separate_folders:
        for cam_serial, cam_dict in cam_dicts.items():
            cam_paths[cam_serial] = os.path.join(output_folder, cam_dict['name'])
            if not os.path.isdir(cam_paths[cam_serial]):
                os.mkdir(cam_paths[cam_serial])

    # Check if cams are ready to go
    for cam_dict in cam_dicts.values():
        if (not cam_dict['obj'].IsInitialized() or not cam_dict['obj'].IsStreaming()):
            print('Cameras not setup for synchronized capture.')
            return

    # Hit it
    image_saver_processes = {}
    image_saver_qs = {}
    for cam_serial, cam_dict in cam_dicts.items():
        if separate_folders:
            dest_folder = cam_paths[cam_serial]
        else:
            dest_folder = output_folder

        image_saver_qs[cam_serial] = multiprocessing.SimpleQueue()
        image_saver_processes[cam_serial] = multiprocessing.Process(
            target=_image_saver_process,
            args=(image_saver_qs[cam_serial], dest_folder, cam_dict['name']))
        image_saver_processes[cam_serial].start()

    cam_dicts[reference_serial]['obj'].TriggerMode.SetValue(
        PySpin.TriggerMode_Off)
    # We want to offload from images in order across cameras to reduce buffer
    # load equally
    for _ in tqdm.tqdm(range(num_images)):
        for cam_serial, cam_dict in cam_dicts.items():
            image = cam_dict['obj'].GetNextImage(500)

            image_dat = image.GetNDArray()
            offsetX = image.GetXOffset()
            offsetY = image.GetYOffset()
            width = image.GetWidth()
            height = image.GetHeight()
            pixelFormat = image.GetPixelFormat()

            f_id = image.GetFrameID()
            time_stamp = image.GetTimeStamp()

            image_saver_qs[cam_serial].put((image_dat, offsetX, offsetY, width, height,
                                            pixelFormat, f_id, time_stamp))
            image.Release()

    # for cam_serial in cam_dict.keys():
    #     image_saver_qs[cam_serial].put(None)
    #     image_saver_processes[cam_serial].join()

    for cam_dict in cam_dicts.values():
        cam_dict['obj'].EndAcquisition()


def _image_saver_process(q, dest_folder, cam_name):
    '''A process dedicated to saving images from a camera.

    Accompanying function to synced_capture_sequence_p.

    Arguments:
        q {multiprocessing.SimpleQueue} -- process reads from this queue the image data.
        dest_folder {string} -- where to save the images.
        cam_name {string} -- camera name, needed for identification in stdout
    '''
    done = False
    thread_list = []
    init_time = None
    print('Camera {} saver process initialized'.format(cam_name))
    while not done:
        # get the image data
        while q.empty():
            pass
        val = q.get()

        if val is None:  # stop if done
            done = True
            continue

        # unpack the data
        (image_dat, offsetX, offsetY, width, height, pixelFormat, f_id, time_stamp) = val

        if init_time is None:
            init_time = time_stamp

        print('Camera {} got data. time_stamp: {}'.format(cam_name, time_stamp-init_time))

        image = PySpin.Image.Create(width, height, offsetX, offsetY, pixelFormat, image_dat)

        thread_list.append(threading.Thread(
            target=save_image_function,
            args=(image, dest_folder, f_id, init_time, time_stamp, cam_name)))

    # cleanup
    for thread in thread_list:
        thread.join()


def synced_capture_sequence_ram(camera_config, num_images, output_folder=None,
                                separate_folders=True):
    '''Captures images from all cameras in synchronized manner, stores in RAM, then dumps.

    Essentially the same as synced_capture_sequence, but speeds up acquisition of images by storing
    them in RAM. Right now it is accumulating 150 fps, but it used up almost 9 gigs for 6.66 sec
    recording and scales linearly, so it would allow at max 45 sec recording on 64 GB RAM PC.
    Experimentally could record 50 fps for 60 seconds, which took ~42 GB RAM.

    Arguments:
        camera_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            dicts {dict of 'camera_dict's} -- keys are serials, values are 'camera_dict'.
            reference_camera_serial {number} -- serial number of the reference camera.
        num_images {number} -- number of images to capture.
    Keyword Arguments:
        output_folder {string} -- where to store the images (default: {os.getcwd()}).
        separate_folders {} -- create individual directories for each camera (default: {True}).
    '''
    cam_dicts = camera_config['dicts']
    reference_serial = camera_config['reference_camera_serial']

    # Sort out file storage
    if output_folder is None:
        output_folder = os.getcwd()

    cam_paths = {}
    if separate_folders:
        for cam_serial, cam_dict in cam_dicts.items():
            cam_paths[cam_serial] = os.path.join(output_folder,
                                                 cam_dict['name'])
            if not os.path.isdir(cam_paths[cam_serial]):
                os.mkdir(cam_paths[cam_serial])

    # Check if cams are ready to go
    for cam_dict in cam_dicts.values():
        if (not cam_dict['obj'].IsInitialized() or
                not cam_dict['obj'].IsStreaming()):
            print('Cameras not setup for synchronized capture.')
            return

    # Hit it
    image_lists = {}
    thread_list = []
    for cam_serial in cam_dicts.keys():
        image_lists[cam_serial] = []
    cam_dicts[reference_serial]['obj'].TriggerMode.SetValue(PySpin.TriggerMode_Off)

    # We want to offload from images in order across cameras to reduce buffer load equally
    print('capturing...')
    for i_image in tqdm.tqdm(range(num_images)):
        for cam_serial, cam_dict in cam_dicts.items():
            image = cam_dict['obj'].GetNextImage(500)
            image_dat = image.GetNDArray()
            offsetX = image.GetXOffset()
            offsetY = image.GetYOffset()
            width = image.GetWidth()
            height = image.GetHeight()
            pixelFormat = image.GetPixelFormat()

            f_id = image.GetFrameID()
            time_stamp = image.GetTimeStamp()

            image_lists[cam_serial].append((
                image_dat, offsetX, offsetY, width, height, pixelFormat, f_id,
                time_stamp))
            image.Release()

    print('Saving...')
    for i_image in tqdm.tqdm(range(num_images)):
        for cam_serial, cam_dict in cam_dicts.items():
            (image_dat, offsetX, offsetY, width, height, pixelFormat, f_id, time_stamp
             ) = image_lists[cam_serial][i_image]
            image = PySpin.Image.Create(width, height, offsetX, offsetY, pixelFormat, image_dat)

            if separate_folders:
                dest_folder = cam_paths[cam_serial]
            else:
                dest_folder = output_folder

            thread_list.append(threading.Thread(
                target=save_image_function2,
                args=((width, height, offsetX, offsetY, pixelFormat, image_dat),
                      dest_folder, f_id, image_lists[cam_serial][0][7], time_stamp,
                      cam_dict['name'])))
            thread_list[-1].start()

    for thread in thread_list:
        thread.join()
    del image_lists

    for cam_dict in cam_dicts.values():
        cam_dict['obj'].EndAcquisition()


# Single camera capture
def camera_thread_capture_sequence(cam_dict, num_images, dest_folder):
    '''Captures a number of images from a single camera.

    Was intended to be used in a separate process for each camera, but did not speed up the
    acquisition.

    Arguments:
        cam_dict {camera_dict} -- see help(ncams.camera_tools).
        num_images {number} -- number of images to capture.
        dest_folder {string} -- where to save the images.
    '''
    thread_list = []
    for i_image in range(num_images):
        image = cam_dict['obj'].GetNextImage(500)
        f_id = image.GetFrameID()
        if i_image == 0:
            print('Camera {} started'.format(cam_dict['name']))
            init_time = image.GetTimeStamp()

        thread_list.append(threading.Thread(
            target=save_image_thread,
            args=(image, dest_folder, f_id, init_time, cam_dict['name'])))
        thread_list[-1].start()

    for thread in thread_list:
        thread.join()


def save_image_thread(input_image, output_path, frame_num, init_time, file_prefix=None):
    '''Saves an image to drive and releases variables.

    Arguments:
        input_image {PySpin ImagePtr} -- image to be saved.
        output_path {string} -- where to save the image.
        frame_num {int} -- number of the frame.
        init_time {int} -- time when the first image on the camera was captured.

    Keyword Arguments:
        file_prefix {string} -- prefix for a file, e.g. camera name (default: {None})
    '''
    time_stamp = input_image.GetTimeStamp()
    delta_time = '{:08.4f}'.format((time_stamp - init_time) / 1E9)
    filename = IMAGE_FILENAME.format(time=delta_time, frame=frame_num)
    if file_prefix is not None:
        filename = file_prefix + '_' + filename
    filename = os.path.join(output_path, filename)

    input_image.Save(filename)
    input_image.Release()


def save_image_function(input_image, output_path, frame_num, init_time, time_stamp,
                        file_prefix=None):
    '''Saves an image to drive and DOES NOT release variables.

    Useful when releasing an image can hurt another process working with it, potential use in
    multiprocessing.

    Arguments:
        input_image {PySpin ImagePtr} -- image to be saved.
        output_path {string} -- where to save the image.
        frame_num {int} -- number of the frame.
        init_time {int} -- time when the first image on the camera was captured.

    Keyword Arguments:
        file_prefix {string} -- prefix for a file, e.g. camera name (default: {None})
    '''
    delta_time = '{:08.4f}'.format((time_stamp - init_time) / 1E9)
    filename = IMAGE_FILENAME.format(time=delta_time, frame=frame_num)
    if file_prefix is not None:
        filename = file_prefix + '_' + filename
    filename = os.path.join(output_path, filename)

    input_image.Save(filename)


def save_image_function2(input_image_vars, output_path, frame_num, init_time, time_stamp,
                         file_prefix=None):
    '''Saves an image to drive same as save_image_function, but accepts image variables.

    Useful when releasing an image can hurt another process working with it, potential use in
    multiprocessing.

    Arguments:
        input_image_vars {list} -- variables defining an image to be saved:
            width
            height
            offsetX
            offsetY
            pixelFormat
            image_dat
        output_path {string} -- where to save the image.
        frame_num {int} -- number of the frame.
        init_time {int} -- time when the first image on the camera was captured.

    Keyword Arguments:
        file_prefix {string} -- prefix for a file, e.g. camera name (default: {None})
    '''
    (width, height, offsetX, offsetY, pixelFormat, image_dat) = input_image_vars
    image = PySpin.Image.Create(width, height, offsetX, offsetY, pixelFormat, image_dat)

    delta_time = '{:08.4f}'.format((time_stamp - init_time) / 1E9)
    filename = IMAGE_FILENAME.format(time=delta_time, frame=frame_num)
    if file_prefix is not None:
        filename = file_prefix + '_' + filename
    filename = os.path.join(output_path, filename)

    image.Save(filename)
    image.Release()


def capture_sequence(cam, num_images=50, output_path=None, file_prefix=''):
    '''Captures a sequence of images from a single camera.

    Arguments:
        cam {PySpin.CameraPtr} -- camera to capture from.

    Keyword Arguments:
        num_images {number} -- number of images to capture (default: {50})
        output_path {str} -- where to save the images (default: {None})
        file_prefix {str} -- prefix for the filename, e.g. camera name (default: {''})
    '''
    # Check input
    if isinstance(cam, list):
        raise ValueError('capture_sequence only accepts "CameraPtr" objects, not lists of them.')

    if output_path is None:
        output_path = os.getcwd()

    set_cam_settings(cam, acquisition_mode=2)
    cam.AcquisitionFrameCount.SetValue(num_images)

    # Ensure buffer mode is suitable
    nodemap = cam.GetTLStreamNodeMap()
    node_bufferhandling_mode = PySpin.CEnumerationPtr(nodemap.GetNode('StreamBufferHandlingMode'))
    node_oldestfirst_mode = node_bufferhandling_mode.GetEntryByName('OldestFirst').GetValue()
    node_bufferhandling_mode.SetIntValue(node_oldestfirst_mode)

    # Begin threaded collection
    thread_list = []
    cam.BeginAcquisition()
    for idx in range(num_images):
        image = cam.GetNextImage()
        if idx == 0:
            init_time = image.GetTimeStamp()

        thread_list.append(threading.Thread(
            target=save_image_thread, args=(image, output_path, idx, init_time, file_prefix)))
        thread_list[-1].start()
        thread_list[-1].join()

    cam.EndAcquisition()


def capture_sequence_gui(cam_obj, num_images=50, output_path=None, file_prefix=''):
    '''Captures a sequence of images from a single camera and prints into matplotlib.

    Arguments:
        cam {PySpin.CameraPtr} -- camera to capture from.

    Keyword Arguments:
        num_images {number} -- number of images to capture (default: {50})
        output_path {str} -- where to save the images (default: {None})
        file_prefix {str} -- prefix for the filename, e.g. camera name (default: {''})
    '''
    # Check input
    if isinstance(cam_obj, list):
        raise ValueError('capture_sequence only accepts "CameraPtr" objects, not lists of them.')

    if output_path is None:
        output_path = os.getcwd()

    # Set parameters
    continue_recording = True
    early_stopping = False
    if not cam_obj.IsInitialized():
        cam_obj.Init()

    max_fr = cam_obj.AcquisitionFrameRate.GetValue()
    set_cam_settings(cam_obj, acquisition_mode=0, frame_rate=max_fr)

    if isinstance(num_images, int):
        early_stopping = True
    else:
        print('Press "CTRL-C" to terminate the viewer.')

    # Set the buffer handling
    nodemap = cam_obj.GetTLStreamNodeMap()
    node_bufferhandling_mode = PySpin.CEnumerationPtr(nodemap.GetNode('StreamBufferHandlingMode'))
    node_newestonly_mode = node_bufferhandling_mode.GetEntryByName('NewestOnly').GetValue()
    node_bufferhandling_mode.SetIntValue(node_newestonly_mode)
    print('Please note that framerate control does not work in this mode.')

    # Get ready
    mpl_pp.figure('Image viewer')
    thread_list = []
    idx = 0
    # Set
    cam_obj.BeginAcquisition()
    # Go
    try:
        while continue_recording:
            image = cam_obj.GetNextImage()
            if idx == 0:
                init_time = image.GetTimeStamp()
            # Save images as threads
            if output_path is not None:
                thread_list.append(threading.Thread(
                    target=save_image_thread,
                    args=(image, output_path, idx, init_time, file_prefix)))
                thread_list[-1].start()
                thread_list[-1].join()

            # Convert to RGB for MPL
            image_converted = image.Convert(PySpin.PixelFormat_BGR8, PySpin.DIRECTIONAL_FILTER)
            width = image_converted.GetWidth()
            height = image_converted.GetHeight()
            pixel_array = image_converted.GetData()
            pixel_array = pixel_array.reshape(height, width, 3)
            rgb_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB)
            # Show it
            mpl_pp.imshow(rgb_array)
            mpl_pp.title('Image ' + str(idx+1) + '/' + str(num_images))
            mpl_pp.pause(0.001)
            mpl_pp.clf()

            # Frame limit termination
            idx += 1
            if early_stopping and idx >= num_images:
                mpl_pp.close()
                continue_recording = False
                cam_obj.EndAcquisition()

     # If early stopping is desired (the keyboard library does not run on linux)
    except KeyboardInterrupt:
        mpl_pp.close()
        cam_obj.EndAcquisition()
        node_oldestfirst_mode = node_bufferhandling_mode.GetEntryByName('OldestFirst').GetValue()
        node_bufferhandling_mode.SetIntValue(node_oldestfirst_mode)
