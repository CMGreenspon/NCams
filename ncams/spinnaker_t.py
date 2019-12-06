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
import threading
import logging
import datetime
import time
import multiprocessing
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as mpl_pp

import cv2
import PySpin
import tqdm


CAMERA_NAME = 'cam{cam_serial}'


#%% Logging
logging.basicConfig(level=logging.INFO)


# System init/denit & test
def get_system():
    '''CHANGED'''
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    print('{} cameras detected.'.format(len(cam_list)))

    cam_dicts = {}
    cam_serials = []
    for icam, cam in enumerate(cam_list):
        serial = int(cam.GetUniqueID())
        cam_serials.append(serial)
        cam_dicts[serial] = {
            'name': CAMERA_NAME.format(cam_serial=cam.GetUniqueID()),
            'obj': cam}

        print('- Camera #{} serial {} name {}'.format(
            icam+1, serial, cam_dicts[serial]['name']))

    return system, cam_list, cam_serials, cam_dicts


def release_system(system, cam_list):
    '''CHANGED'''
    if system.IsInUse() is False:
        print('No system to release.')
        return

    for cam in cam_list:
        if cam.IsStreaming():
            cam.EndAcquisision()
        if cam.IsInitialized():
            cam.DeInit()

    cam_list.Clear()
    system.ReleaseInstance()
    print('System released.')


def get_image(cam, output=False, output_path=None):
    '''CHANGED and moved out of test_system_capture'''
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
        image.Release()
        cam.EndAcquisition()
    else:
        image_converted = image.Convert(
            PySpin.PixelFormat_BGR8, PySpin.DIRECTIONAL_FILTER)

        width = image_converted.GetWidth()
        height = image_converted.GetHeight()
        pixel_array = image_converted.GetData()
        pixel_array = pixel_array.reshape(height, width, 3)
        rgb_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2RGB)

        cam.EndAcquisition()
        return rgb_array


# Just tries to get a picture (get_image) from every camera in the system
def test_system_capture(cam_list):
    '''CHANGED'''
    num_cameras = len(cam_list)

    if num_cameras == 1:
        rgb_array = get_image(cam_list[0], output=False, output_path=None)

        fig, ax = mpl_pp.subplots()
        ax.imshow(rgb_array)

    else:
        num_vert_plots = int(np.ceil(np.sqrt(num_cameras)))
        num_horz_plots = int(np.ceil(num_cameras/num_vert_plots))
        fig, axs = mpl_pp.subplots(num_horz_plots, num_vert_plots,
                                   squeeze=False)
        vert_ind, horz_ind = 0, 0

        for cam in cam_list:
            rgb_array = get_image(cam, output=False, output_path=None)

            axs[vert_ind, horz_ind].imshow(rgb_array)
            axs[vert_ind, horz_ind].set_xticks([])
            axs[vert_ind, horz_ind].set_yticks([])

            horz_ind += 1
            if horz_ind == (num_horz_plots):
                horz_ind = 0
                vert_ind += 1

#%% Camera settings
def reset_cams(cam_objs):
    if isinstance(cam_objs, PySpin.CameraPtr):  # A single camera object
        if cam_objs.IsInitialized() is False:
            cam_objs.Init()
        if cam_objs.IsStreaming():
            cam_objs.EndAcquisision()

        cam_objs.UserSetSelector.SetValue(PySpin.UserSetSelector_Default)
        cam_objs.UserSetLoad()

    elif isinstance(cam_objs, list):  # A list of camera objects
        for cam in range(len(cam_objs)):
            if cam_objs[cam].IsInitialized() is False:
                cam_objs[cam].Init()
            if cam_objs[cam].IsStreaming():
                cam_objs[cam].EndAcquisition()

            cam_objs[cam].UserSetSelector.SetValue(
                PySpin.UserSetSelector_Default)
            cam_objs[cam].UserSetLoad()


def set_cam_settings(cam, default=False, frame_rate=None,
                     exposure_time=None, gain=None, trigger_mode=None,
                     acquisition_mode=None, pixel_format=None):
    '''
    CHANGED

    Function for setting most useful camera settings:
        Inputs:
            cam: A camera_ptr object that has been initalized.
            default: boolean for the default params. Cannot be used in
                conjunction with others.
            frame_rate: integer or float of desired frames per second.
            exposure_time: integer of exposure time in microseconds
            gain: integer or float if manual control OR 'auto' for auto gain
            trigger_mode: boolean - On means that it will not take images
                unless a trigger is present
            acquisition_mode: integer:
                0 = continuous - essentially a video
                1 = singleframe - a predefined number of images are taken
                2 = multiframe - a single frame occurs before acquisition ends
            pixel_format: PySpin.PixelFormat_[opt] note that not all options
                are available to all cameras

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
               frame_rate, exposure_time, gain, trigger_mode, acquisition_mode,
               pixel_format]):
            print('Warning: both the default flag and parameters have been'
                  ' passed.\nDefault parameters will be used.')
        frame_rate = 30
        exposure_time = 5000
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
                print('Maximum gain exceeded; gain set to maximum: {}'.format(
                    round(gain, 3)))

            cam.Gain.SetValue(gain)

    # Trigger Mode
    if trigger_mode is not None:
        if trigger_mode:
            cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        else:
            cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

    # Acquisition mode
    if (acquisition_mode is not None and
            not cam.AcquisitionMode.GetValue() == acquisition_mode):
        cam.AcquisitionMode.SetValue(acquisition_mode)

    # Pixel format
    if (pixel_format is not None and
            not cam.PixelFormat.GetValue() == pixel_format):
        cam.PixelFormat.SetValue(pixel_format)


#%% Synchronized capture
def init_sync_settings(cam_objs, primary_idx, secondary_idx, frame_rate = 30, num_images = None):
    # Settings for each camera
    nodemap = []
    for cam in range(len(cam_objs)):
        # Check cam states
        if cam_objs[cam].IsInitialized() is False:
            cam_objs[cam].Init()
        if cam_objs[cam].IsStreaming():
            cam_objs[cam].EndAcquisition()

        # Set buffers for all cameras to oldest first to ensure they are pulled in order
        nodemap.append(cam_objs[cam].GetTLStreamNodeMap())
        PySpin.CEnumerationPtr(nodemap[cam].GetNode('StreamBufferHandlingMode')).SetIntValue(0) 

        # Acquisition mode
        if num_images is None:
            cam_objs[cam].AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        elif isinstance(num_images, int):
            cam_objs[cam].AcquisitionMode.SetValue(PySpin.AcquisitionMode_MultiFrame)
            cam_objs[cam].AcquisitionFrameCount.SetValue(num_images)
            if cam == 0:
                print('Warning: Multiframe acquisition mode is currently broken for synced capture.')

    if isinstance(primary_idx, list): # Just in case
        primary_idx = primary_idx[0]

    # Primary cam settings
    # Triggering
    cam_objs[primary_idx].LineSelector.SetValue(PySpin.LineSelector_Line2)
    cam_objs[primary_idx].LineMode.SetValue(PySpin.LineMode_Output)
    cam_objs[primary_idx].TriggerSource.SetValue(PySpin.TriggerSource_Software)
    # Frame rate
    cam_objs[primary_idx].AcquisitionFrameRateEnable.SetValue(True)
    cam_objs[primary_idx].AcquisitionFrameRate.SetValue(frame_rate)
    cam_objs[primary_idx].TriggerMode.SetValue(PySpin.TriggerMode_On)
    # Enable acquisition - won't start until trigger mode is turned off
    cam_objs[primary_idx].BeginAcquisition()


    # Secondary cam settings
    for cam in secondary_idx:
        cam_objs[cam].TriggerSource.SetValue(PySpin.TriggerSource_Line3)
        cam_objs[cam].TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
        cam_objs[cam].TriggerMode.SetValue(PySpin.TriggerMode_On)
        cam_objs[cam].AcquisitionFrameRateEnable.SetValue(False) # Controlled by trigger
        cam_objs[cam].BeginAcquisition() # This should only prime the cameras


def init_sync_settings_serials(cam_dicts, primary_serial,
                               frame_rate=30, num_images=None):
    '''Accepts serial numbers for primary instead of indices'''
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
            cam_dict['obj'].AcquisitionMode.SetValue(
                PySpin.AcquisitionMode_Continuous)
        elif isinstance(num_images, int):
            cam_dict['obj'].AcquisitionMode.SetValue(
                PySpin.AcquisitionMode_MultiFrame)
            cam_dict['obj'].AcquisitionFrameCount.SetValue(num_images)

    # Primary cam settings
    # Triggering
    cam_dicts[primary_serial]['obj'].LineSelector.SetValue(PySpin.LineSelector_Line2)
    cam_dicts[primary_serial]['obj'].LineMode.SetValue(PySpin.LineMode_Output)
    cam_dicts[primary_serial]['obj'].TriggerSource.SetValue(PySpin.TriggerSource_Software)
    # Frame rate
    cam_dicts[primary_serial]['obj'].AcquisitionFrameRateEnable.SetValue(True)
    cam_dicts[primary_serial]['obj'].AcquisitionFrameRate.SetValue(frame_rate)
    cam_dicts[primary_serial]['obj'].TriggerMode.SetValue(PySpin.TriggerMode_On)
    # Enable acquisition - won't start until trigger mode is turned off
    cam_dicts[primary_serial]['obj'].BeginAcquisition()

    # Secondary cam settings
    for cam_dict in [c for ic, c in cam_dicts.items()
                     if not ic == primary_serial]:
        cam_dict['obj'].TriggerSource.SetValue(PySpin.TriggerSource_Line3)
        cam_dict['obj'].TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
        cam_dict['obj'].TriggerMode.SetValue(PySpin.TriggerMode_On)
        # As is being controlled by the trigger
        cam_dict['obj'].AcquisitionFrameRateEnable.SetValue(False)
        # This should only prime the cameras:
        cam_dict['obj'].BeginAcquisition()


def synced_capture_sequence(cam_objs, primary_idx, secondary_idx, num_images, output_folder = None,
                            separate_folders = True):
    # Sort out file storage
    if output_folder is None:
        output_folder = os.getcwd()

    cam_paths = []
    if separate_folders:
        for cam in range(len(cam_objs)):
            cam_paths.append(os.path.join(output_folder, 'cam'+ str(cam+1)))
            if os.path.isdir(cam_paths[cam]) is False:
                os.mkdir(cam_paths[cam])

    # Check if cams are ready to go
    for cam in cam_objs:
        if cam.IsInitialized() is False or cam.IsStreaming() is False:
            print('Cameras not setup for synchronized capture.')
            return

    # Hit it
    init_time, thread_list = [], []
    cam_objs[primary_idx].TriggerMode.SetValue(PySpin.TriggerMode_Off)
    # We want to offload from images in order across cameras to reduce buffer load equally
    for i in tqdm(range(num_images)):
        for cam in range(len(cam_objs)):
            image = cam_objs[cam].GetNextImage(500)
            f_id = image.GetFrameID()
            if i == 0:
                init_time.append(image.GetTimeStamp())

            if separate_folders:
                dest_folder = cam_paths[cam]
            else:
                dest_folder = output_folder

            thread_list.append(threading.Thread(
                    target = save_image_thread, args = (image, dest_folder, f_id, init_time[cam],
                                                        'cam'+str(cam+1))))
            thread_list[-1].start()


    for thread in thread_list:
        thread.join()

    for cam in cam_objs:
        cam.EndAcquisition()


def synced_capture_sequence_serials(cam_dicts, primary_serial,
                                    num_images, output_folder=None,
                                    separate_folders=True):
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
    init_time = {}
    thread_list = []
    cam_dicts[primary_serial]['obj'].TriggerMode.SetValue(
        PySpin.TriggerMode_Off)
    # We want to offload from images in order across cameras to reduce buffer
    # load equally
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
                args=(image, dest_folder, f_id, init_time[cam_serial],
                      cam_dict['name'])))
            thread_list[-1].start()

    for thread in thread_list:
        thread.join()

    for cam_dict in cam_dicts.values():
        cam_dict['obj'].EndAcquisition()


def synced_capture_sequence_serials_p(
        cam_dicts, primary_serial, num_images,
        output_folder=None, separate_folders=True):
    '''Save in a separate process'''
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
    image_saver_processes = {}
    image_saver_qs = {}
    for cam_serial, cam_dict in cam_dicts.items():
        if separate_folders:
            dest_folder = cam_paths[cam_serial]
        else:
            dest_folder = output_folder

        image_saver_qs[cam_serial] = multiprocessing.SimpleQueue()
        image_saver_processes[cam_serial] = multiprocessing.Process(
            target=image_saver_process,
            args=(image_saver_qs[cam_serial], dest_folder, cam_dict['name']))
        image_saver_processes[cam_serial].start()

    cam_dicts[primary_serial]['obj'].TriggerMode.SetValue(
        PySpin.TriggerMode_Off)
    # We want to offload from images in order across cameras to reduce buffer
    # load equally
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

            image_saver_qs[cam_serial].put((
                image_dat, offsetX, offsetY, width, height, pixelFormat, f_id,
                time_stamp))
            image.Release()

    # for cam_serial in cam_dict.keys():
    #     image_saver_qs[cam_serial].put(None)
    #     image_saver_processes[cam_serial].join()

    for cam_dict in cam_dicts.values():
        cam_dict['obj'].EndAcquisition()


def synced_capture_sequence_serials_ram(
        cam_dicts, primary_serial, num_images,
        output_folder=None, separate_folders=True):
    '''Save in a ram and then dump'''
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
    init_time = {}
    image_lists = {}
    thread_list = []
    for cam_serial in cam_dicts.keys():
        image_lists[cam_serial] = []
    cam_dicts[primary_serial]['obj'].TriggerMode.SetValue(
        PySpin.TriggerMode_Off)
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
            (image_dat, offsetX, offsetY, width, height, pixelFormat, f_id,
             time_stamp) = image_lists[cam_serial][i_image]
            image = PySpin.Image.Create(
                width, height, offsetX, offsetY, pixelFormat, image_dat)

            if separate_folders:
                dest_folder = cam_paths[cam_serial]
            else:
                dest_folder = output_folder

            thread_list.append(threading.Thread(
                target=save_image_function,
                args=(image, dest_folder, f_id, image_lists[cam_serial][0][7],
                      time_stamp, cam_dict['name'])))
            thread_list[-1].start()

    for thread in thread_list:
        thread.join()

    for cam_dict in cam_dicts.values():
        cam_dict['obj'].EndAcquisition()


def image_saver_process(q, dest_folder, cam_name):
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
        (image_dat, offsetX, offsetY, width, height, pixelFormat, f_id,
         time_stamp) = val

        if init_time is None:
            init_time = time_stamp

        print('Camera {} got data. time_stamp: {}'.format(
            cam_name, time_stamp-init_time))

        image = PySpin.Image.Create(width, height, offsetX, offsetY,
                                    pixelFormat, image_dat)

        thread_list.append(threading.Thread(
            target=save_image_function,
            args=(image, dest_folder, f_id, init_time,
                  time_stamp, cam_name)))
    # cleanup
    for thread in thread_list:
        thread.join()


def camera_thread_capture_sequence(cam_dict, num_images, dest_folder):
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


#%% Single camera capture
def save_image_thread(input_image, output_path, frame_num, init_time,
                      file_prefix=None):
    time_stamp = input_image.GetTimeStamp()
    delta_time = '{:08.4f}'.format((time_stamp - init_time) / 1E9)
    filename = '{}_image{}.jpeg'.format(delta_time, frame_num)
    if file_prefix is not None:
        filename = file_prefix + '_' + filename
    filename = os.path.join(output_path, filename)

    input_image.Save(filename)
    input_image.Release()


def save_image_function(input_image, output_path, frame_num, init_time,
                        time_stamp, file_prefix=None):
    '''Does not release'''
    delta_time = '{:08.4f}'.format((time_stamp - init_time) / 1E9)
    filename = '{}_image{}.jpeg'.format(delta_time, frame_num)
    if file_prefix is not None:
        filename = file_prefix + '_' + filename
    filename = os.path.join(output_path, filename)

    input_image.Save(filename)


def capture_sequence(cam_obj, num_images = 50, output_path = None, file_prefix = ''):
    # Check input
    if isinstance(cam_obj, list):
        raise Exception('capture_sequence only accepts "CameraPtr" objects, not lists of them.')
        return

    if output_path is None:
        output_path = os.getcwd()

    set_cam_settings(cam_obj, acquisition_mode = 2)
    cam_obj.AcquisitionFrameCount.SetValue(num_images)

    # Ensure buffer mode is suitable
    nodemap = cam_obj.GetTLStreamNodeMap()
    node_bufferhandling_mode = PySpin.CEnumerationPtr(nodemap.GetNode('StreamBufferHandlingMode'))
    node_oldestfirst_mode = node_bufferhandling_mode.GetEntryByName('OldestFirst').GetValue()
    node_bufferhandling_mode.SetIntValue(node_oldestfirst_mode)

    # Begin threaded collection
    thread_list = []
    cam_obj.BeginAcquisition()
    for idx in range(num_images):
        image = cam_obj.GetNextImage()
        if idx == 0:
            init_time = image.GetTimeStamp()

        thread_list.append(threading.Thread(target = save_image_thread, args = (image, output_path,
                                                                                idx, init_time,
                                                                                file_prefix)))
        thread_list[-1].start()
        thread_list[-1].join()

    cam_obj.EndAcquisition()


def capture_sequence_GUI(cam_obj, num_images = 50, output_path = None, file_prefix = ''):
    # Check input
    if isinstance(cam_obj, list):
        raise Exception('capture_sequence only accepts "CameraPtr" objects, not lists of them.')
        return

    if output_path == 'cwd':
        output_path = os.getcwd()

    # Set parameters
    continue_recording, early_stopping = True, False
    if cam_obj.IsInitialized() is False:
        cam_obj.Init()

    max_fr = cam_obj.AcquisitionFrameRate.GetValue()
    set_cam_settings(cam_obj, acquisition_mode = 0, frame_rate = max_fr)

    if isinstance(num_images, int):
        early_stopping = True
    else:
        print('Press "CNTRL C" to terminate the viewer.')

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
                        target = save_image_thread, args = (image, output_path, idx, init_time,
                                                            file_prefix)))
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
