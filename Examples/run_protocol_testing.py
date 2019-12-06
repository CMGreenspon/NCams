"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams
"""
import os
import datetime
import math

import SpinnakerTools


def main():
    working_dir = 'F:\\FLIR_cameras\\exp2019.11.21_14.50.48'
    system, cam_list, cam_serials, cam_dicts = SpinnakerTools.get_system()
    reference_cam_serial = 19194009

    # %% Set up experiment
    pre_touchpad_early_start = 1000  # msec before you press Start Protocol
    # msec additional recording after protocol is finished:
    trailing_waiting_period = 1000
    # from ProtocolParameters.cpp:
    total_trial_time = 3000  # msec maxWaitTime
    intertrial_time = 100  # msec intertrialTime
    number_trials_desired = 110  # nTrialsDesired, first ten for sync
    number_trials_desired = 4  # nTrialsDesired, first ten for sync
    # from Protocol.cpp:
    pretouch_wait = 1500  # msec PRETOUCH_WAIT

    camera_frame_rate = 200

    session_time_length_sec = (
        pre_touchpad_early_start +
        number_trials_desired*(total_trial_time+intertrial_time) +
        trailing_waiting_period) / 1e3
    session_time_length_sec = .5
    session_number_frames = math.ceil(session_time_length_sec * camera_frame_rate)
    print('Going to capture {} frames over {} seconds'.format(
        session_number_frames, session_time_length_sec))
    session_number_frames = 5

    session_number = 10
    session_user = 'AS'
    session_subject = 'CMG'
    session_path = os.path.join(working_dir, 'exp_session_{}_{}_{}'.format(
        session_user, session_subject, session_number))
    if not os.path.isdir(session_path):
        os.mkdir(session_path)
    session_info = dict(
        user=session_user,  # The person present during the recordings
        subject=session_subject,  # Subject of the recording
        date=datetime.date.today(),  # Date of recording
        session_number=1,  # If multiple sessions
        # In case the session config is saved separately from the data:
        session_path=session_path,
        # To set fr of cameras during acquisition
        camera_frame_rate=camera_frame_rate)

    # %% Run experiment
    SpinnakerTools.reset_cams(cam_list)
    for cam in cam_list:
        SpinnakerTools.set_cam_settings(cam, default=True)
    print('Cameras set up done.')

    SpinnakerTools.init_sync_settings_serials(
        cam_dicts, reference_cam_serial,
        frame_rate=session_info['camera_frame_rate'], num_images=None)
    print('Cameras sync done.')

    print('Starting capture')
    SpinnakerTools.synced_capture_sequence_serials_p(
        cam_dicts, reference_cam_serial, session_number_frames,
        output_folder=session_info['session_path'], separate_folders=True)

    print('Capture done')
    SpinnakerTools.reset_cams(cam_list)


if __name__ == '__main__':
    main()
