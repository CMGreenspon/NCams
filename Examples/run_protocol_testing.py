"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams
"""
import os
import time
import math
import pylab

import ncams


BASE_DIR = os.path.join('C:\\', 'FLIR_cameras', 'PublicExample')


def main():
    cdatetime = '2019.12.19_10.38.38';
    camera_config_dir = os.path.join(BASE_DIR, 'camconf_'+cdatetime)
    camera_config = ncams.yaml_to_config(os.path.join(camera_config_dir, 'config.yaml'))

    calibration_config, pose_estimation_config = ncams.load_camera_config(camera_config)

    session_shortnames = (
        'exp_session_2019.12.20_09.49.42_AS_CMG_1',
        'exp_session_2019.12.20_09.56.37_AS_CMG_2',
        'exp_session_2019.12.20_09.57.31_AS_CMG_3',
        'exp_session_2019.12.20_09.58.36_AS_CMG_4',
        'exp_session_2019.12.20_10.09.44_AS_CMG_5',
        'exp_session_2019.12.20_10.16.13_AS_CMG_6',
        'exp_session_2019.12.20_10.34.40_AS_CMG_7',
        'exp_session_2019.12.20_10.39.45_AS_CMG_8',
        'exp_session_2019.12.20_10.45.01_AS_CMG_9',
        'exp_session_2019.12.20_10.51.06_AS_CMG_10',
        'exp_session_2019.12.20_11.11.21_AS_CMG_11',
        'exp_session_2019.12.20_11.17.24_AS_CMG_12',
        'exp_session_2019.12.20_11.21.52_AS_CMG_13',
    )

    for session_shortname in session_shortnames:
        print('Processing session {}'.format(session_shortname))
        session_full_filename = os.path.join(BASE_DIR, session_shortname, 'session_config.yaml')
        session_config = ncams.import_session_config(session_full_filename)

        session_config['video_path'] = 'videos'
        session_config['ud_video_path'] = 'undistorted_videos'

        for p in (os.path.join(session_config['session_path'], session_config['video_path']),
                  os.path.join(session_config['session_path'], session_config['ud_video_path'])):
            if not os.path.isdir(p):
                print('Making dir {}'.format(p))
                os.mkdir(p)

        for serial in camera_config['serials']:
            session_config['cam_dicts'][serial]['pic_dir'] = session_config['cam_dicts'][serial]['name']
            session_config['cam_dicts'][serial]['video'] = os.path.join(
                session_config['video_path'], session_config['cam_dicts'][serial]['name']+'.mp4')
            session_config['cam_dicts'][serial]['ud_video'] = os.path.join(
                session_config['ud_video_path'], session_config['cam_dicts'][serial]['name']+'.mp4')

        for cam_dict in session_config['cam_dicts'].values():
            image_list = ncams.utils.get_image_list(
                sort=True, path=os.path.join(session_config['session_path'], cam_dict['pic_dir']))
            print('\tMaking a video for camera {} from {} images.'.format(
                cam_dict['name'], len(image_list)))
            ncams.images_to_video(
                image_list, cam_dict['video'], fps=session_config['frame_rate'],
                output_folder=session_config['session_path'])

        for icam, serial in enumerate(camera_config['serials']):
            cam_dict = session_config['cam_dicts'][serial]
            ncams.undistort_video(
                os.path.join(session_config['session_path'], cam_dict['video']),
                calibration_config['dicts'][serial],
                crop_and_resize=False,
                output_filename=os.path.join(session_config['session_path'], cam_dict['ud_video']))
            print('\tCamera {} video undistorted.'.format(cam_dict['name']))

        ncams.export_session_config(session_config)


if __name__ == '__main__':
    main()
    pylab.show()
