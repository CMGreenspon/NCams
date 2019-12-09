"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams
"""
import os
import datetime

import deeplabcut

import ncams.camera_io
import ncams.reconstruction_t


BASE_DIR = os.path.join('C:/', 'FLIR_cameras')


def main():
    cdatetime = '2019.11.22_10.00.09'
    camera_config_dir = os.path.join(BASE_DIR, 'camconf_'+cdatetime)
    camera_config = ncams.camera_io.yaml_to_config(os.path.join(camera_config_dir, 'config.yaml'))

    calibration_config, pose_estimation_config = ncams.camera_io.load_camera_config(camera_config)

    #  Load a session config from a file
    session_full_filename = os.path.join(BASE_DIR, 'exp_session_2019.11.22_10.00.09_AS_CMG_2',
                                         'session_config.yaml')
    session_config = ncams.utils.import_session_config(session_full_filename)

    # which videos do you want to train on?
    training_videos = [session_config['cam_dicts'][cs]['video'] for cs in camera_config['serials']]

    # %% 2b Load an existing DLC project with the labeled frames
    dlc_prj_name = 'CMGPretrainedNetwork'
    scorer = 'CMG'
    prj_date = '2019-12-03'
    dlc_proj_name = '-'.join([dlc_prj_name, scorer, prj_date])
    proj_path = os.path.join(session_config['session_path'], dlc_proj_name)
    config_path = os.path.join(proj_path, 'config.yaml')

    print('Existing config_path: "{}"'.format(config_path))

    labeled_csv_path = os.path.join(proj_path, 'labeled_videos')
    if not os.path.isdir(labeled_csv_path):
        os.mkdir(labeled_csv_path)

    # %% 3 Triangulation from multiple cameras
    triangulated_path = os.path.join(proj_path, 'triangulated')
    if not os.path.exists(triangulated_path):
        os.mkdir(triangulated_path)

    method = 'full_rank'
    triangulated_csv = os.path.join(triangulated_path, 'triangulated_points_'+method+'.csv')
    threshold = 0.9
    # ncams.reconstruction_t.triangulate(
    #     camera_config, session_config, calibration_config, pose_estimation_config, labeled_csv_path,
    #     threshold=threshold, method=method, output_csv=triangulated_csv)

    # %% 4 Make markered videos
    # ncams.reconstruction_t.make_triangulation_videos(
    #     camera_config, session_config, calibration_config, pose_estimation_config, triangulated_csv,
    #     triangulated_path=triangulated_path, overwrite_temp=True)

    # %% 5 Interactive demonstration with a slider
    ncams.reconstruction_t.interactive_3d_plot(
        camera_config['serials'][0], camera_config, session_config, triangulated_csv,
        num_frames_limit=None)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    print('Time elapsed:')
    print(datetime.datetime.now() - start_time)
