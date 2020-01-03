"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams
"""
import os
import datetime

import deeplabcut

import ncams


BASE_DIR = os.path.join('C:\\', 'FLIR_cameras', 'PublicExample')
os.environ['DLC_PER_PROCESS_GPU_MEMORY_FRACTION'] = '0.9'


def main():
    # cdatetime = '2019.12.19_10.38.38'
    # camera_config_dir = os.path.join(BASE_DIR, 'camconf_'+cdatetime)
    # camera_config = ncams.yaml_to_config(os.path.join(camera_config_dir, 'config.yaml'))

    # calibration_config, pose_estimation_config = ncams.load_camera_config(camera_config)

    # #  Load a session config from a file
    # session_full_filename = os.path.join(BASE_DIR, 'exp_session_2019.12.09_16.40.45_AS_CMG_2',
    #                                      'session_config.yaml')
    # session_config = ncams.import_session_config(session_full_filename)


    # %% 2b Load an existing DLC project with the labeled frames
    dlc_prj_name = '2019.12.20_8camsNoMarkers'
    scorer = 'AS'
    prj_date = '2019-12-23'
    dlc_proj_name = '-'.join([dlc_prj_name, scorer, prj_date])
    proj_path = os.path.join(BASE_DIR, dlc_proj_name)
    config_path = os.path.join(proj_path, 'config.yaml')

    print('Existing config_path: "{}"'.format(config_path))

    labeled_csv_path = os.path.join(proj_path, 'labeled_videos')
    if not os.path.isdir(labeled_csv_path):
        os.mkdir(labeled_csv_path)

    # print('training network')
    # deeplabcut.train_network(config_path, gputouse=0, saveiters=100, maxiters=250000,
    #                          displayiters=10)

    print('Evaluating network...')
    deeplabcut.evaluate_network(config_path, plotting=False)

    # which videos do you want to train on?
    training_videos = [os.path.join(BASE_DIR, 'exp_session_2019.12.20_videos', fname)
                       for fname in (
            '4_cam19194005.mp4', '4_cam19194008.mp4', '4_cam19194009.mp4', '4_cam19194013.mp4',
            '4_cam19335177.mp4', '4_cam19340298.mp4', '4_cam19340300.mp4', '4_cam19340396.mp4',
            '5_cam19194005.mp4', '5_cam19194008.mp4', '5_cam19194009.mp4', '5_cam19194013.mp4',
            '5_cam19335177.mp4', '5_cam19340298.mp4', '5_cam19340300.mp4', '5_cam19340396.mp4',
            '6_cam19194005.mp4', '6_cam19194008.mp4', '6_cam19194009.mp4', '6_cam19194013.mp4',
            '6_cam19335177.mp4', '6_cam19340298.mp4', '6_cam19340300.mp4', '6_cam19340396.mp4')]
    print('analyzing videos')
    deeplabcut.analyze_videos(config_path, training_videos,
                              gputouse=0, save_as_csv=True, destfolder=labeled_csv_path)

    print('making labeled videos')
    deeplabcut.create_labeled_video(config_path, training_videos, destfolder=labeled_csv_path,
                                    draw_skeleton=True)

    # # %% 3 Triangulation from multiple cameras
    # triangulated_path = os.path.join(proj_path, 'triangulated')
    # if not os.path.exists(triangulated_path):
    #     os.mkdir(triangulated_path)

    # method = 'full_rank'
    # triangulated_csv = os.path.join(triangulated_path, 'triangulated_points_'+method+'.csv')
    # threshold = 0.9
    # ncams.triangulate(
    #     camera_config, session_config, calibration_config, pose_estimation_config, labeled_csv_path,
    #     threshold=threshold, method=method, output_csv=triangulated_csv)

    # # %% 4 Make markered videos
    # ncams.make_triangulation_videos(
    #     camera_config, session_config, triangulated_csv,
    #     triangulated_path=triangulated_path, overwrite_temp=True, parallel=12)

    # %% 5 Interactive demonstration with a slider
    # ncams.reconstruction.interactive_3d_plot(
    #     camera_config['serials'][0], camera_config, session_config, triangulated_csv,
    #     num_frames_limit=None)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    print('Time elapsed:')
    print(datetime.datetime.now() - start_time)
