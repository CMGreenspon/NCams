"""
NCams Toolbox
Copyright 2019-2020 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams
"""
import os
import datetime
import ntpath

# import deeplabcut

import ncams


BASE_DIR = os.path.join('C:\\', 'FLIR_cameras', 'PublicExample')
os.environ['DLC_PER_PROCESS_GPU_MEMORY_FRACTION'] = '0.9'


def main():
    cdatetime = '2019.12.19_10.38.38'
    camera_config_dir = os.path.join(BASE_DIR, 'camconf_'+cdatetime)
    camera_config = ncams.yaml_to_config(os.path.join(camera_config_dir, 'config.yaml'))

    calibration_config, pose_estimation_config = ncams.load_camera_config(camera_config)

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

    # print('Evaluating network...')
    # deeplabcut.evaluate_network(config_path, plotting=False)

    # which videos do you want to train on?
    video_path = os.path.join(BASE_DIR, 'exp_session_2019.12.20_videos')
    training_videos = [os.path.join(video_path, fname) for fname in (
        '4_cam19194005.mp4', '4_cam19194008.mp4', '4_cam19194009.mp4', '4_cam19194013.mp4',
        '4_cam19335177.mp4', '4_cam19340298.mp4', '4_cam19340300.mp4', '4_cam19340396.mp4',
        '5_cam19194005.mp4', '5_cam19194008.mp4', '5_cam19194009.mp4', '5_cam19194013.mp4',
        '5_cam19335177.mp4', '5_cam19340298.mp4', '5_cam19340300.mp4', '5_cam19340396.mp4',
        '6_cam19194005.mp4', '6_cam19194008.mp4', '6_cam19194009.mp4', '6_cam19194013.mp4',
        '6_cam19335177.mp4', '6_cam19340298.mp4', '6_cam19340300.mp4', '6_cam19340396.mp4')]
    # print('analyzing videos')
    # deeplabcut.analyze_videos(config_path, training_videos,
    #                           gputouse=0, save_as_csv=True, destfolder=labeled_csv_path)

    # print('making labeled videos')
    # deeplabcut.create_labeled_video(config_path, training_videos, destfolder=labeled_csv_path,
    #                                 draw_skeleton=True)

    # %% 3 Triangulation from multiple cameras
    method = 'full_rank'
    threshold = 0.9
    triangulated_path = os.path.join(proj_path, 'triangulated_{}_{}'.format(method, threshold))
    if not os.path.exists(triangulated_path):
        os.mkdir(triangulated_path)

    # make all videos
    # file_prefixes = ['4', '5', '6']
    # for file_prefix in file_prefixes:
    #     print('Working on session {}'.format(file_prefix))
    #     triangulated_path2 = os.path.join(triangulated_path, 'session{}'.format(file_prefix))
    #     triangulated_csv = os.path.join(triangulated_path2, 'triangulated_points{}.csv'.format(
    #         '' if len(file_prefix) == 0 else '_'+file_prefix))
    #     triangulated_csv_p = os.path.join(
    #         triangulated_path2, 'triangulated_points{}_smoothed.csv'.format(
    #             '' if len(file_prefix) == 0 else '_'+file_prefix))

    #     # ncams.triangulate(
    #     #     camera_config, triangulated_csv, calibration_config, pose_estimation_config, labeled_csv_path,
    #     #     threshold=threshold, method=method, undistorted_data=True, file_prefix=file_prefix)

    #     # ncams.process_triangulated_data(triangulated_csv, output_csv=triangulated_csv_p)

    #     # videos = [i for i in training_videos if ntpath.basename(i)[0] == file_prefix]

    #     # # %% 4 Make markered videos
    #     # ncams.make_triangulation_videos(camera_config, camera_config['serials'], videos,
    #     #                                 triangulated_csv_p, skeleton_config=config_path)

    #     # %% 5 Interactive demonstration with a slider
    #     # ncams.reconstruction.interactive_3d_plot(
    #     #     camera_config['serials'][0], camera_config, session_config, triangulated_csv,
    #     #     num_frames_limit=None)

    # make 1 pretty one
    file_prefix = '4'
    print('Working on session {}'.format(file_prefix))
    triangulated_path2 = os.path.join(triangulated_path, 'session{}'.format(file_prefix))
    triangulated_csv_p = os.path.join(
        triangulated_path2, 'triangulated_points{}_smoothed.csv'.format(
            '' if len(file_prefix) == 0 else '_'+file_prefix))

    serial = 19335177
    frame_range = (260, 360)
    video_path = [i for i in training_videos
                  if ntpath.basename(i)[0] == file_prefix and str(serial) in i][0]
    output_path = os.path.join(triangulated_path2, 'pretty_{}_{}.mp4'.format(serial, file_prefix))
    ik_video_path = os.path.join(proj_path, 'ik_videos', 'marshmallow.webm')
    ncams.make_triangulation_video(
        video_path, triangulated_csv_p, skeleton_config=config_path,
        frame_range=frame_range, output_path=output_path,
        thrd_video_path=ik_video_path, thrd_video_frame_offset=0,
        third_video_crop_hw=[slice(50, -100), slice(350, -700)], figure_dpi=300,
        ranges=((-0.33, 3), (-2, 2), (-1.33, 6.74)))
    # ((-0.33, 8.84), (-1.28, 7.30), (-1.33, 6.74))

    # ncams.reconstruction.interactive_3d_plot(video_path, triangulated_csv_p,
    #                                          skeleton_path=config_path)



if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    print('Time elapsed:')
    print(datetime.datetime.now() - start_time)
