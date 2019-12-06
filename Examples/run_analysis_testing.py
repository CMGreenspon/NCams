"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams
"""
import os
import datetime

import deeplabcut
import CameraTools

import ReconstructionTools


def main():
    working_dir = 'C:/FLIR_cameras/exp2019.11.22_10.00.09'
    session_path = os.path.join(working_dir, 'exp_session_AS_CMG_2')
    camera_frame_rate = 50
    video_path = os.path.join(session_path, 'videos')
    ud_video_path = os.path.join(session_path, 'undistorted_videos')
    camera_config = CameraTools.yaml_to_config(os.path.join(working_dir, '2019-11-22_config.yaml'))
    cam_dicts = {}
    for cam_serial, cam_name in zip(camera_config['camera_serials'],
                                    camera_config['camera_names']):
        cam_dicts[cam_serial] = {}
        cam_dicts[cam_serial]['name'] = cam_name
        cam_dicts[cam_serial]['video'] = os.path.join(video_path, cam_name+'.mp4')
        cam_dicts[cam_serial]['ud_video'] = os.path.join(ud_video_path, cam_name+'_undistorted.mp4')
    cam_serials = sorted(camera_config['camera_serials'])

    (camera_matrices, distortion_coefficients, reprojection_errors,
     world_locations, world_orientations) = CameraTools.load_camera_config(
         camera_config)

    training_videos = [cam_dicts[cs]['video'] for cs in cam_serials]

    # %% 2 Make DLC project
    dlc_prj_name = 'CMGPretrainedNetwork'
    scorer = 'CMG'
    prj_date = '2019-12-03'
    dlc_proj_name = '-'.join([dlc_prj_name, scorer, prj_date])

    proj_path = os.path.join(session_path, dlc_proj_name)

    labeled_video_path = os.path.join(proj_path, 'labeled_videos')

    # if project exists, return to it - could not find an "open project" function
    config_path = os.path.join(proj_path, 'config.yaml')

    # %%
    # 0 is the GPU number, see in nvidia-smi
    # deeplabcut.train_network(config_path, gputouse=0, displayiters=1, saveiters=50, maxiters=250)
    # deeplabcut.train_network(config_path, gputouse=0, displayiters=1, saveiters=50, maxiters=100, allow_growth=True)
    # deeplabcut.train_network(config_path, gputouse=0, saveiters=25000, maxiters=250000)
    # deeplabcut.evaluate_network(config_path)
    # deeplabcut.analyze_videos(config_path, training_videos,
    #                           gputouse=0, save_as_csv=True, destfolder=labeled_video_path)


    # deeplabcut.create_labeled_video(config_path, training_videos, destfolder=labeled_video_path,
    #                                 draw_skeleton=True)


    images_3d_path = os.path.join(proj_path, 'rec_3d')
    threshold = 0.9

    method = 'best_pair'
    triangulated_csv = os.path.join(images_3d_path, 'triangulated_points_'+method+'.csv')
    # ReconstructionTools.triangulate(
    #     cam_dicts, camera_config, session_path, labeled_video_path,
    #     threshold=threshold, images_3d_path=images_3d_path,
    #     method=method, num_frames_limit=None,
    #     output_csv=triangulated_csv)

    ReconstructionTools.make_triangulation_videos(
        camera_config, cam_dicts, session_path, triangulated_csv,
        images_3d_path=images_3d_path, overwrite_temp=True, fps=camera_frame_rate,
        num_frames_limit=None, parallel=12)
    # ReconstructionTools.interactive_3d_plot(
    #     19194005, camera_config, cam_dicts, session_path, triangulated_csv,
    #     num_frames_limit=None)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    print('Time elapsed:')
    print(datetime.datetime.now() - start_time)
