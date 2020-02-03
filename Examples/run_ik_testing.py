#!python36
# -*- coding: utf-8 -*-
"""
NCams Toolbox
Copyright 2019 Charles M Greenspon, Anton Sobinov
https://github.com/CMGreenspon/NCams

"""
import os

from scipy.spatial.transform import Rotation as R

import ncams


def import_kinematics():
    # Translate the triangulated data into OSim trc format
    BASE_DIR = os.path.join('C://', 'FLIR_cameras', 'PublicExample')
    proj_path = os.path.join(BASE_DIR, '2019.12.20_8camsNoMarkers-AS-2019-12-23')
    triangulated_path = os.path.join(proj_path, 'triangulated_full_rank_0.9', 'session4')
    ik_dir = os.path.join(proj_path, 'inverse_kinematics')
    if not os.path.isdir(ik_dir):
        os.mkdir(ik_dir)

    # load a csv file into a dictionary by specified column names
    marker_name_dict = ncams.utils.dic_from_csv('marker_meta.csv', 'sDlcMarker', 'sOpenSimMarker')

    triangulated_csv = os.path.join(triangulated_path, 'triangulated_points_4_smoothed.csv')

    # rotate the data from the NCams coordinate system
    # preview the rotations by loading the model and using 'File->Preview experimental data'
    # the right click on the loaded kinematics and 'Transform'. If using our model and our
    # calibration, the rotations should be as described below:
    r = R.from_euler('zyx', [0, 90, 180], degrees=True)
    # scipy.spatial.transform.Rotation.apply returns an ndarray with vertical vectors, so the
    # function is changed in the lambda
    rot = lambda v: r.apply(v)[0].tolist()

    suffixes = ['remote', 'marshmallow', 'wave', 'pen']
    frame_ranges = [(103, 140), (260, 360), (510, 650), (1919, 2019)]
    for frame_range, suffix in zip(frame_ranges, suffixes):
        trc_file = os.path.join(ik_dir, 'triangulated_4_{}.trc'.format(suffix))

        # makes an IK config
        ik_file = os.path.join(ik_dir, 'full_arm_model_IK_4_{}.xml'.format(suffix))
        ik_out_mot_file = os.path.join(ik_dir, 'out_inv_kin_4_{}.mot'.format(suffix))

        ncams.inverse_kinematics.triangulated_to_trc(
            triangulated_csv, trc_file, marker_name_dict,
            data_unit_convert=lambda x: x*100,  # dm to mm
            rate=50, zero_marker='scapula_anterior', frame_range=frame_range, rotation=rot,
            ik_file=ik_file, ik_out_mot_file=ik_out_mot_file)


def filter_joint_angles():
    # Translate the triangulated data into OSim trc format
    BASE_DIR = os.path.join('C://', 'FLIR_cameras', 'PublicExample')
    proj_path = os.path.join(BASE_DIR, '2019.12.20_8camsNoMarkers-AS-2019-12-23')
    ik_dir = os.path.join(proj_path, 'inverse_kinematics')

    suffixes = ['remote', 'marshmallow', 'wave', 'pen']
    for suffix in suffixes:
        ik_out_mot_file = os.path.join(ik_dir, 'out_inv_kin_4_{}.mot'.format(suffix))
        ik_filtered_mot_file = os.path.join(ik_dir, 'out_inv_kin_4_{}_filtered.mot'.format(suffix))
        ncams.inverse_kinematics.smooth_motion(ik_out_mot_file, ik_filtered_mot_file,
                                               median_kernel_size=11)


def make_combined_videos():
    # Translate the triangulated data into OSim trc format
    BASE_DIR = os.path.join('C://', 'FLIR_cameras', 'PublicExample')
    proj_path = os.path.join(BASE_DIR, '2019.12.20_8camsNoMarkers-AS-2019-12-23')
    config_path = os.path.join(proj_path, 'config.yaml')
    ik_dir = os.path.join(proj_path, 'inverse_kinematics')

    triangulated_path = os.path.join(proj_path, 'triangulated_full_rank_0.9', 'session4')
    triangulated_csv = os.path.join(triangulated_path, 'triangulated_points_4_smoothed.csv')

    suffixes = ['remote', 'marshmallow', 'wave', 'pen']
    frame_ranges = [(103, 140), (260, 360), (510, 650), (1919, 2019)]
    # estimate manually with an external program, e.g. MPC-HC, easier if recorded more than one loop
    # from OpenSim
    frame_offsets = [23, 0, 7, 3]
    video_path = os.path.join(BASE_DIR, 'exp_session_2019.12.20_videos', '4_cam19335177.mp4')
    for frame_range, suffix, frame_offset in zip(frame_ranges, suffixes, frame_offsets):
        # Load the motion generated during inverse kinematics and play it.
        # To record a video, press a camera button in the top right corner of the viewer. To stop
        # recording, press the button again. Save the video path to 'ik_video_path'.
        ik_video_path = os.path.join(ik_dir, '4_{}.webm'.format(suffix))  # manually set filename
        output_path = os.path.join(ik_dir, '4_{}_19335177_4.mp4'.format(suffix))
        ncams.make_triangulation_video(
            video_path, triangulated_csv, skeleton_config=config_path,
            frame_range=frame_range, output_path=output_path,
            thrd_video_path=ik_video_path,
            thrd_video_frame_offset=frame_offset,  # if the IK movement starts later
            third_video_crop_hw=[slice(0, -100), slice(350, -700)],  # crops the IK video
            figure_dpi=300,
            ranges=((-0.33, 3), (-2, 2), (-1.33, 6.74)),  # manually set ranges for 3D plot
            plot_markers=False)


if __name__ == '__main__':
    # import_kinematics()
    # filter_joint_angles()
    make_combined_videos()
