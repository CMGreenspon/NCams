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


def main():
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
    trc_file = os.path.join(ik_dir, 'triangulated_4_marshmallow.trc')
    frame_range = (260, 360)

    # makes an IK config
    ik_file = os.path.join(ik_dir, 'full_arm_model_IK_4_marshmallow.xml')
    ik_out_mot_file = os.path.join(ik_dir, 'out_inv_kin_4_marshmallow_lt.mot')

    # rotate the data from the NCams coordinate system
    # preview the rotations by loading the model and using 'File->Preview experimental data'
    # the right click on the loaded kinematics and 'Transform'. If using our model and our
    # calibration, the rotations should be as described below:
    r = R.from_euler('zyx', [0, 90, 180], degrees=True)
    # scipy.spatial.transform.Rotation.apply returns an ndarray with vertical vectors, so the
    # function is changed in the lambda
    rot = lambda v: r.apply(v)[0].tolist()

    ncams.inverse_kinematics.triangulated_to_trc(
        triangulated_csv, trc_file, marker_name_dict,
        data_unit_convert=lambda x: x*100,  # dm to mm
        rate=50, zero_marker='scapula_anterior', frame_range=frame_range, rotation=rot,
        ik_file=ik_file, ik_out_mot_file=ik_out_mot_file)


if __name__ == '__main__':
    main()
