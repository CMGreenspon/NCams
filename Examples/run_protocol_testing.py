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


BASE_DIR = os.path.join('C:/', 'FLIR_cameras', 'PublicExample')


def main():
    cdatetime = '2019.11.22_10.00.09'
    camera_config_dir = os.path.join(BASE_DIR, 'camconf_'+cdatetime)
    camera_config = ncams.yaml_to_config(os.path.join(camera_config_dir, 'config.yaml'))

    # %% 2 Run the multi-calibration on all cameras
    calibration_config = ncams.multi_camera_calibration(camera_config, inspect=True)

    return
    calibration_config = ncams.import_calibration(camera_config)

    pose_estimation_config = ncams.camera_positions.one_shot_multi_PnP(
        camera_config, calibration_config)

    # Does it look okay?
    ncams.camera_positions.plot_poses(pose_estimation_config)

    # If so lets export it
    ncams.export_pose_estimation(pose_estimation_config)


if __name__ == '__main__':
    main()
    pylab.show()
