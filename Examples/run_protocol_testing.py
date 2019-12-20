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

    # pose_estimation_config = ncams.camera_pose.one_shot_multi_PnP(
    #     camera_config, calibration_config)



if __name__ == '__main__':
    main()
    pylab.show()
