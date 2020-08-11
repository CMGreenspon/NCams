# Main NCams module

- [utils.py](utils.py) -- Utilities for general use, e.g. sort and file search.
- [image_tools.py](image_tools.py) -- Toolbox for working with images and making videos.
- [camera_io.py](camera_io.py) -- File I/O functions for camera configurations.
- [camera_tools.py](camera_tools.py) -- General camera functions and tools. `help(ncams.camera_tools)` has information on most configuration structures.
- [camera_calibration.py](camera_calibration.py) -- Camera lense calibration.
- [camera_pose.py](camera_pose.py) -- Estimation of relative positions and orientations of the cameras.
- [reconstruction.py](reconstruction.py) -- Integration of marker information from multiple cameras.
- [spinnaker_tools.py](spinnaker_tools.py) -- Recording from FLIR cameras. Is not automatically imported.
- [inverse_kinematics.py](inverse_kinematics.py) -- Exporting the triangulated data for OpenSim and processing kinematics.
