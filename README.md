# NCams

NCams is a toolbox for recording videos from multiple cameras, analyzing them and extracting 3D marker trajectories.

NCams is installed as a Pythron module with several submodules that include camera calibration, estimation of relative camera positions, triangulation of the marker information from multiple cameras. The module can run on Windows or *nix.

## Getting Started

This project's code is available on [GitHub](https://github.com/CMGreenspon/NCams).

### Prerequisites

Hardware:
- NVIDIA videocard that supports CUDA
- FLIR cameras (if using provided [tools](ncams/spinnaker_t.py) to record)

Software:
- [NVIDIA drivers](https://www.nvidia.com/download/index.aspx)
- [CUDA 10.0](https://developer.nvidia.com/cuda-downloads)
- [DeepLabCut](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md) - if using it for labeling markers. See below for installation recommendations.
- [Spinnaker module](https://flir.app.boxcn.net/v/SpinnakerSDK/folder/68522911814) if using FLIR cameras for capture.

If you have problems with installations, check our [installation tips](INSTALLATION.md).

## Installation

1. Download the [repository](https://github.com/CMGreenspon/NCams) or clone it using git into a folder: `git clone https://github.com/CMGreenspon/NCams.git`.
2. Open Terminal or Command Line or the desired Anaconda environment in the project folder.
3. Run `python setup.py install`.

## Moving DLC network to a different project

1. Change filenames in config.yaml
2. Change filenames in pose_cfg.yaml (two of them)
3. Change directory name `dlc-models/iteration-0/<PROJECT NAME>-trainset95shuffle1`

## Examples of use



## Structure of the repository

- ncams/ -- imported module
    + utils.py -- Utilities for general use, e.g. sort and file search.
    + image_t.py -- Toolbox for working with images and making videos.
    + camera_io.py -- File I/O functions for camera configurations.
    + camera_t.py -- General camera functions and tools.
    + camera_calibration.py -- Camera lense calibration.
    + camera_positions.py -- Estimation of relative positions of the cameras.
    + reconstruction_t.py -- Integration of marker information from multiple cameras
    + spinnaker_t.py -- Recording from FLIR cameras.
- Examples/
    + example_with_recording_data.py -- Record images with FLIR cameras, setup camera parameters, make videos in interactive environment.
    + example_with_supplied_data.py -- Use provided example or own images to setup camera parameters and make videos in interactive environment.
    + analysis.py -- label markers, extract their 3D locations.
    + run_protocol_testing.py -- execute parts of example_with_*_data.py in command line.
    + run_analysis_testing.py -- execute parts of analysis.py in command line.


## Authors

- **Charles M Greenspon** - *Code* - [CMGreenspon](https://github.org/CMGreenspon)
- **Anton Sobinov** - *Code* - [nishbo](https://github.org/nishbo)
- Developed in [Bensmaia Lab](http://bensmaialab.org/).
