# NCams

NCams is a toolbox for recording videos from multiple cameras, analyzing them and extracting 3D marker trajectories.

NCams is installed as a Pythron module with several submodules that include camera calibration, estimation of relative camera positions, triangulation of the marker information from multiple cameras. The module can run on Windows or *nix.

## Getting Started

This project's code is available on [GitHub](https://github.com/CMGreenspon/NCams). The example raw data is available on [Dropbox](https://www.dropbox.com/sh/k7obdf85dhmynvp/AABiCadSKrQEHYv0nfdjUpjBa).

### Prerequisites

Hardware:
- NVIDIA videocard that supports CUDA
- FLIR cameras (if using provided [tools](ncams/spinnaker_tools.py) to record)

Software:
- [DeepLabCut](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md) - if using it for labeling markers. See below for installation recommendations.
 - [NVIDIA drivers](https://www.nvidia.com/download/index.aspx)
 - [CUDA 10.0](https://developer.nvidia.com/cuda-downloads)
- [Spinnaker module](https://flir.app.boxcn.net/v/SpinnakerSDK/folder/68522911814) if using FLIR cameras for capture.

If you have problems with installations, check our [installation tips](INSTALLATION.md).

## Installation

1. Download the [repository](https://github.com/CMGreenspon/NCams) or clone it using git into a folder: `git clone https://github.com/CMGreenspon/NCams.git`.
2. Open Terminal or Command Line or the desired Anaconda environment in the project folder.
3. Run `python setup.py install`.

## Examples of use

### Calibration and pose estimation

[Example with capturing data](Examples/example_with_recording_data.py) contains example code for setting up multiple FLIR cameras, calibration of lenses, estimation of their poses, and creation of videos.

[Example with supplied data](Examples/example_with_supplied_data.py) can help you calibrate the lenses on the cameras, estimate their poses, and create of videos. The data to run the code on is stored on [Dropbox](???)

### Labeling and 3D marker reconstruction

The [analysis](Examples/analysis.py) goes over marking images with DeepLabCut and triangulation of the marker data.

### Moving DLC network to a different project

1. Change filenames in config.yaml
2. Change filenames in pose_cfg.yaml (two of them)
3. Change directory name `dlc-models/iteration-0/<PROJECT NAME>-trainset95shuffle1`

### Continuing teaching a NN on new videos

1. Edit your config.yaml by replacing the video list with the new videos, save the text referencing the old videos. You may want to change the `numframes2pick` variable in config, too.
2. Extract frames: `deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', crop=False, userfeedback=False)`
3. Label frames: `deeplabcut.label_frames(config_path)`
4. Put back the old videos paths (do not remove the new ones) into the config.yaml.
5. Merge datasets: `deeplabcut.merge_datasets(config_path)`
6. Create training dataset. `deeplabcut.create_training_dataset(config_path)`
7. Go to train and test pose_cfg.yaml of the new interation (e.g. in '<DLC_PROJECT>/dlc-models/iteration-1/CMGPretrainedNetworkDec3-trainset95shuffle1/train/pose_cfg.yaml' and '<DLC_PROJECT>/dlc-models/iteration-1/CMGPretrainedNetworkDec3-trainset95shuffle1/test/pose_cfg.yaml') and change the `init_weights` variable to point to the snapshot from previous network or iteration (for example, '<DLC_PROJECT>\dlc-models\iteration-0\CMGPretrainedNetworkDec3-trainset95shuffle1\train\snapshot-250000' without file extension).
Note: put <DLC_PROJECT> as a full path from root directory or drive.
8. Start training.


## Structure of the repository

- ncams/ -- imported module
    + utils.py -- Utilities for general use, e.g. sort and file search.
    + image_tools.py -- Toolbox for working with images and making videos.
    + camera_io.py -- File I/O functions for camera configurations.
    + camera_tools.py -- General camera functions and tools. `help(ncams.camera_tools)` has information on most configuration structures.
    + camera_calibration.py -- Camera lense calibration.
    + camera_pose.py -- Estimation of relative positions and orientations of the cameras.
    + reconstruction.py -- Integration of marker information from multiple cameras
    + spinnaker_tools.py -- Recording from FLIR cameras. Is not automatically imported.
- Examples/
    + example_with_recording_data.py -- Record images with FLIR cameras, setup camera parameters, make videos in interactive environment.
    + example_with_supplied_data.py -- Use provided example or own images to setup camera parameters and make videos in interactive environment.
    + analysis.py -- label markers, extract their 3D locations.
    + run_protocol_testing.py -- execute parts of example_with_*_data.py in command line.
    + run_analysis_testing.py -- execute parts of analysis.py in command line.


## Authors

- [**Charles M Greenspon**](https://github.org/CMGreenspon)
- [**Anton Sobinov**](https://github.org/nishbo)
- Developed in [Bensmaia Lab](http://bensmaialab.org/).
