# NCams

NCams is a toolbox for recording and triangulating pictures and movies from multiple cameras.

(long description)

## Getting Started

This project's code is available on [GitHub](https://github.com/CMGreenspon/NCams).

The toolbox is intended to be used in an interactive Python environment, for example Spyder. The Setting up DLC chapters provide a step-by-step guide on how to setup the DeepLabCut for use either with Anaconda or through a system-wide Python.

### Prerequisites

- NVIDIA videocard that supports CUDA
- FLIR cameras

Install EITHER through and Anaconda approach or through System-wide installation. Anaconda approach is easier, while system-wide approach provides better flexibility and control.

For FLIR cameras setup, you will need Spinnaker installed:
1. Install SpinView for testing whether the cameras work with full [SDK](https://flir.app.boxcn.net/v/SpinnakerSDK/folder/73503062578).
2. Test whether the cameras are properly setup by running SpinView (for example, located in `C:\Program Files\Point Grey Research\Spinnaker\bin64\vs2015\SpinView_WPF.exe`)
3. Integrate Spinnaker [module](https://flir.app.boxcn.net/v/SpinnakerSDK/folder/68522911814) into your Python (after installing DeepLabCut, using pip install).

### Setting up DLC using Anaconda approach

1. Install [NVIDIA drivers](https://www.nvidia.com/download/index.aspx). No need to install NVIDIA Experience.

2. Install [CUDA 10.0](https://developer.nvidia.com/cuda-downloads). When installing, choose a custom installation and DO NOT install other drivers (e.g. videocard drivers).

3. Install [Anaconda3](https://www.anaconda.com/distribution/#download-section)

4. Follow instructions on [DeepLabCut README](https://github.com/AlexEMG/DeepLabCut/blob/master/conda-environments/README.md) to install DeepLabCut.
The Toolbox is ready to use!

Additional notes:
- [General DeepLabCut installation](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md)

### Setting up DLC using system-wide approach

1. Install [NVIDIA drivers](https://www.nvidia.com/download/index.aspx). No need to install NVIDIA Experience.

2. Install [CUDA 10.0](https://developer.nvidia.com/cuda-downloads). When installing, choose a custom installation and DO NOT install other drivers (e.g. videocard drivers).

3. Install [CuDNN 7.6.5 for CUDA 10.0](https://developer.nvidia.com/rdp/cudnn-download)

4. Install [Python 3.6](https://www.python.org/downloads/release/python-368/). 3.6 needs to be used because DeepLabCut depends on `tables` module version 3.4.3, which we had problems with when compiling on Python 3.7.

5. Install TensorFlow:
```
> pip install tensorflow-gpu=1.14
```
See below for instructions if you have more than one Python3 installed.

6. Install the needed modules on Python:
    1. DeepLabCut
    2. Spyder
    3. PySpin
    4. opencv-contrib-python
The easiest way to install is to use pip from command line:
```
> pip install deeplabcut
```

If you have more than one Python installed (e.g. Python 3.7 and Python 3.6), you can target a specific one using `py -<version>` modifier, if you have `py` configured. You can target scripts like `pip` from them: `py -3.6 -m pip`. For example, this will install the module:
```
> py -3.6 -m pip install deeplabcut
```
If the installation script brakes at any point, check which module it could not install, then try a [wheel](https://www.lfd.uci.edu/~gohlke/pythonlibs/) for that module. The wheel is installed using pip from command line, e.g.:
```
Downloads> py -3.6 -m pip install Shapely-1.6.4.post2-cp36-cp36m-win_amd64.whl
```

7. Open Spyder using `Python36/Scripts/spyder3.exe` and open Protocol.py in it.
The Toolbox is ready to use!

## Moving DLC network to a different project

1. Change filenames in config.yaml
2. Change filenames in pose_cfg.yaml (two of them)
3. Change directory name `dlc-models/iteration-0/<PROJECT NAME>-trainset95shuffle1`

## Protocol Example Script


## Structure of the repository

- Protocol.py - example of using the FLIR cameras to record an experiment, make movies, train DeepLabCut, use it to predict marker positions, triangulate and extract 3D kinematic marker positions. See Protocol Example Script for more information.
- CameraTools.py
- SpinnakerTools.py
- ImageTools.py
- ReconstructionTools.py

## Authors

- **Charles M Greenspon** - *Code* - [CMGreenspon](https://github.org/CMGreenspon)
- **Anton Sobinov** - *Code* - [nishbo](https://github.org/nishbo)
- Developed for use in [Bensmaia Lab](http://bensmaialab.org/).
