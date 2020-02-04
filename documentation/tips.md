# Tips

This file has assorted tips related to NCams and DeepLabCut.

## Moving DLC network to a different project

1. Change filenames in config.yaml
2. Change filenames in pose_cfg.yaml (two of them)
3. Change directory name `dlc-models/iteration-0/<PROJECT NAME>-trainset95shuffle1`

## Continuing teaching a NN on new videos

1. Edit your config.yaml by replacing the video list with the new videos, save the text referencing the old videos. You may want to change the `numframes2pick` variable in config, too.
2. Extract frames: `deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', crop=False, userfeedback=False)`
3. Label frames: `deeplabcut.label_frames(config_path)`
4. Put back the old videos paths (do not remove the new ones) into the config.yaml.
5. Merge datasets: `deeplabcut.merge_datasets(config_path)`
6. Create training dataset. `deeplabcut.create_training_dataset(config_path)`
7. Go to train and test pose_cfg.yaml of the new interation (e.g. in '<DLC_PROJECT>/dlc-models/iteration-1/CMGPretrainedNetworkDec3-trainset95shuffle1/train/pose_cfg.yaml' and '<DLC_PROJECT>/dlc-models/iteration-1/CMGPretrainedNetworkDec3-trainset95shuffle1/test/pose_cfg.yaml') and change the `init_weights` variable to point to the snapshot from previous network or iteration (for example, '<DLC_PROJECT>\dlc-models\iteration-0\CMGPretrainedNetworkDec3-trainset95shuffle1\train\snapshot-250000' without file extension).
Note: put <DLC_PROJECT> as a full path from root directory or disk.
8. Start training.
