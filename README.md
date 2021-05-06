# E2Calib: How to Calibrate Your Event Camera

This repository contains code that implements video reconstruction from event data for calibration as described in the paper [Muglikar et al. CVPRW'21](http://rpg.ifi.uzh.ch/docs/CVPRW21_Muglikar.pdf).

If you use this code in an academic context, please cite the following work:

[Manasi Muglikar](http://manasi94.github.io/), Mathias Gehrig, [Daniel Gehrig](https://danielgehrig18.github.io/), [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html), "How to Calibrate Your Event Camera", Computer Vision and Pattern Recognition Workshops (CVPRW), 2021

```bibtex
@InProceedings{Muglikar2021CVPR,
  author = {Manasi Muglikar and Mathias Gehrig and Daniel Gehrig and Davide Scaramuzza},
  title = {How to Calibrate Your Event Camera},
  booktitle = {{IEEE} Conf. Comput. Vis. Pattern Recog. Workshops (CVPRW)},
  month = {June},
  year = {2021}
}
```

## Installation
The installation procedure is divided into two parts.
First, installation of packages for the conversion code that must be completed outside of any virtual environment for compatibility reasons.
Second, installation of packages in a conda environment to run the reconstruction code.

### Conversion to H5
Our current conversion code supports 2 event file formats:
1. Rosbags with [dvs\_msgs](https://github.com/uzh-rpg/rpg_dvs_ros/tree/master/dvs_msgs)
2. Prophesee raw format using [Metavision 2.2](https://docs.prophesee.ai/2.2.0/installation/index.html)

Regardeless of the event file format:
```bash
pip3 install --no-cache-dir -r requirements.txt
pip3 install dataclasses # if your system Python version is < 3.7
```

* If you want to convert Prophesee raw format, install [Metavision 2.2](https://docs.prophesee.ai/2.2.0/installation/index.html).
* If you want to convert Rosbags, install:

```bash
pip3 install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
```

### Reconstruction
For running the reconstruction code, we create a new conda environment. Use an appropriate cuda version.

```bash
cuda_version=10.1

conda create -y -n e2calib python=3.7
conda activate e2calib
conda install -y -c anaconda numpy scipy
conda install -y -c conda-forge h5py opencv tqdm
conda install pytorch torchvision cudatoolkit=$cuda_version -c pytorch

```

The reconstruction code uses events saved in the h5 file format to reconstruct images with [E2VID](http://rpg.ifi.uzh.ch/docs/TPAMI19_Rebecq.pdf).

### Reconstructions to Rosbag
If you want to use [kalibr](https://github.com/ethz-asl/kalibr), you may want to create a rosbag from the reconstructed images.
To achieve this, additionally install (outside of the conda environment)

```bash
pip3 install tqdm opencv-python
pip3 install --extra-index-url https://rospypi.github.io/simple/ sensor-msgs
```

## Calibration Procedure

The calibration procedure is based on three steps:
1. Conversion of different event data files into a common hdf5 format.
2. Reconstruction of images at a certain frequency from this file. Requires the activation of the conda environment `e2calib`.
3. Calibration using your favorite image-based calibration toolbox.

### Conversion to H5

The [conversion script](https://github.com/uzh-rpg/e2calib_private/blob/main/python/convert.py) simply requires the path to the event file and optionally a ros topic in case of a rosbag.

### Reconstruction

The [reconstruction](https://github.com/uzh-rpg/e2calib_private/blob/wip/manasi/python/offline_reconstruction.py) requires the h5 file to convert events to frames.
Additionally, you also need to specify the height and width of the event camera and the frequency at which you want to reconstruct the frames.
To run the image reconstruction code on the test data use the following command:
```
  cd python
  python offline_reconstruction.py  --h5file file --freq_hz 5 --height 480 --width 640 
```

The images will be written by default in the ```python/frames/e2calib``` folder.

#### Parameters
Main parameters

* ```--freq_hz``` (default 5) Frequency of reconstructed images. This is the parameter that has most influence on the image reconstruction quality.
The default frequency is 5Hz. This value may need to be adapted depending on the scene dynamics to improve the quality of the reconstruction.
* ```--upsample_rate``` (default 1) Reconstruct frames at intermediate times. This parameter reconstructs frames at intermediate times between frequency window.
The intermediate frames are not saved. This parameter can be adjusted in a scenario where you cannot increase the reconstruction frequency (due to synchronization with an external sensor) but would like to improve the quality of reconstructions. 

## Example Files
For each file, we provide the original event file format (raw or rosbag) but also the already converted h5 file.

### Prophesee Gen 3
*Without Triggers:*
```bash
wget https://download.ifi.uzh.ch/rpg/e2calib/prophesee/without_triggers/data.raw
wget https://download.ifi.uzh.ch/rpg/e2calib/prophesee/without_triggers/data.h5
```

*With Triggers:*
```bash
wget https://download.ifi.uzh.ch/rpg/e2calib/prophesee/with_triggers/data.raw
wget https://download.ifi.uzh.ch/rpg/e2calib/prophesee/with_triggers/data.h5
```

### Samsung Gen 3
*Without Triggers*
```bash
wget https://download.ifi.uzh.ch/rpg/e2calib/samsung/samsung.bag
wget https://download.ifi.uzh.ch/rpg/e2calib/samsung/samsung.h5
```
