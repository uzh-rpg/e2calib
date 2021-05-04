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

### Conversion to H5
Our current conversion code supports 2 event file formats:
1. Rosbags with [dvs\_msgs](https://github.com/uzh-rpg/rpg_dvs_ros/tree/master/dvs_msgs)
2. Prophesee Raw Format: Metavision 2.2

This requires the use of system Python.
First,
* install [dvs\_msgs](https://github.com/uzh-rpg/rpg_dvs_ros/tree/master/dvs_msgs), if you want to use rosbags.
* install [Metavision 2.2](https://docs.prophesee.ai/2.2.0/installation/index.html), if you want to use prophesee raw files.
Second,
```bash
pip3 install --no-cache-dir -r requirements.txt
pip3 install dataclasses # if your system Python version is < 3.7
```

### Reconstruction

```bash
cuda_version=10.1

conda create -y -n e2calib python=3.7
conda activate e2calib
conda install -y -c anaconda numpy
conda install -y -c conda-forge h5py
conda install pytorch torchvision cudatoolkit=$cuda_version -c pytorch
conda install pandas
conda install -c conda-forge opencv
conda install -c conda-forge matplotlib
conda install -c anaconda scipy

```

The reconstruction code uses events saved in the h5 file format to images using the paper [**High Speed and High Dynamic Range Video with an Event Camera**](http://rpg.ifi.uzh.ch/docs/TPAMI19_Rebecq.pdf)

* Download the pretrained model:
```wget "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar" -O pretrained/E2VID_lightweight.pth.tar```



TODO(Manasi): Move the pretrained model from env variable to path.


## Calibration Procedure

The calibration procedure is based on three steps:
1. Conversion of different event data files into a common hdf5 format.
2. Reconstruction of images at a certain frequency from this file.
3. Calibration using your favorite image-based calibration toolbox.

## Conversion to H5

The [conversion script](https://github.com/uzh-rpg/e2calib_private/blob/main/python/convert.py) simply requires the path to the event file and optionally a ros topic in case of a rosbag.

## Reconstruction

The [reconstruction](https://github.com/uzh-rpg/e2calib_private/blob/wip/manasi/reconstruction/offline_reconstruction.py) requires the h5 file to convert events to frames.
Additionally, you also need to specify the height and width of the event camera and the frequency at which you want to reconstruct the frames.