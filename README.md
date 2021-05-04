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
2. Prophesee Raw Format: Metavision 2.2

First,
* install [dvs\_msgs](https://github.com/uzh-rpg/rpg_dvs_ros/tree/master/dvs_msgs), if you want to use rosbags.
* install [Metavision 2.2](https://docs.prophesee.ai/2.2.0/installation/index.html), if you want to use prophesee raw files.
Second,
```bash
pip3 install --no-cache-dir -r requirements.txt
pip3 install dataclasses # if your system Python version is < 3.7
```

### Reconstruction
For running the reconstruction code, we create a new conda environment. Use an appropriate cuda version.

```bash
cuda_version=10.1

conda create -y -n e2calib python=3.7
conda activate e2calib
conda install -y -c anaconda numpy scipy
conda install -y -c conda-forge h5py opencv
conda install pytorch torchvision cudatoolkit=$cuda_version -c pytorch

```

The reconstruction code uses events saved in the h5 file format to images using the paper [**High Speed and High Dynamic Range Video with an Event Camera**](http://rpg.ifi.uzh.ch/docs/TPAMI19_Rebecq.pdf)

* Download the pretrained model:
```
mkdir -p python/reconstruction/pretrained
wget "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar" -O python/reconstruction/pretrained/E2VID_lightweight.pth.tar
```

* Download the test data:
ToDo: link the test file


## Calibration Procedure

The calibration procedure is based on three steps:
1. Conversion of different event data files into a common hdf5 format.
2. Reconstruction of images at a certain frequency from this file. Requires the activation of the conda environment `e2calib`.
3. Calibration using your favorite image-based calibration toolbox.

## Conversion to H5

The [conversion script](https://github.com/uzh-rpg/e2calib_private/blob/main/python/convert.py) simply requires the path to the event file and optionally a ros topic in case of a rosbag.

## Reconstruction

The [reconstruction](https://github.com/uzh-rpg/e2calib_private/blob/wip/manasi/python/offline_reconstruction.py) requires the h5 file to convert events to frames.
Additionally, you also need to specify the height and width of the event camera and the frequency at which you want to reconstruct the frames.
To run the image reconstruction code on the test data use the following command:
```
cd python
python offline_reconstruction.py  --freq_hz 5
```
<<<<<<< HEAD
The images will be written by default in the ```frames/reconstruction``` folder.
=======
The images will be written by default in the ```data/reconstruction``` folder.
>>>>>>> 97e7c9fa95367b717835dfa807a119566f43970a
