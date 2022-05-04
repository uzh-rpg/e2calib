# E2Calib: How to Calibrate Your Event Camera

<p align="center">
   <img src="http://rpg.ifi.uzh.ch/img/papers/CVPRW21_Muglikar.png" height="300"/>
</p>

This repository contains code that implements video reconstruction from event data for calibration as described in the paper [Muglikar et al. CVPRW'21](http://rpg.ifi.uzh.ch/docs/CVPRW21_Muglikar.pdf).

If you use this code in an academic context, please cite the following work:

[Manasi Muglikar](https://manasi94.github.io/), [Mathias Gehrig](https://magehrig.github.io/), [Daniel Gehrig](https://danielgehrig18.github.io/), [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html), "How to Calibrate Your Event Camera", Computer Vision and Pattern Recognition Workshops (CVPRW), 2021

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
Our current conversion code supports 3 event file formats:
1. Rosbags with [dvs\_msgs](https://github.com/uzh-rpg/rpg_dvs_ros/tree/master/dvs_msgs)
2. Pocolog with [base/samples/EventArray](https://github.com/rock-core/base-orogen-types)
3. Prophesee raw format using [Metavision 2.2](https://docs.prophesee.ai/2.2.0/installation/index.html)
4. Prophesee dat format using [Metavision 2.X](https://docs.prophesee.ai/stable/data_formats/file_formats/dat.html)

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


### Image Reconstruction
For running the reconstruction code, we create a new conda environment. Use an appropriate cuda version.

```bash
cuda_version=10.1

conda create -y -n e2calib python=3.7
conda activate e2calib
conda install -y -c anaconda numpy scipy
conda install -y -c conda-forge h5py opencv tqdm
conda install -y -c pytorch pytorch torchvision cudatoolkit=$cuda_version

pip install python/ # this installs e2vid
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

### Conversion to H5 from ROS bags

The [conversion script](https://github.com/uzh-rpg/e2calib/blob/main/python/convert.py) simply requires the path to the event file and optionally a ros topic in case of a rosbag.

If you have an older Metavision version (for example Metavision 2.0), first convert the '.raw' files to '.dat' and then convert it to h5 file format. 

*Note* : The '.dat' file format takes up more space on the disk but for metavision 2.0, the python API can only read '.dat' format.

### Conversion to H5 from Pocolog/Rock files

Pocolog is the file format for logging data in [Rock](https://www.rock-robotics.org/). It is equivalent to the bag format in [ROS](https://www.ros.org/). More specifically [Pocolog](https://github.com/rock-core/tools-pocolog) is the tool to manupilate [log](https://github.com/rock-core/tools-logger) files.

The [conversion script](https://github.com/uzh-rpg/e2calib/blob/main/python/convert.py) understands the Pocolog file format from the pocolog [conversion](https://github.com/uzh-rpg/e2calib/blob/main/python/conversion/pocolog.py) file. You need a Rock installation with the Pocolog python bindings [Pocolog Pybind](https://github.com/jhidalgocarrio/tools-pocolog_pybind) installed in order to convert events in pocolog to h5 format. Please follow the installation guide to install Rock on your system: [How to install Rock](https://www.rock-robotics.org/documentation/installation.html). Afterwards clone the Pocolog Python bindings:

```bash
git clone git@github.com:jhidalgocarrio/tools-pocolog_pybind.git <rock-path>/tools/pocolog_pybind
```

Compile and install the Pocolog Python bindings:

```bash
source <rock-path>/env.sh
cd <rock-path>/tools/pocolog_pybind
amake
python3 -m pip install <rock-path>/tools/pocolog_pybind
```

You can now simply use the [conversion script](https://github.com/uzh-rpg/e2calib/blob/main/python/convert.py) with the path to the pocolog file and the port name  (i.e.: similar to ros topic name) containing the events (e.g.: --topic /camera_prophesee.events).

```bash
python convert.py <pocolog_file> -t <port_name>
```

If you don't know the port names in a log file just run pocolog (i.e.: similar to rosbag info):

```bash
source <rock-path>/env.sh
pocolog <pocolog_file>
```

You can also use our [Dockerfile](https://download.ifi.uzh.ch/rpg/e2calib/pocolog/Dockerfile) to create a docker image with all the necessary tools. You should be able to convert from pocolog to rosbag and h5 format out-of-the-box. To build the docker image run:

```bash
docker build -t <image_name> -f Dockerfile .
```

### Reconstruction

The [reconstruction](https://github.com/uzh-rpg/e2calib/blob/main/python/offline_reconstruction.py) requires the h5 file to convert events to frames.
Additionally, you also need to specify the height and width of the event camera and the frequency or timestamps at which you want to reconstruct the frames.
As an example, to run the image reconstruction code on one of the example files use the following command:
```bash
  cd python
  python offline_reconstruction.py  --h5file file --freq_hz 5 --upsample_rate 4 --height 480 --width 640 
```

The images will be written by default in the ```python/frames/e2calib``` folder.

#### Fixed Frequency

Reconstruction can be performed at a fixed frequency. This is useful for intrinsic calibration. The argument `--freq_hz` specifies the frequency at which the image reconstructions will be saved.

#### Specified Timestamps

You can also specify the timestamps for image reconstruction from a text file. As an example, these timestamps can be trigger signals that synchronize the event camera with the exposure time of a frame-based camera. In this scenario, you may want to reconstruct images from the event camera at the trigger timestamps for extrinsic calibration. The argument `--timestamps_file` must point to a text file containing the timestamps in microseconds for this option to take effect.

We provide a script to [extract trigger signals from a prophesee raw file](python/extract_triggers_prophesee.py).

#### Upsampling

We provide the option to multiply the reconstruction rate by a factor via the `--upsample_rate` argument. For example, setting this value to 3 will lead to 3 times higher reconstruction rate but does not influence the final number of reconstructed images that will be saved. This parameter can be used to finetune the reconstruction performance. For example setting `--freq_hz` to 5 without upsampling can lead to suboptimal performance because too many events are fed to E2VID. Instead, it is often a good start to work with 20 Hz reconstruction, thus setting the upsampling rate to 4.


### Calibration

Once the reconstructed images are ready, you can use any image calibration toolbox.
We provide a [script](python/images_to_rosbag.py) to convert the reconstructed images to rosbag, that can be used with [kalibr](https://github.com/ethz-asl/kalibr) calibration toolbox for intrinsic calibration. Please use this script outside the conda environment.
```bash
cd python
python3 images_to_rosbag.py --rosbag_folder python/frames/ --image_folder  python/frames/e2calib --image_topic /dvs/image_reconstructed
```

In case you would like to combine images with other sensors for extrinsics calibration, please take a look at the [kalibr bagcreator script](https://github.com/ethz-asl/kalibr/wiki/bag-format#bagcreater) 

Here are instructions to run kalibr in docker, using the [following docker image](https://hub.docker.com/r/stereolabs/kalibr). The following instructions are copied from there:

```bash
folder=path/to/calibration/bag/
bagname=name_of.bag
targetname=target.yaml
topic1=/cam0/image_raw 
topic2=/cam1/image_raw

sudo snap install docker
sudo docker pull stereolabs/kalibr
xhost +local:root
sudo docker run -it -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -v "$folder:/calib" stereolabs/kalibr:kinetic
kalibr_calibrate_cameras --bag /calib/$bagname --target /calib/$targetname --models 'pinhole-radtan' 'pinhole-radtan' --topics $topic1 $topic2
```

## Example Files
For each file, we provide the original event file format (raw or rosbag) but also the already converted h5 file.

### Prophesee Gen 3
**Without Triggers:**
```bash
wget https://download.ifi.uzh.ch/rpg/e2calib/prophesee/without_triggers/data.raw
wget https://download.ifi.uzh.ch/rpg/e2calib/prophesee/without_triggers/data.h5
```
*Reconstruction Example*

To reconstruct images from events at a fixed frequency, you can follow this example command:
```bash
  conda activate e2calib
  cd python
  python offline_reconstruction.py  --freq_hz 10 --upsample_rate 2 --h5file data.h5 --output_folder gen3_no_trigger --height 480 --width 640
```
![Sample reconstruction](img/gen3_no_trigger_0000000001700066000.png?raw=true)

**With Triggers:**

We also extracted the trigger signals using the provided [script](python/extract_triggers_prophesee.py) and provide them in the `triggers.txt` file.
```bash
wget https://download.ifi.uzh.ch/rpg/e2calib/prophesee/with_triggers/data.raw
wget https://download.ifi.uzh.ch/rpg/e2calib/prophesee/with_triggers/data.h5
wget https://download.ifi.uzh.ch/rpg/e2calib/prophesee/with_triggers/triggers.txt
```
*Reconstruction Example*

To reconstruct images from events at the trigger time, you can follow this example command:
```bash
  conda activate e2calib
  cd python
  python offline_reconstruction.py  --upsample_rate 2 --h5file data.h5 --output_folder gen3_with_trigger/ --timestamps_file triggers.txt --height 480 --width 640
```

### Samsung Gen 3
**Without Triggers:**
```bash
wget https://download.ifi.uzh.ch/rpg/e2calib/samsung/samsung.bag
wget https://download.ifi.uzh.ch/rpg/e2calib/samsung/samsung.h5
```
*Reconstruction Example*

To reconstruct images from events at fixed frequency, you can follow this example command:
```bash
  conda activate e2calib
  cd python
  python offline_reconstruction.py --freq_hz 5 --upsample_rate 4 --h5file samsung.h5 --output_folder samsung_gen3 --height 480 --width 640
```
