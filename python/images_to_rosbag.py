import rosbag
import cv2
from cv_bridge import CvBridge
from os.path import join
import rospy
import argparse
import os
import glob
import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag_folder", required=True,
                        type=str, help="Path to the base folder containing the rosbags")
    parser.add_argument("--image_folder", required=True,
                        type=str, help="Path to the base folder containing the image reconstructions")
    parser.add_argument("--image_topic", required=True, type=str,
                        help="Name of the topic which will contain the reconstructed images")
    parser.set_defaults(feature=False)

    args = parser.parse_args()

    print('Images to process: {}'.format(args.image_folder))
    reconstructed_images_folder = args.image_folder

    bridge = CvBridge()
    
    if not os.path.exists(args.rosbag_folder):
        os.makedirs(args.rosbag_folder)

    input_bag_filename = join(args.rosbag_folder, 'reconstruction.bag')
    if os.path.exists(input_bag_filename):
        print('Detected existing rosbag: {}.'.format(input_bag_filename))
        print('Will overwrite the existing bag.')

    # list all images in the folder
    images = [f for f in glob.glob(join(reconstructed_images_folder, "*.png"))]
    images = sorted(images)
    print('Found {} images'.format(len(images)))

    pbar = tqdm.tqdm(total=len(images))
    with rosbag.Bag(input_bag_filename, 'w') as outbag:
        for i, image_path in enumerate(images):
            stamp = image_path.split('/')[-1].split('.')[0]
            assert os.path.exists(image_path)==True
            img = cv2.imread(image_path, 0)

            try:
                img_msg = bridge.cv2_to_imgmsg(img, encoding='mono8')
                stamp_ros = rospy.Time(secs=int(stamp[0:-9]), nsecs=int(stamp[-9:]) )
                img_msg.header.stamp = stamp_ros
                img_msg.header.seq = i
                outbag.write(args.image_topic, img_msg, img_msg.header.stamp)
                pbar.update(1)
            except:
                print("error in reading file ", image_path)