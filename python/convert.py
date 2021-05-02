from pathlib import Path

import conversion.format
import conversion.h5writer


if __name__ == '__main__':
    proph_generator = conversion.format.get_generator(conversion.format.InputFormat.PROPHESEE)
    ros_generator = conversion.format.get_generator(conversion.format.InputFormat.ROSBAG, topic='/dvs/events')

    h5writer_proph = conversion.h5writer.H5Writer(Path('proph.h5'))
    h5writer_ros = conversion.h5writer.H5Writer(Path('ros.h5'))

    testfile_prophesee = Path('/home/mathias/Documents/projects/cvprw21/opensource/data/conversion_test/proph.raw')
    testfile_ros = Path('/home/mathias/Documents/projects/cvprw21/opensource/data/conversion_test/ros.bag')

    for event_slice in ros_generator(testfile_ros):
        h5writer_ros.add_data(
                event_slice['x'],
                event_slice['y'],
                event_slice['p'],
                event_slice['t'])
        print(f"t_s = {event_slice['t'][0]}, t_e = {event_slice['t'][-1]}")
    print('----------------------------------------------')
    for event_slice in proph_generator(testfile_prophesee):
        h5writer_proph.add_data(
                event_slice['x'],
                event_slice['y'],
                event_slice['p'],
                event_slice['t'])
        print(f"t_s = {event_slice['t'][0]}, t_e = {event_slice['t'][-1]}")
