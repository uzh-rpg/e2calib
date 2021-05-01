from pathlib import Path

#from conversion.format import get_generator
import conversion.format


if __name__ == '__main__':
    proph_generator = conversion.format.get_generator(conversion.format.InputFormat.PROPHESEE)
    ros_generator = conversion.format.get_generator(conversion.format.InputFormat.ROSBAG, topic='/dvs/events')
    print(proph_generator)
    print(ros_generator)

    testfile_prophesee = Path('/home/mathias/Downloads/data.raw')
    testfile_ros = Path('/home/mathias/Downloads/indoor_45_16_davis.bag')

    for event_slice in ros_generator(testfile_ros):
        print(f"t_s = {event_slice['t'][0]}, t_e = {event_slice['t'][-1]}")
    print('----------------------------------------------')
    for event_slice in proph_generator(testfile_prophesee):
        print(f"t_s = {event_slice['t'][0]}, t_e = {event_slice['t'][-1]}")
