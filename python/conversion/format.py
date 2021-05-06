from pathlib import Path

metavision_found = True
ros_found = True
try:
    import conversion.ros
except ImportError:
    print("Conversion from .bag is not possible. If you want to extract .bag files, please install the ROS packages specified in the README.md")
    ros_found = False
try:
    import conversion.prophesee
except ImportError:
    print("Conversion from .raw is not possible. If you want to extract .raw files, please install Metavision 2.2")
    metavision_found = False


def get_generator(input_file: Path, delta_t_ms: int=1000, topic: str='/dvs/events'):
    if input_file.suffix == '.raw':
        assert metavision_found, 'Could not find Metavision packages'
        return lambda: conversion.prophesee.ev_generator(input_file, delta_t_ms=delta_t_ms)
    assert input_file.suffix == '.bag', f'File format {input_file.suffix} is not supported'
    assert ros_found, 'Could not not find ROS packages'
    return lambda: conversion.ros.ev_generator(input_file, delta_t_ms=delta_t_ms, topic=topic)
