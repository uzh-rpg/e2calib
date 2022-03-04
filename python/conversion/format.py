from pathlib import Path

dat_conversion_possible = True
raw_conversion_possible = True
ros_found = True
pocolog_found = True
try:
    import conversion.ros
except ImportError:
    print("Conversion from .bag is not possible. If you want to extract .bag files, please install the ROS packages specified in the README.md")
    ros_found = False
try:
    import conversion.prophesee
except ImportError:
    print("Conversion from .raw is not possible. If you want to extract .raw files, please install Metavision 2.2")
    raw_conversion_possible = False
try:
    import conversion.prophesee_dat
except ImportError:
    print("Conversion from .dat is not possible. If you want to extract .dat files, please install Metavision 2.0")
    dat_conversion_possible = False
    metavision_found = False
try:
    import conversion.pocolog
except ImportError:
    print("Conversion from .log is not possible. If you want to extract .log files, please install the ROCK packages specified in the README.md")
    pocolog_found = False


def get_generator(input_file: Path, delta_t_ms: int=1000, topic: str='/dvs/events'):
    if input_file.suffix == '.raw':
        assert raw_conversion_possible, 'Could not find Metavision packages to read .raw file'
        return lambda: conversion.prophesee.ev_generator(input_file, delta_t_ms=delta_t_ms)
    if input_file.suffix == '.dat':
        assert dat_conversion_possible, 'Could not find Metavision packages to read .dat file'
        return lambda: conversion.prophesee_dat.ev_generator(input_file, delta_t_ms=delta_t_ms)
    if input_file.suffix == '.log':
        assert pocolog_found, 'Could not find Rock packages'
        return lambda: conversion.pocolog.ev_generator(input_file, delta_t_ms=delta_t_ms, topic=topic)
    assert input_file.suffix == '.bag', f'File format {input_file.suffix} is not supported'
    assert ros_found, 'Could not not find ROS packages'
    return lambda: conversion.ros.ev_generator(input_file, delta_t_ms=delta_t_ms, topic=topic)
