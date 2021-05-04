from pathlib import Path

import conversion.ros
metavision_found = True
try:
    import conversion.prophesee
except ImportError:
    print("Conversion from .raw is not possible. If you want to extract .raw files please first install Metavision 2.1/2.2")
    metavision_found = False


def get_generator(input_file: Path, delta_t_ms: int=1000, topic: str='/dvs/events'):
    if input_file.suffix == '.raw':
        assert metavision_found, "Conversion from .raw is not possible. Please first install Metavision 2.1/2.2"
        return lambda: conversion.prophesee.ev_generator(input_file, delta_t_ms=delta_t_ms)
    assert input_file.suffix == '.bag', f'File format {input_file.suffix} is not supported'
    return lambda: conversion.ros.ev_generator(input_file, delta_t_ms=delta_t_ms, topic=topic)
