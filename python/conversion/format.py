from pathlib import Path

import conversion.prophesee
import conversion.ros


def get_generator(input_file: Path, delta_t_ms: int=1000, topic: str='/dvs/events'):
    if input_file.suffix == '.raw':
        return lambda: conversion.prophesee.ev_generator(input_file, delta_t_ms=delta_t_ms)
    assert input_file.suffix == '.bag', f'File format {input_file.suffix} is not supported'
    return lambda: conversion.ros.ev_generator(input_file, delta_t_ms=delta_t_ms, topic=topic)
