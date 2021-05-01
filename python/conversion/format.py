from enum import Enum, auto

import conversion.prophesee
import conversion.ros


class InputFormat(Enum):
    PROPHESEE = auto()
    ROSBAG = auto()

def get_generator(input_format: InputFormat, topic: str='/dvs/events'):
    if input_format == InputFormat.PROPHESEE:
        return conversion.prophesee.ev_generator
    assert input_format == InputFormat.ROSBAG
    return lambda bagpath, delta_t_ms=1000: conversion.ros.ev_generator(bagpath, delta_t_ms=delta_t_ms, topic=topic)
