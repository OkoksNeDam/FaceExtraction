from enum import Enum


class ModelOutputs(Enum):
    INCORRECT_RESOLUTION = 0,
    PEOPLE_ABSENCE = 1,
    SEVERAL_PEOPLE = 2,
    CLOSED_FACE = 3
