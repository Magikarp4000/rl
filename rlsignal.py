from enum import Enum


class RLSignal(Enum):
    DEFAULT = 0
    EP_UPDATE = 1
    TRAIN_START = 2
    TRAIN_STOP = 3
