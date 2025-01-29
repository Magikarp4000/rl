from enum import Enum


class RLSignal(Enum):
    DEFAULT = 0
    EP_UPDATE = 1
    TRAIN_START = 2
    TRAIN_STOP = 3
    FPS_CHANGE = 4
    CLOCK_UPDATE = 5
    SYNC_WITH_AGENT = 6
    TOGGLE_AUTO_SYNC = 7
