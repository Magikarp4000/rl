from enum import Enum


class RLSignal(Enum):
    DEFAULT = 0
    EP_UPDATE = 1
    TRAIN_START = 2
    TEST_START = 3
    STOP_SIMULATION = 4
    FPS_CHANGE = 5
    CLOCK_UPDATE = 6
    SYNC_WITH_AGENT = 7
    TOGGLE_AUTO_SYNC = 8
