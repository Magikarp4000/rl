from enum import Enum


class RLSignal(Enum):
    DEFAULT = 0
    VIEW_UPDATE = 1
    TRAIN_CLICKED = 2
    TEST_CLICKED = 3
    STOP_SIMULATION = 4
    FPS_CHANGE = 5
    CLOCK_UPDATE = 6
    SYNC_WITH_AGENT = 7
    TOGGLE_AUTO_SYNC = 8
    VIEW_NEW_EP = 9
    EP_START = 10
    EP_END = 11
    TRAIN_START = 12
    TRAIN_END = 13
