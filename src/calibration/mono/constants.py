from enum import IntEnum


class FixedPointMode(IntEnum):
    NONE = -1

    TOP_LEFT = -1000
    TOP_RIGHT = -999
    BOTTOM_LEFT = -1001
    BOTTOM_RIGHT = -1002

    MIDDLE = -1003

    DEFAULT = TOP_RIGHT
