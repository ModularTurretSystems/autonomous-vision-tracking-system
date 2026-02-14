"""
src/targeting/constants.py

Author
------
KrutayaBabka : initial implementation (2026-02-14)

Description
-----------
Semantic constants for target selection in the multi-object tracking and targeting
subsystem. These enums define strategies for selecting objects of interest and
specify how many targets should be selected.

They are used by selection modules to drive decision-making logic, e.g.,
prioritizing targets for aiming, display, or tracking.
"""


from enum import IntEnum


# =========================
# Target selection strategies
# =========================

class SelectorStrategy(IntEnum):
    """
    Strategies for selecting objects among multiple detected/tracked targets.

    Attributes
    ----------
    CLOSEST_TO_CENTER : int
        Select the object whose bounding box center is closest to the image/frame center.
    FURTHEST_TO_CENTER : int
        Select the object whose bounding box center is furthest from the image/frame center.
    LARGEST : int
        Select the object with the largest bounding box area.
    SMALLEST : int
        Select the object with the smallest bounding box area.
    NEAREST : int
        Select the object nearest in real-world distance (if available from depth data).
    """

    CLOSEST_TO_CENTER  = 1
    FURTHEST_TO_CENTER = 2
    LARGEST            = 3
    SMALLEST           = 5
    NEAREST            = 8


# =========================
# Target selection quantity
# =========================

class SelectorQuantity(IntEnum):
    """
    Specifies how many objects to select from the available detected/tracked targets.

    Attributes
    ----------
    ALL : int
        Select all detected/tracked objects (-1 as a sentinel value).
    ONE : int
        Select a single target (usually the highest-priority according to the strategy).
    """
    ALL  = -1
    ONE  = 1
