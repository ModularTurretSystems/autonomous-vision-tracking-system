"""
src/targeting/selector.py

Author
------
KrutayaBabka : initial implementation (2026-02-12)

Description
-----------
Target selection module for multi-object tracking results.

Provides the `TargetSelector` class, which applies semantic strategies
to select one or more objects from a `TrackedObjects` frame based on
criteria such as proximity to frame center, bounding box area, etc.

The selected objects are returned as `SelectedObject` instances containing
bounding box, class info, and computed selection metrics (raw and normalized).
"""


import numpy as np

from .constants import SelectorStrategy, SelectorQuantity

from cv2.typing import Point2f
from typing import List, Optional, Callable
from .types import SelectedObject
from src.tracking.types import TrackedObjects


class TargetSelector:
    """
    High-level object selector for a single frame.

    Applies a selection strategy to prioritize objects based on user-defined
    criteria and returns the top N objects.

    Parameters
    ----------
    strategy : SelectorStrategy
        The strategy to use when selecting targets.
    center : Point2f, optional
        Reference center for distance-based strategies. Defaults to frame center.
    max_number : int, optional
        Maximum number of objects to select. Negative or zero values mean "all".

    Attributes
    ----------
    strategy : SelectorStrategy
        Currently applied selection strategy.
    strategy_method : Callable[[TrackedObjects], List[SelectedObject]]
        Internal function implementing the selection logic for the strategy.
    center : Point2f | None
        Reference center for distance calculations.
    max_number : int
        Maximum number of selected objects.
    """

    def __init__(
        self,
        strategy: SelectorStrategy,
        *,
        center: Optional[Point2f] = None,
        max_number: int = SelectorQuantity.ALL
    ) -> None:
        self.change_strategy(strategy=strategy)
        self.center = center
        self.max_number = SelectorQuantity.ALL if max_number <= 0 else max_number


    def select(self, objects: TrackedObjects) -> List[SelectedObject]:
        """
        Select objects from a `TrackedObjects` container according to the current strategy.

        Parameters
        ----------
        objects : TrackedObjects
            Frame results containing detected/tracked objects.

        Returns
        -------
        List[SelectedObject]
            List of selected objects sorted according to the strategy.
            Each object contains:
            - `metric_raw`: raw selection metric (distance or area)
            - `metric_norm`: normalized metric in [0, 1]
            - `center`: geometric center of the object

        Notes
        -----
        If `max_number` is set to ALL (-1), all objects are returned.
        Selection strategies normalize metrics relative to frame dimensions or
        maximum area.
        """

        if self.max_number == SelectorQuantity.ALL: self.max_number = len(objects.objects) + 1
        return self.strategy_method(objects)[:self.max_number]

    
    def change_strategy(self, strategy: SelectorStrategy):
        """
        Change the selection strategy at runtime.

        Parameters
        ----------
        strategy : SelectorStrategy
            The new strategy to apply.
        """

        self.strategy_method = self._select_strategy_method(strategy=strategy)
        self.strategy = strategy


    def _select_strategy_method(self, strategy: SelectorStrategy) -> Callable[[TrackedObjects], List[SelectedObject]]:
        """
        Map a strategy enum to the corresponding internal method.

        Parameters
        ----------
        strategy : SelectorStrategy
            Strategy to map.

        Returns
        -------
        Callable[[TrackedObjects], List[SelectedObject]]
            Function implementing the selection logic.

        Raises
        ------
        ValueError
            If the strategy is unknown.
        """

        if   strategy == SelectorStrategy.CLOSEST_TO_CENTER:  return self._distance_to_center
        elif strategy == SelectorStrategy.FURTHEST_TO_CENTER: return self._furthest_to_center
        elif strategy == SelectorStrategy.SMALLEST:           return self._smallest_box
        elif strategy == SelectorStrategy.LARGEST:            return self._largest_box

        raise ValueError(
            f"Unknown strategy={self.strategy}"
        )


    def _distance_to_center(self, objects: TrackedObjects, *, reverse: bool = False) -> List[SelectedObject]:
        """
        Compute distance-based metrics to select objects relative to a reference point.

        Parameters
        ----------
        objects : TrackedObjects
            Objects to evaluate.
        reverse : bool, optional
            If True, sorts in descending distance order (furthest first).

        Returns
        -------
        List[SelectedObject]
            Objects sorted by distance to the reference point, with raw and normalized metrics.
        """

        if len(objects.objects) == 0: return []

        xyxy = objects.raw_data
        if xyxy is None: return []

        xyxy = xyxy[:, 0:4]
        centers = (xyxy[:, 0:2] + xyxy[:, 2:4]) * 0.5

        if self.center is None:
            frame_center = np.array(objects.orig_shape[::-1]) * 0.5 
        else:
            frame_center = np.array(self.center, dtype=np.float32)

        deltas = centers - frame_center
        dists = np.linalg.norm(deltas, axis=1)
        max_dist = np.linalg.norm(frame_center)
        norm_dists = dists / max_dist

        items: List[SelectedObject] = []
        for obj, center, distance_raw, distance_norm in zip(objects.objects, centers, dists, norm_dists):
            items.append(
                SelectedObject.from_Object(
                    object=obj, 
                    center=center, 
                    metric_raw=distance_raw,
                    metric_norm=distance_norm
                )
            ) 

        items.sort(key=lambda x: x.metric_norm, reverse=reverse)

        return items
        
    
    def _box_area(self, objects: TrackedObjects, *, reverse: bool = False) -> List[SelectedObject]:
        """
        Compute area-based metrics to select objects by bounding box size.

        Parameters
        ----------
        objects : TrackedObjects
            Objects to evaluate.
        reverse : bool, optional
            If True, sorts in descending area order (largest first).

        Returns
        -------
        List[SelectedObject]
            Objects sorted by bounding box area, with raw and normalized metrics.
        """
        
        if len(objects.objects) == 0: return []

        xyxy = objects.raw_data
        if xyxy is None: return []
        
        xyxy = xyxy[:, 0:4]

        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        areas = widths * heights

        max_area = np.max(areas)
        norm_areas = areas / max_area

        centers = (xyxy[:, 0:2] + xyxy[:, 2:4]) * 0.5

        items: List[SelectedObject] = []
        for obj, center, area_raw, area_norm in zip(objects.objects, centers, areas, norm_areas):
            items.append(
                SelectedObject.from_Object(
                    object=obj,
                    center=center,
                    metric_raw=area_raw,
                    metric_norm=area_norm
                )
            )

        items.sort(key=lambda x: x.metric_norm, reverse=reverse)

        return items
    

    def _closest_to_center(self, objects: TrackedObjects) -> List[SelectedObject]: 
        return self._distance_to_center(objects=objects)      
    

    def _furthest_to_center(self, objects: TrackedObjects) -> List[SelectedObject]: 
        return self._distance_to_center(objects=objects, reverse=True)
    

    def _smallest_box(self, objects: TrackedObjects) -> List[SelectedObject]:
        return self._box_area(objects=objects)


    def _largest_box(self, objects: TrackedObjects) -> List[SelectedObject]:
        return self._box_area(objects=objects, reverse=True)
    