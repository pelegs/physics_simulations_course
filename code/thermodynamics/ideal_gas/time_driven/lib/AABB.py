import numpy as np

from .constants import AXES, npdarr, npiarr


class AABB:
    """Docstring for AABB."""

    def __init__(self, obj: object, pts: npdarr) -> None:
        self.obj: object = obj
        self.pts: npdarr = pts

    def move(self, dr: npdarr) -> None:
        self.pts = self.pts + dr


class SweepPrune:
    """Docstring for SweepPrune."""

    def __init__(self, AABB_list: list[AABB]) -> None:
        self.AABB_list: list[AABB] = AABB_list

        # Setup overlap stuff
        self.num_AABBs: int = len(AABB_list)
        self.reset_overlaps()

    def reset_overlaps(self) -> None:
        self.overlap_matrix: npiarr = np.zeros(
            (self.num_AABBs, self.num_AABBs),
            dtype=int,
        )
        self.overlap_matrix: npiarr = np.zeros(
            (3, self.num_AABBs, self.num_AABBs),
            dtype=int,
        )

    def check_axis_overlaps(self, axis: int) -> npiarr:
        pass

    def calc_overlaps(self):
        self.reset_overlaps()
        # for axis in AXES:
        #     self.overlap_matrix[axis] =
