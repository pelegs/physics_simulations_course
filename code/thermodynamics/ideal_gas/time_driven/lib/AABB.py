from copy import deepcopy

import numpy as np
from lib.constants import UNION, Axes, Pts, npdarr, npiarr


class AABB:
    """Docstring for AABB."""

    def __init__(self, obj: object, pts: npdarr) -> None:
        self.id: int = -1
        self.obj: object = obj
        self.pts: npdarr = pts
        self.center = np.average(self.pts, axis=0).flatten()
        self.sides = np.diff(self.pts, axis=0).flatten()
        self.diff_sides_center = self.pts - self.center

    def set_id(self, id: int) -> None:
        self.id = id

    def set_pos(self, new_pos) -> None:
        self.center = new_pos
        self.pts = self.center + self.diff_sides_center

    def __str__(self) -> str:
        return f"id: {self.id}, LLF: {self.pts[0]}, URB: {self.pts[1]}"


def AABB_order(axis: int):
    def order_axis(bbox: AABB):
        return bbox.pts[Pts.LLF, axis]

    return order_axis


class SweepPruneSystem:
    """Docstring for SweepPrune."""

    overlap_ids: npiarr

    def __init__(self, AABB_list: list[AABB]) -> None:
        self.AABB_list: list[AABB] = AABB_list

        # Setup overlap stuff
        self.assign_ids()
        self.num_AABBs: int = len(AABB_list)
        self.AABB_sorted: list[list[AABB]] = [
            deepcopy(self.AABB_list),
            deepcopy(self.AABB_list),
            deepcopy(self.AABB_list),
        ]
        self.reset_overlaps()

    def assign_ids(self):
        for id, bbox in enumerate(self.AABB_list):
            bbox.set_id(id)

    def reset_overlaps(self) -> None:
        self.overlap_matrix: npiarr = np.zeros(
            (4, self.num_AABBs, self.num_AABBs),
            dtype=int,
        )

    def check_axis_overlaps(self, axis: int) -> None:
        self.AABB_sorted[axis] = sorted(self.AABB_list, key=AABB_order(axis))
        for bbox1_id, bbox1 in enumerate(self.AABB_sorted[axis]):
            for bbox2 in self.AABB_sorted[axis][bbox1_id + 1 :]:
                if bbox1.pts[Pts.RHB, axis] >= bbox2.pts[Pts.LLF, axis]:
                    self.overlap_matrix[axis, bbox1.id, bbox2.id] = 1
                    self.overlap_matrix[axis, bbox2.id, bbox1.id] = 1
                else:
                    break

    def set_full_overlaps(self) -> None:
        # reduce() is used because there are three overlap matrices
        self.overlap_matrix[UNION] = np.triu(
            np.logical_and.reduce(self.overlap_matrix[:UNION])
        )
        self.overlap_ids = np.vstack(np.where(self.overlap_matrix[UNION])).T

    def calc_overlaps(self) -> npiarr:
        self.reset_overlaps()
        for axis in Axes:
            self.check_axis_overlaps(axis)
        self.set_full_overlaps()
        return self.overlap_ids


if __name__ == "__main__":
    # Testing overlaps with a visualization
    from random import randint

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    # Create n circles, each centered in a random position
    # within a certain area (xmin, ymin), (xmax, ymax)
    xmin, xmax = 0, 400
    ymin, ymax = 0, 400
    zmin, zmax = 0, 0
    num_circles = 20
    centers = np.random.uniform(
        (xmin, ymin, zmin), (xmax, ymax, zmax), (num_circles, 3)
    )
    radii = np.random.uniform(10, 25, num_circles)
    bbox_list = [
        AABB(obj=1, pts=np.array([center - radius, center + radius]))
        for center, radius in zip(centers, radii)
    ]
    aabb_system = SweepPruneSystem(bbox_list)
    aabb_system.calc_overlaps()

    # Draw figure
    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("y", rotation=0, fontsize=15)

    intersected_set = set(
        [
            aabb_system.AABB_list[idx].id
            for pair in aabb_system.overlap_ids
            for idx in pair
        ]
    )

    for center, radius, bbox in zip(centers, radii, bbox_list):
        circle_fill_color = "#00BBFF"
        if bbox.id in intersected_set:
            circle_fill_color = f'{f"#{randint(0, 256**3):06x}"}'.upper()

        circle = Circle(
            center,
            radius,
            lw=1,
            edgecolor="black",
            facecolor=circle_fill_color,
        )
        ax.add_patch(circle)
        rect = Rectangle(
            bbox.pts[0],
            bbox.sides[Axes.X],
            bbox.sides[Axes.Y],
            lw=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.annotate(f"{bbox.id}", center[: Axes.Z])

    ax.annotate(
        ",".join([f"({p1},{p2})" for p1, p2 in aabb_system.overlap_ids]),
        (10, 10),
    )

    plt.show()
