import json
from pathlib import Path

#
import numpy as np
import numpy.typing as npt
from ideal_gas import distance, npdarr


def generate_grid(L: list[float], N: list[int], dw: float) -> npdarr:
    pts: list[npdarr] = list()
    for dim in range(3):
        pts.append(np.linspace(0 + dw, L[dim] - dw, N[dim]))
    return np.vstack(np.meshgrid(pts[0], pts[1], pts[2])).reshape(3, -1).T


def remove_spheres_from_grid(
    grid: npdarr, spheres: npdarr, pt_radii: npdarr
) -> npdarr:
    pts_remaining: npdarr = np.empty((0, 3), dtype=np.float64)
    for pt, rad in zip(grid, pt_radii):
        for sphere in spheres:
            if distance(pt, sphere[:3]) > rad + sphere[3]:
                pts_remaining = np.vstack([pts_remaining, pt[:3]])
    return pts_remaining


if __name__ == "__main__":
    grid: npdarr = generate_grid([200.0, 200.0, 0.0], [10, 10, 1], 5.0)
    spheres: npdarr = np.array([[50.0, 50.0, 0.0, 50.0]])
    pt_radii: npdarr = np.array([5.0] * 12)
    grid = remove_spheres_from_grid(grid, spheres, pt_radii)
    np.save("tests/grid_with_hole_1.npy", grid)
