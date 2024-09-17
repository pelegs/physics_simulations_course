import json

# from pathlib import Path
from sys import argv

import numpy as np
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
        add_pt: bool = True
        for sphere in spheres:
            if distance(pt, sphere[:3]) <= rad + sphere[3]:
                add_pt = False
                break
        if add_pt:
            pts_remaining = np.vstack([pts_remaining, pt[:3]])
    return pts_remaining


if __name__ == "__main__":
    L: list[float] = [120.0, 120.0, 0.0]
    N: list[int] = [30, 30, 1]
    rad: float = 1.0
    grid: npdarr = generate_grid(L, N, rad)
    spheres: npdarr = np.zeros((1, 4))
    spheres[0, :3] = np.array(L) / 2.0
    spheres[0, 3] = 25.0
    pt_radii: npdarr = np.array([5.0] * N[0] * N[1] * N[2])
    grid = remove_spheres_from_grid(grid, spheres, pt_radii)

    output_dict: dict = {
        "time": {
            "dt": float(argv[2]),
            "max_t": float(argv[3]),
        },
        "container": {
            "dimensions": L,
        },
        "particles": list(),
    }
    for id, pt in enumerate(grid):
        vel = np.random.uniform(-100.0, 100.0, size=3)
        vel[2] = 0.0
        output_dict["particles"].append(
            {
                "id": id,
                "pos": pt.tolist(),
                "vel": vel.tolist(),
                "rad": rad,
                "mass": 1.0,
            }
        )
    output_dict["particles"].append(
        {
            "id": len(grid),
            "pos": spheres[0, :3].tolist(),
            "vel": [0.0, 0.0, 0.0],
            "rad": spheres[0, 3],
            "mass": 10.0,
        }
    )

    with open(f"scenes/{argv[1]}.json", "w") as f:
        json.dump(output_dict, f, indent=4)
