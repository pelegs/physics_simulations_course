import numpy as np
from prettytable import PrettyTable


def estimate_pi(num_points=1000):
    points = np.random.uniform(size=(num_points, 2))
    pts_inside_rad = points[np.where(np.linalg.norm(points, axis=1) <= 1.0)[0]]
    pi_est = 4 * pts_inside_rad.shape[0] / num_points
    return pi_est


table = PrettyTable()
table.field_names = ["n", "π (est.)", "Δ"]
table.align = "l"


if __name__ == "__main__":
    for n in [
        5,
        10,
        50,
        100,
        500,
        1000,
        5000,
        10000,
        50000,
        100000,
        500000,
        1000000,
        5000000,
        10000000,
        50000000,
    ]:
        pi_est = estimate_pi(n)
        table.add_row(
            [f"{n}", f"{pi_est:0.4f}", f"{abs(pi_est - np.pi):0.4f}"]
        )
    print(table)
