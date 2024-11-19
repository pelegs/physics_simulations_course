from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from gravity import rotate

if __name__ == "__main__":
    data = np.load(f"data/{argv[1]}.npz")
    pos = data["pos"]
    vel = data["vel"]
    mass, rad = data["params"]
    e_vec = data["ecc"]
    a, b = data["ellipse"]

    num_steps = pos.shape[0]
    ts = np.linspace(0, 1, num_steps)

    e = np.linalg.norm(e_vec)
    e_hat = e_vec / e
    e_angle = np.arctan2(e_hat[1], e_hat[0])
    pos = rotate(pos.T, -e_angle).T

    th = np.arctan2(pos[:, 1], pos[:, 0])
    sth = np.sin(th)
    cth = np.cos(th)
    E = np.arctan2(cth, sth)
    A = 0.5 * a * b * (E - e * np.sin(E))

    fig, ax = plt.subplots()
    ax.set_title("Test")
    # ax.plot(ts[1:], th[1:])
    # ax.plot(ts[1:], E[1:])
    ax.plot(ts[1:], A[1:])
    plt.show()
