from sys import argv

import matplotlib.pyplot as plt
import numpy as np

data = np.load(f"data/{argv[1]}.npz")
speeds = np.linalg.norm(data["vel"], axis=1)
dists = np.linalg.norm(data["pos"], axis=1)

M = 1.0e7
a = M * dists / (2 * M - dists * speeds**2)
A = np.mean(a)
vv = np.sqrt(M * (2 / dists - 1 / A))

fig, ax = plt.subplots()
ax.set_xlabel("r", fontsize=10)
ax.set_ylabel("v", rotation=0, fontsize=10)

ax.plot(vv, dists, linewidth=5, c="#6BBFFF")
ax.plot(speeds, dists, linewidth=1, c="black")
plt.show()
