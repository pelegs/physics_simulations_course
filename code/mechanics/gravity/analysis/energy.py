from sys import argv

import matplotlib.pyplot as plt
import numpy as np

# Data
data = np.load(f"data/{argv[1]}.npz")

# Params
G = data["global_params"][0]
M = data["massive_obj_params"][0]
m = data["own_params"][0]

# Orbital data
pos = data["pos"]
vel = data["vel"]

# Basic calcs
dist = np.linalg.norm(pos, axis=1)
speed = np.linalg.norm(vel, axis=1)

# Time
ts = np.linspace(0, 1, pos.shape[0])

# Enegrgy
E_kinetic = speed**2 / 2
E_potential = -G * (M + m) / dist
E_total = E_kinetic + E_potential

# Graph
plt.rcParams["text.usetex"] = True

fig, ax = plt.subplots()
ax.set_title(f"Orbital energies over time for file {argv[1]}.npz", fontsize=25)
ax.set_xlim(0, 1)
ax.set_xlabel(r"$t / t_{\mbox{max}}$", fontsize=15)
ax.set_ylabel(r"$E$", fontsize=15, rotation=0)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)

ax.plot(ts, E_total, c="red", label="Total")
ax.plot(ts, E_kinetic, c="green", label="Kintetic")
ax.plot(ts, E_potential, c="blue", label="Potential")

ax.legend(loc=1, prop={"size": 25})
plt.show()
