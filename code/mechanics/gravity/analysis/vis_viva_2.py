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

# Orbital variables/constants
distance = np.linalg.norm(pos, axis=1)
mu = G * (M + m)
h = np.cross(pos, vel)
ecc = 1 / mu * (np.cross(vel, h)) - pos / distance[:, None]
ecc_norm = np.linalg.norm(ecc, axis=1)
ecc_hat = ecc / ecc_norm[:, None]
theta = np.arccos((ecc_hat * pos).sum(1) / distance)
a_arr = distance * (1 + ecc_norm * np.cos(theta)) / (1 - ecc_norm**2)
a_mean, a_err = np.mean(a_arr), np.std(a_arr)
print(f"a={a_mean:0.3f}±{a_err:0.3f}")
speed_vis_viva = np.sqrt(mu * (2 / distance - 1 / a_arr))

# Time
ts = np.linspace(0, 1, pos.shape[0])

# Graph
# plt.rcParams["text.usetex"] = True

fig, ax = plt.subplots()
ax.set_title(f"Vis-Viva equation validation", fontsize=25)
ax.set_xlim(0, np.max(distance) * 1.05)
ax.set_xlabel(r"Distance to star", fontsize=15)
ax.set_ylabel(r"Velocity", fontsize=15, rotation=0)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)

ax.plot(
    distance,
    speed_vis_viva,
    linewidth=5,
    c="red",
    alpha=0.5,
    label="Theory",
)
ax.plot(
    distance,
    speed,
    linewidth=2,
    c="black",
    alpha=0.5,
    label="Simulation",
)
# ax.plot(ts, a, c="blue", label="Semi-major axis length")

ax.legend(loc=1, prop={"size": 25})
plt.show()
