import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLines


def get_data(filename):
    data = np.load(filename)

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
    # a_mean, a_err = np.mean(a_arr), np.std(a_arr)
    speed_vis_viva = np.sqrt(mu * (2 / distance - 1 / a_arr))

    return dist, speed, speed_vis_viva, np.mean(np.abs(ecc))


dist_list = list()
speed_list = list()
vis_viva_speed_list = list()
ecc_list = list()
for i in range(1, 7):
    dist, speed, vis_viva_speed, ecc = get_data(f"data/planet_{i}.npz")
    dist_list.append(dist)
    speed_list.append(speed)
    vis_viva_speed_list.append(vis_viva_speed)
    ecc_list.append(ecc)

# Graphics
fig, ax = plt.subplots()
ax.set_title("Vis-Viva equation validation", fontsize=25)
ax.set_xlabel(r"Distance to star", fontsize=15)
ax.set_ylabel(r"Velocity", fontsize=15)
ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)

for distance, speed, vis_viva_speed, ecc in zip(
    dist_list, speed_list, vis_viva_speed_list, ecc_list
):
    ax.plot(
        distance,
        vis_viva_speed,
        linewidth=7,
        alpha=0.5,
        label=f"e={ecc:0.3f}",
    )
    ax.plot(
        distance,
        speed,
        linewidth=1,
        c="black",
        alpha=0.5,
    )
    # ax.annotate("test")
plt_lines = plt.gca().get_lines()[::2]
mid_dists = [np.mean(dist) for dist in dist_list]
labelLines(plt_lines, align=True, xvals=mid_dists)

plt.show()
