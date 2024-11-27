from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

##################################
#        Helper functions        #
##################################


def normalize(vec):
    if not np.any(vec):
        raise ValueError("Can't normalize the zero vector")
    return vec / np.linalg.norm(vec)


def pol2car(pts_pol):
    pts_car = np.zeros(pts_pol.shape)
    pts_car[0, :] = pts_pol[0, :] * np.cos(pts_pol[1, :])
    pts_car[1, :] = pts_pol[0, :] * np.sin(pts_pol[1, :])
    return pts_car


def get_angle(v1, v2):
    if not np.any(v1) or not np.any(v2):
        raise ValueError("Can't calculate angles for zero vector")
    return np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )


##############################
#        Vector const        #
##############################

Z_ = np.array([0, 0, 1], dtype=float)


######################
#        Data        #
######################

data = np.load(f"data/{argv[1]}.npz")

# Params
G = data["global_params"][0]
M = data["massive_obj_params"][0]
m = data["own_params"][0]

# Orbital data
pos = data["pos"].T
vel = data["vel"].T


##############################
#        Calculations        #
##############################

mu = G * (M + m)
dists = np.linalg.norm(pos, axis=0)
rp = np.min(dists)
ra = np.max(dists)
r0 = pos[:, 0]
v0 = vel[:, 0]
h = np.cross(r0, v0)
h_hat = normalize(h)
e_vec = np.cross(v0, h) / mu - r0 / np.linalg.norm(r0)
e = np.linalg.norm(e_vec)
e_hat = e_vec / e

num_angles = 250
ths = np.linspace(0, 2 * np.pi, num_angles)
r_calc = np.dot(h, h) / (mu * (1 + e * np.cos(ths)))
orbit_calc_pol = np.zeros((2, num_angles))
orbit_calc_pol[0] = r_calc
orbit_calc_pol[1] = ths
orbit_calc_car = pol2car(orbit_calc_pol)

orbit_calc = np.zeros((3, num_angles))
orbit_calc[:2, :] = orbit_calc_car

rot_ax_z_h = np.cross(Z_, h_hat)
rot_ang_z_h = get_angle(Z_, h_hat)
Q1 = Quaternion(axis=rot_ax_z_h, angle=rot_ang_z_h)
orbit_calc = np.array([Q1.rotate(v) for v in orbit_calc.T]).T

e_calc = orbit_calc[:, 0]
rot_ax_ecc_vec = np.cross(e_calc, e_hat)
rot_ang_ecc_vec = get_angle(e_calc, e_hat)
Q2 = Quaternion(axis=rot_ax_ecc_vec, angle=rot_ang_ecc_vec)
orbit_calc = np.array([Q2.rotate(v) for v in orbit_calc.T]).T

e_hat_test = normalize(orbit_calc[:, 0])


##########################
#        Graphics        #
##########################

# Star
num_pts_star = 50
r = 25
u = np.linspace(0, 2 * np.pi, num_pts_star)
v = np.linspace(0, np.pi, num_pts_star)
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax_lim = np.max(np.abs(np.concatenate((pos[0], pos[1], pos[2]))))
ax.set_xlim(-ax_lim, ax_lim)
ax.set_ylim(-ax_lim, ax_lim)
ax.set_zlim(-ax_lim, ax_lim)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ith = 100
ax.scatter3D(pos[0, ::ith], pos[1, ::ith], pos[2, ::ith], marker="o", c="blue")
ax.plot3D(
    orbit_calc[0, :],
    orbit_calc[1, :],
    orbit_calc[2, :],
    c="green",
)
ax.plot_surface(x, y, z, color="red", alpha=1)
# ax.quiver(0, 0, 0, e_hat[0], e_hat[1], e_hat[2], length=200, colors="red")
# ax.quiver(
#     0,
#     0,
#     0,
#     e_hat_test[0],
#     e_hat_test[1],
#     e_hat_test[2],
#     length=150,
#     colors="blue",
# )
plt.show()
