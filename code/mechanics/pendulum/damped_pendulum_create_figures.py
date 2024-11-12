from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

##################################
#        Helper functions        #
##################################


def rotate(vec, th):
    c, s = np.cos(th), np.sin(th)
    return np.dot(np.array([[c, -s], [s, c]]), vec)


def argmax(arr):
    return np.max(np.abs(arr))


############################
#        Simultaion        #
############################

# Constants
g = 9.8  # [m/s^2]
L = 1.0  # [m]
vel_res = np.sqrt(g / L)  # [rad/s]
damping_coeff = float(argv[1])
beta = damping_coeff * vel_res  # [1/s]

# Parameters
t_max = 15.0  # [s]
dt = 0.01  # [s]

# Variables
time_series = np.arange(0, t_max, dt)
num_steps = time_series.shape[0]
theta = np.zeros(num_steps, dtype=np.float32)
omega = np.zeros(num_steps, dtype=np.float32)

# Initial conditions
theta[0] = np.pi / 2
omega[0] = 0.0

# Run simulation
for i, t in enumerate(tqdm(time_series[1:-1]), start=1):
    a_grav = -g / L * np.sin(theta[i - 1])
    a_drag = -2 * beta * omega[i - 1]
    omega[i] = omega[i - 1] + (a_grav + a_drag) * dt
    theta[i] = theta[i - 1] + omega[i] * dt

# Some preparation for plotting
theta_norm = theta[:-1] / np.max(np.abs(theta[:-1]))
omega_norm = omega[:-1] / np.max(np.abs(omega[:-1]))


##########################
#        Graphics        #
##########################

# General
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
fig = plt.figure(figsize=(10, 9), layout="constrained", dpi=200)
gs = GridSpec(2, 1, figure=fig, hspace=0.1)
ax_phase = fig.add_subplot(gs[0])
ax_time = fig.add_subplot(gs[1])
figure_title = (
    rf"Damped pendulum, $\beta/\overline{{\omega}}={damping_coeff:0.1f}$"
)
fig.suptitle(figure_title, fontsize=25)

# Phase-space
ax_phase.set_title("Phase space", fontsize=20)
ax_phase.set_xlabel(r"$\theta/\theta_{\max}$", fontsize=15)
ax_phase.set_ylabel(r"$\omega/\omega_{\max}$", fontsize=15, rotation=0)
ax_phase.yaxis.set_label_coords(-0.2, 0.5)
ax_phase.set_xlim(-1.2, 1.2)
ax_phase.set_ylim(-1.2, 1.2)
ax_phase.set_aspect("equal", "box")

# Time plot
ax_time.set_title("Time plot", fontsize=20)
ax_time.set_xlabel(r"$t$\ [s]", fontsize=15)
ax_time.set_ylabel(r"$\theta$\ [rad]", fontsize=15, rotation=0)
ax_time.yaxis.set_label_coords(-0.075, 0.5)
ax_time.set_xlim(0, t_max)
ax_time.set_ylim(-1.2, 1.2)

# Plot
phase_plt = ax_phase.plot(theta_norm, omega_norm, "red")[0]
time_plt_theta = ax_time.plot(time_series[:-1], theta_norm, "blue")[0]

# Save
fig_filename = f"damped_pendulum_beta_{damping_coeff:0.1f}.png"
plt.savefig(f"figures/{fig_filename}", bbox_inches="tight")
