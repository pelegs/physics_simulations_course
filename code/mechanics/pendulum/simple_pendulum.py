import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

############################
#        Simultaion        #
############################

# Constants
g = 9.8  # [m/s^2]
L = 1.0  # [m]

# Parameters
t_max = 10.0  # [s]
dt = 0.05  # [s]
vel_res_sqr = g / L  # [rad/s]

# Variables
time_series = np.arange(0, t_max, dt)
num_steps = time_series.shape[0]
theta = np.zeros(num_steps, dtype=np.float32)
vel = np.zeros(num_steps, dtype=np.float32)

# Initial conditions
theta[0] = np.pi / 4
vel[0] = 0.0

# Run simulation
for i, t in enumerate(tqdm(time_series[1:-1]), start=1):
    a_grav = -g / L * np.sin(theta[i - 1])
    vel[i] = vel[i - 1] + a_grav * dt
    theta[i] = theta[i - 1] + vel[i] * dt

###########################
#        Animation        #
###########################

# General
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
fig = plt.figure(figsize=(10, 9), layout="constrained")
gs = GridSpec(2, 2, figure=fig)
ax_vis = fig.add_subplot(gs[0, 0])
ax_phase = fig.add_subplot(gs[0, 1])
ax_time = fig.add_subplot(gs[1, :])
fig.suptitle("Damped pendulum", fontsize=25)

# Visual
ax_vis.set_title("Visual view", fontsize=20)
ax_vis.get_xaxis().set_ticks([])
ax_vis.get_yaxis().set_ticks([])
ax_vis.set_xlim(-1.25 * L, 1.25 * L)
ax_vis.set_ylim(-1.25 * L, 1.25 * L)
ax_vis.set_aspect("equal", "box")

# Phase-space
ax_phase.set_title("Phase space", fontsize=20)
ax_phase.set_xlabel(r"$\theta$\ [rad]", fontsize=15)
ax_phase.set_ylabel(r"$\omega$\ [rad/s]", fontsize=15)
ax_phase.set_xlim(min(theta) - 0.5, max(theta) + 0.5)
ax_phase.set_ylim(min(vel) - 0.5, max(vel) + 0.5)
ax_phase.set_aspect("equal", "box")

# Time plot
ax_time.set_title("Time plot", fontsize=20)
ax_time.set_xlabel(r"$t$\ [s]", fontsize=15)
ax_time.set_ylabel(r"$\theta$\ [rad]", fontsize=15)
ax_time.set_xlim(0, t_max)
ax_time.set_ylim((min(theta) - 0.5), max(theta) + 0.5)

# Set up everything
(rod,) = ax_vis.plot(
    (0, L * np.sin(theta[0])),
    (0, -L * np.cos(theta[0])),
    color="blue",
    solid_capstyle="round",
    lw=3,
)
(bob,) = ax_vis.plot(
    [L * np.sin(theta[0])],
    [-L * np.cos(theta[0])],
    "o",
    markersize=20,
    color="red",
)
(fix,) = ax_vis.plot([0, 0], [0, 0], "o", markersize=5, color="black")
(phase_plt,) = ax_phase.plot(theta[0], vel[0], "red")
(time_plt,) = ax_time.plot(time_series[0], theta[0], "blue")


# Animation function
def animate(frame):
    rod_xs = np.array([0, L * np.sin(theta[frame])])
    rod_ys = np.array([0, -L * np.cos(theta[frame])])
    rod.set_data(rod_xs, rod_ys)
    bob.set_data(
        [L * np.sin(theta[frame])],
        [-L * np.cos(theta[frame])],
    )
    phase_plt.set_data(theta[:frame], vel[:frame])
    time_plt.set_data(time_series[:frame], theta[:frame])
    return [rod, bob, phase_plt, time_plt]


# Run animation
anim = FuncAnimation(fig, animate, frames=num_steps, interval=20, blit=True)
plt.show()
