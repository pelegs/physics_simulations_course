import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

##############################
#        General vars        #
##############################

# Constants
g = 9.8  # [m/s^2]
L = 1.0  # [m]

# Parameters
t_max = 30.0  # [s]
dt = 0.025  # [s]
beta = 0.25
vel_res_sqr = g / L  # [rad/s]

# Variables
time_series = np.arange(0, t_max, dt)
num_steps = time_series.shape[0]
theta = np.zeros(num_steps, dtype=np.double)
vel = np.zeros(num_steps, dtype=np.double)

# Initial conditions
theta[0] = np.pi / 5
vel[0] = 0.0

##########################
#        Graphics        #
##########################

# General
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
fig = plt.figure(figsize=(10, 9), layout="constrained")
gs = GridSpec(2, 2, figure=fig)
ax_vis = fig.add_subplot(gs[0, 0])
ax_phase = fig.add_subplot(gs[0, 1])
ax_time = fig.add_subplot(gs[1, :])
fig.suptitle("Damped pendulum")

# Visual
ax_vis.set_title("Visual view")
ax_vis.get_xaxis().set_ticks([])
ax_vis.get_yaxis().set_ticks([])
ax_vis.set_xlim(-1.25 * L, 1.25 * L)
ax_vis.set_ylim(-1.25 * L, 1.25 * L)
ax_vis.text(
    0.05, 0.95, rf"$\beta={beta}$", size=15, transform=ax_vis.transAxes
)

# Phase-space
ax_phase.set_title("Phase space")
ax_phase.set_xlabel(r"$\theta$\ [rad]")
ax_phase.set_ylabel(r"$v$\ [m/s]")

# Time plot
ax_time.set_title("Time plot")
ax_time.set_xlabel(r"$t$\ [s]")
ax_time.set_ylabel(r"$\theta$\ [rad]")

######################
#        Main        #
######################

if __name__ == "__main__":
    # Simulation
    for i, t in enumerate(time_series[1:-1], start=1):
        a_grav = -g / L * np.sin(theta[i - 1])
        a_drag = -2 * beta * vel[i - 1] - vel_res_sqr * np.sin(theta[i - 1])
        vel[i] = vel[i - 1] + (a_grav + a_drag) * dt
        theta[i] = theta[i - 1] + vel[i] * dt

    # Animate?
    imgs = []
    for i, t in enumerate(tqdm(time_series)):
        (rod,) = ax_vis.plot(
            (0, L * np.sin(theta[i])),
            (0, -L * np.cos(theta[i])),
            color="blue",
            solid_capstyle="round",
            lw=3,
        )
        (bob,) = ax_vis.plot(
            [L * np.sin(theta[i])],
            [-L * np.cos(theta[i])],
            "o",
            markersize=20,
            color="red",
        )
        (fix,) = ax_vis.plot([0, 0], [0, 0], "o", color="black")
        (phase_plt,) = ax_phase.plot(theta[:i], vel[:i], "red")
        (time_plt,) = ax_time.plot(time_series[:i], theta[:i], "blue")

        imgs.append([rod, bob, fix, phase_plt, time_plt])

    ani = animation.ArtistAnimation(fig, imgs, interval=10)
    writervideo = animation.FFMpegWriter(fps=30)
    ani.save("code/mechanics/videos/pendulum_test_1.mp4", writer=writervideo)
