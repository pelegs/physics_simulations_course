import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
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
L1 = 5.0  # [m]
L2 = 1.0  # [m]

# Parameters
t_max = 10.0  # [s]
dt = 0.01  # [s]
m1 = 1.0  # [kg]
m2 = 1.5  # [kg]
A = m2 / m1
B = L2 / L1
C = g / L1  # [s^(-2)]

# Variables
time_series = np.arange(0, t_max, dt)
num_steps = time_series.shape[0]
theta_1 = np.zeros(num_steps)
theta_2 = np.zeros(num_steps)
omega_1 = np.zeros(num_steps)
omega_2 = np.zeros(num_steps)
alpha_1 = np.zeros(num_steps)
alpha_2 = np.zeros(num_steps)

# Initial conditions
theta_1[0] = theta_2[0] = np.pi / 1.5


# Calc acceleration at time ti
def get_acc(i):
    dth = theta_1[i - 1] - theta_2[i - 1]
    N = 1 + A * np.sin(dth) ** 2

    acc1 = (
        -(
            (1 + A) * C * np.sin(theta_1[i - 1])
            + A * B * omega_1[i - 1] ** 2 * np.sin(dth)
            + A
            * np.cos(dth)
            * (omega_1[i - 1] ** 2 * np.sin(dth) - C * np.sin(theta_2[i - 1]))
        )
        / N
    )

    acc2 = (
        (1 + A)
        * (omega_1[i - 1] ** 2 * np.sin(dth) - C * np.sin(theta_2[i - 1]))
        + np.cos(dth)
        * (
            (1 + A) * C * np.sin(theta_1[i - 1])
            + A * B * omega_2[i - 1] ** 2 * np.sin(dth)
        )
    ) / (B * N)

    return acc1, acc2


# Run simulation
for i, t in enumerate(time_series[1:-1], start=1):
    alpha_1[i], alpha_2[i] = get_acc(i)
    omega_1[i] = omega_1[i - 1] + alpha_1[i] * dt
    omega_2[i] = omega_2[i - 1] + alpha_2[i] * dt
    theta_1[i] = theta_1[i - 1] + omega_1[i] * dt
    theta_2[i] = theta_2[i - 1] + omega_2[i] * dt

# Bobs position in Cartesian coordinates
bob1_pos = L1 * np.array([np.sin(theta_1), -np.cos(theta_1)]).T
bob2_pos = bob1_pos + L2 * np.array([np.sin(theta_2), -np.cos(theta_2)]).T


###########################
#        Animation        #
###########################

# General
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
fig = plt.figure(figsize=(10, 9), layout="constrained")
gs = GridSpec(2, 2, figure=fig)
ax_vis = fig.add_subplot(gs[0, 0])
ax_th1_vs_th2 = fig.add_subplot(gs[0, 1])
ax_time = fig.add_subplot(gs[1, :])
fig.suptitle("Damped pendulum", fontsize=25)

# Visual
ax_vis.set_title("Visual view", fontsize=20)
ax_vis.get_xaxis().set_ticks([])
ax_vis.get_yaxis().set_ticks([])
ax_vis.set_xlim(-1.25 * (L1 + L2), 1.25 * (L1 + L2))
ax_vis.set_ylim(-1.25 * (L1 + L2), 1.25 * (L1 + L2))
ax_vis.set_aspect("equal", "box")

# Phase-space
ax_th1_vs_th2.set_title(r"$\theta_{1}$ vs. $\theta_{2}$", fontsize=20)
ax_th1_vs_th2.set_xlabel(r"$\theta_{1}$\ [rad]", fontsize=15)
ax_th1_vs_th2.set_ylabel(r"$\theta_{2}$\ [rad]", fontsize=15)
ax_th1_vs_th2.set_xlim(np.min(theta_1) - 0.5, np.max(theta_1) + 0.5)
ax_th1_vs_th2.set_ylim(np.min(theta_2) - 0.5, np.max(theta_2) + 0.5)

# Time plot
ax_time.set_title("Time plot", fontsize=20)
ax_time.set_xlabel(r"$t$\ [s]", fontsize=15)
ax_time.set_ylabel(r"$\theta$\ [rad]", fontsize=15)
ax_time.set_xlim(0, t_max)
ax_time.set_ylim(-1.2, 1.2)

# Set up everything
trace1 = ax_vis.plot(
    bob1_pos[0, 0],
    bob1_pos[0, 1],
    "-",
    color="red",
    alpha=0.5,
)[0]
trace2 = ax_vis.plot(
    bob2_pos[0, 0],
    bob2_pos[0, 1],
    "-",
    color="blue",
    alpha=0.5,
)[0]
rod1 = ax_vis.plot(
    (0, bob1_pos[0, 0]),
    (0, bob1_pos[0, 1]),
    color="blue",
    solid_capstyle="round",
    lw=3,
)[0]
rod2 = ax_vis.plot(
    (bob1_pos[0, 0], bob2_pos[0, 0]),
    (bob1_pos[0, 1], bob2_pos[0, 1]),
    color="blue",
    solid_capstyle="round",
    lw=3,
)[0]

bob1 = ax_vis.plot(
    (bob1_pos[0, 0]),
    (bob1_pos[0, 1]),
    "o",
    markersize=20,
    color="red",
)[0]
bob2 = ax_vis.plot(
    (bob2_pos[0, 0]),
    (bob2_pos[0, 1]),
    "o",
    markersize=20,
    color="red",
)[0]

th1_vs_th2 = ax_th1_vs_th2.plot(theta_1[0], theta_2[0], "purple")[0]
time_series1 = ax_time.plot(time_series[0], theta_1[0], "red")[0]
time_series2 = ax_time.plot(time_series[0], theta_2[0], "blue")[0]


# Animation function
def animate(frame):
    trace1.set_data([bob1_pos[:frame, 0]], [bob1_pos[:frame, 1]])
    trace2.set_data([bob2_pos[:frame, 0]], [bob2_pos[:frame, 1]])
    rod1.set_data([0, bob1_pos[frame, 0]], [0, bob1_pos[frame, 1]])
    rod2.set_data(
        [bob1_pos[frame, 0], bob2_pos[frame, 0]],
        [bob1_pos[frame, 1], bob2_pos[frame, 1]],
    )
    bob1.set_data(
        [bob1_pos[frame, 0]],
        [bob1_pos[frame, 1]],
    )
    bob2.set_data(
        [bob2_pos[frame, 0]],
        [bob2_pos[frame, 1]],
    )

    th1_vs_th2 = ax_th1_vs_th2.plot(
        theta_1[:frame], theta_2[:frame], "purple"
    )[0]
    time_series_1 = ax_time.plot(
        time_series[:frame], theta_1[:frame] / argmax(theta_1), "red"
    )[0]
    time_series_2 = ax_time.plot(
        time_series[:frame], theta_2[:frame] / argmax(theta_2), "blue"
    )[0]

    return [
        trace1,
        trace2,
        rod1,
        rod2,
        bob1,
        bob2,
        th1_vs_th2,
        time_series_1,
        time_series_2,
    ]


# Run animation
anim = FuncAnimation(fig, animate, frames=num_steps, interval=0, blit=True)
plt.show()
