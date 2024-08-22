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
L1 = 1.0  # [m]
L2 = 1.0  # [m]
m1 = 1.0  # [kg]
m2 = 1.0  # [kg]

# Parameters
t_max = 10.0  # [s]
dt = 0.01  # [s]

# Variables
time_series = np.arange(0, t_max, dt)
num_steps = time_series.shape[0]
th1 = np.zeros(num_steps)
th2 = np.zeros(num_steps)
th1_d = np.zeros(num_steps)
th2_d = np.zeros(num_steps)
th1_dd = np.zeros(num_steps)
th2_dd = np.zeros(num_steps)

# Initial conditions
th1[0] = np.random.uniform(-np.pi, np.pi)
th2[0] = np.random.uniform(-np.pi, np.pi)


##########################
#        Graphics        #
##########################

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
fig = plt.figure(figsize=(10, 9), layout="constrained")
gs = GridSpec(2, 1, figure=fig)
ax_vis = fig.add_subplot(gs[0])
ax_walk = fig.add_subplot(gs[1])
fig.suptitle("Double pendulum")

# Visual
ax_vis.set_title("Visual view")
ax_vis.get_xaxis().set_ticks([])
ax_vis.get_yaxis().set_ticks([])

# Random walk?
ax_walk.set_title("Random walk?")
ax_walk.set_xlabel(r"$\theta_{1}$")
ax_walk.set_ylabel(r"$\theta_{2}$")


###########################
#        Functions        #
###########################


def get_th1_acc(t):
    M1 = -g * (2 * m1 + m2) * np.sin(th1[t - 1])
    M2 = -m2 * g * np.sin(th1[t - 1] - 2 * th2[t - 1])
    interaction = (
        -2
        * np.sin(th1[t - 1] - th2[t - 1])
        * m2
        * np.cos(
            th2_d[t - 1] ** 2 * L2
            + th1_d[t - 1] ** 2 * L1 * np.cos(th1[t - 1] - th2[t - 1])
        )
    )
    normalization = L1 * (
        2 * m1 + m2 - m2 * np.cos(2 * th1[t - 1] - 2 * th2[t - 1])
    )
    return (M1 + M2 + interaction) / normalization


def get_th2_acc(t):
    system = (
        2
        * np.sin(th1[t - 1] - th2[t - 1])
        * (
            th1_d[t - 1] ** 2 * L1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(th1[t - 1])
            + th2_d[t - 1] ** 2 * L2 * m2 * np.cos(th1[t - 1] - th2[t - 1])
        )
    )
    normalization = L1 * (
        2 * m1 + m2 - m2 * np.cos(2 * th1[t - 1] - 2 * th2[t - 1])
    )
    return system / normalization


##########################
#        Main run        #
##########################

for i, time in enumerate(time_series[1:], start=1):
    th1_dd[i] = get_th1_acc(i)
    th2_dd[i] = get_th2_acc(i)
    th1_d[i] = th1_d[i - 1] + th1_dd[i] * dt
    th2_d[i] = th2_d[i - 1] + th2_dd[i] * dt
    th1[i] = th1[i - 1] + th1_d[i] * dt
    th2[i] = th2[i - 1] + th2_d[i] * dt

bob1x = L1 * np.sin(th1)
bob1y = -L1 * np.cos(th1)
bob2x = bob1x + L2 * np.sin(th2)
bob2y = bob1y - L2 * np.cos(th2)

ax_walk.set_xlim(min(th1), max(th1))
ax_walk.set_ylim(min(th2), max(th2))


#########################
#        Figures        #
#########################

# Animate?
imgs = []
for i, time in enumerate(tqdm(time_series)):
    (ln1,) = ax_vis.plot(
        [0, bob1x[i]],
        [0, bob1y[i]],
        color="black",
        lw=3,
    )
    (ln2,) = ax_vis.plot(
        [bob1x[i], bob2x[i]],
        [bob1y[i], bob2y[i]],
        color="black",
        lw=3,
    )
    (bob1,) = ax_vis.plot(
        bob1x[i],
        bob1y[i],
        "o",
        markersize=22,
        color="darkturquoise",
        zorder=100,
    )
    (bob2,) = ax_vis.plot(
        bob2x[i],
        bob2y[i],
        "o",
        markersize=22,
        color="darkturquoise",
        zorder=100,
    )
    (walk_plt,) = ax_walk.plot(th1[:i], th2[:i], "red")

    imgs.append([ln1, ln2, bob1, bob2, walk_plt])

ani = animation.ArtistAnimation(fig, imgs, interval=10)
writervideo = animation.FFMpegWriter(fps=30)
ani.save("double_pendulum_test.mp4", writer=writervideo)
