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
t_max = 50.0  # [s]
dt = 0.01  # [s]

# Variables
time_series = np.arange(0, t_max, dt)
num_steps = time_series.shape[0]
num_objects = 5

th = np.zeros((2, num_objects, num_steps))
th_vel = np.zeros((2, num_objects, num_steps))
th_acc = np.zeros((2, num_objects, num_steps))

# Initial conditions
th0_mean = np.random.uniform(-np.pi / 4, np.pi / 4, 2)
th0_dev = 0.1
th[:, :, 0] = np.random.uniform(
    low=np.repeat(th0_mean - th0_dev, num_objects).reshape(2, num_objects),
    high=np.repeat(th0_mean + th0_dev, num_objects).reshape(2, num_objects),
    size=(2, num_objects),
)


##########################
#        Graphics        #
##########################

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
fig = plt.figure(figsize=(10, 9), layout="constrained")
gs = GridSpec(2, 2, figure=fig)
ax_vis = fig.add_subplot(gs[0, 0])
ax_walk = fig.add_subplot(gs[0, 1])
ax_ps_th1 = fig.add_subplot(gs[1, 0])
ax_ps_th2 = fig.add_subplot(gs[1, 1])
fig.suptitle("Double pendulum")

# Visual
ax_vis.set_title("Visual view")
ax_vis.get_xaxis().set_ticks([])
ax_vis.get_yaxis().set_ticks([])

# random walk?
ax_walk.set_title("random walk?")
ax_walk.set_xlabel(r"$\theta_{1}$")
ax_walk.set_ylabel(r"$\theta_{2}$")
colors = plt.color_sequences.get("Set1")

# Phase space th1
ax_ps_th1.set_title(r"Phase space, $\theta_{1}$")
ax_ps_th1.set_xlabel(r"$\theta$\ [rad]")
ax_ps_th1.set_ylabel(r"$\dot{\theta}_{1}$\ [rad/s]")

# Phase space th2
ax_ps_th2.set_title(r"Phase space, $\theta_{2}$")
ax_ps_th2.set_xlabel(r"$\theta$\ [rad]")
ax_ps_th2.set_ylabel(r"$\dot{\theta}_{2}$\ [rad/s]")


###########################
#        Functions        #
###########################


def get_th1_acc(t):
    M1 = -g * (2 * m1 + m2) * np.sin(th[0, :, t - 1])
    M2 = -m2 * g * np.sin(th[0, :, t - 1] - 2 * th[1, :, t - 1])
    interaction = (
        -2
        * np.sin(th[0, :, t - 1] - th[1, :, t - 1])
        * m2
        * np.cos(
            th_vel[1, :, t - 1] ** 2 * L2
            + th_vel[0, :, t - 1] ** 2
            * L1
            * np.cos(th[0, :, t - 1] - th[1, :, t - 1])
        )
    )
    normalization = L1 * (
        2 * m1 + m2 - m2 * np.cos(2 * th[0, :, t - 1] - 2 * th[1, :, t - 1])
    )
    return (M1 + M2 + interaction) / normalization


def get_th2_acc(t):
    system = (
        2
        * np.sin(th[0, :, t - 1] - th[1, :, t - 1])
        * (
            th_vel[0, :, t - 1] ** 2 * L1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(th[0, :, t - 1])
            + th_vel[1, :, t - 1] ** 2
            * L2
            * m2
            * np.cos(th[0, :, t - 1] - th[1, :, t - 1])
        )
    )
    normalization = L1 * (
        2 * m1 + m2 - m2 * np.cos(2 * th[0, :, t - 1] - 2 * th[1, :, t - 1])
    )
    return system / normalization


##########################
#        Main run        #
##########################

for i, time in enumerate(time_series[1:], start=1):
    th_acc[0, :, i] = get_th1_acc(i)
    th_acc[1, :, i] = get_th2_acc(i)
    th_vel[:, :, i] = th_vel[:, :, i - 1] + th_acc[:, :, i] * dt
    th[:, :, i] = th[:, :, i - 1] + th_vel[:, :, i] * dt

bob1x = L1 * np.sin(th[0])
bob1y = -L1 * np.cos(th[0])
bob2x = bob1x + L2 * np.sin(th[1])
bob2y = bob1y - L2 * np.cos(th[1])

# ax_walk.set_xlim(min(th1), max(th1))
# ax_walk.set_ylim(min(th2), max(th2))


#########################
#        Figures        #
#########################

# Animate?
imgs = []
anim_interval = 10
for i, time in enumerate(
    tqdm(time_series[::anim_interval], desc="Creating frames")
):
    k = i * anim_interval
    (ln1,) = ax_vis.plot(
        [0, bob1x[0, k]],
        [0, bob1y[0, k]],
        color="black",
        lw=3,
    )
    (ln2,) = ax_vis.plot(
        [bob1x[0, k], bob2x[0, k]],
        [bob1y[0, k], bob2y[0, k]],
        color="black",
        lw=3,
    )
    (bob1,) = ax_vis.plot(
        bob1x[0, k],
        bob1y[0, k],
        "o",
        markersize=22,
        color="darkturquoise",
        zorder=100,
    )
    (bob2,) = ax_vis.plot(
        bob2x[0, k],
        bob2y[0, k],
        "o",
        markersize=22,
        color="darkturquoise",
        zorder=100,
    )
    walk_plt = [
        ax_walk.plot(th[0, i, :k], th[1, i, :k], colors[i])[0]
        for i in range(num_objects)
    ]
    ps_th1 = [
        ax_ps_th1.plot(th[0, i, :k], th_vel[0, i, :k], colors[i])[0]
        for i in range(num_objects)
    ]
    ps_th2 = [
        ax_ps_th1.plot(th[1, i, :k], th_vel[1, i, :k], colors[i])[0]
        for i in range(num_objects)
    ]
    new_frame = [ln1, ln2, bob1, bob2] + walk_plt + ps_th1 + ps_th2
    imgs.append(new_frame)


ani = animation.ArtistAnimation(fig, imgs, interval=1)
writervideo = animation.FFMpegWriter(fps=30)
# ani.save("double_pendulum_test.mp4", writer=writervideo)
update_func = lambda _i, _: progress_bar.update(1)
with tqdm(
    total=int(num_steps / anim_interval), desc="Saving video"
) as progress_bar:
    ani.save(
        "code/mechanics/videos/double_pendulum_test.mp4",
        writer=writervideo,
        progress_callback=update_func,
    )
