import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["figure.constrained_layout.use"] = True

num_steps = 100
num_particles = 1000


def Brown(num_steps, num_particles):
    pos = np.zeros((num_steps, num_particles))
    vel = np.random.normal(size=(num_steps - 1, num_particles))
    pos[1:, :] = np.cumsum(vel, axis=0)

    time = np.linspace(0, 1, num_steps)
    SD = np.zeros((num_steps, num_particles))
    for i, t in enumerate(time):
        SD[i, :] = (pos[i, :] - pos[0, :]) ** 2
    MSD = np.mean(SD, axis=1)
    STD = np.std(SD, axis=1)

    fig, ax = plt.subplots(2)
    plt.gcf().set_size_inches(15, 10)
    plt.subplots_adjust(wspace=1)

    fig.suptitle(
        rf"Simulated Brownian motion, {num_particles} particles, $\mu=0$ and $\sigma=1$",
        fontsize=35,
    )

    num_shown_traj = 100
    ax[0].set_title(
        f"First {num_shown_traj} trajectories over time", fontsize=25
    )
    ax[0].set_xlabel("Time", fontsize=20)
    ax[0].set_ylabel("Position", fontsize=20)
    ax[0].plot(time, pos[:, :num_shown_traj])

    ax[1].set_title(
        r"Mean Square Displacement, $\langle \left( x(t)-x_{0} \right)^{2} \rangle$, over time",
        fontsize=25,
    )
    ax[1].set_xlabel("Time", fontsize=20)
    ax[1].set_ylabel("Displacement", fontsize=20)
    ax[1].fill_between(time, MSD - STD, MSD + STD, color="#00AAFF", alpha=0.25)
    ax[1].plot(time, MSD, lw=2)

    plt.savefig(f"figs/{num_particles}_particles.png", dpi=300)

    # plt.show()


for n in [100, 500, 1000, 5000, 10000, 50000, 100000]:
    Brown(100, n)
