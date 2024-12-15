import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from tqdm import tqdm


class Pendulum:
    def __init__(self, n=1, thetas=None, theta_dots=None, g=-9.8):
        self.n = n
        self.g = g
        self.thetas = np.array(
            thetas if thetas is not None else [0.5 * np.pi] * 6
        )
        self.theta_dots = np.array(
            theta_dots if theta_dots is not None else [0] * 6
        )

    def A(self, thetas):
        M = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                M[i, j] = (self.n - max(i, j)) * np.cos(thetas[i] - thetas[j])
        return M

    def b(self, thetas, theta_dots):
        b = np.zeros(self.n)
        for i in range(self.n):
            b_i = 0
            for j in range(self.n):
                b_i -= (
                    (self.n - max(i, j))
                    * np.sin(thetas[i] - thetas[j])
                    * theta_dots[j] ** 2
                )
            b_i -= self.g * (self.n - i) * np.sin(thetas[i])
            b[i] = b_i
        return b

    def f(self, thetas, theta_dots):
        A = self.A(thetas)
        b = self.b(thetas, theta_dots)

        theta_ddots = np.linalg.solve(A, b)

        return theta_dots, theta_ddots

    def RK4(self, dt, thetas, theta_dots):
        k1 = self.f(thetas, theta_dots)
        k2 = self.f(thetas + 0.5 * dt * k1[0], theta_dots + 0.5 * dt * k1[1])
        k3 = self.f(thetas + 0.5 * dt * k2[0], theta_dots + 0.5 * dt * k2[1])
        k4 = self.f(thetas + dt * k3[0], theta_dots + dt * k3[1])

        theta_delta = ((k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * dt) / 6
        theta_dot_delta = ((k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * dt) / 6
        return thetas + theta_delta, theta_dots + theta_dot_delta

    def tick(self, dt):
        self.thetas, self.theta_dots = self.RK4(
            dt, self.thetas, self.theta_dots
        )

    @property
    def coordinates(self):
        x, y = 0, 0
        coords = []
        for i in range(self.n):
            x += np.sin(self.thetas[i])
            y += np.cos(self.thetas[i])
            coords.append((x, y))
        return coords

    @property
    def _coordinates(self):
        coords_all = np.zeros((self.n, 2))
        coords_all[:, 0] = np.sin(thetas)
        coords_all[:, 1] = np.cos(thetas)
        return np.cumsum(coords_all, axis=0)


def precompute_motion(pendulum, total_time, dt):
    steps = int(total_time / dt)
    theta_history = []
    theta_dot_history = []
    coord_history = []

    for _ in tqdm(range(steps), desc="Precomputing simulation"):
        pendulum.tick(dt)
        theta_history.append(pendulum.thetas.copy())
        theta_dot_history.append(pendulum.theta_dots.copy())
        coord_history.append(pendulum.coordinates)

    return np.array(theta_history), np.array(theta_dot_history), coord_history


###########################
#        Simulation       #
###########################

# theta : place
# theta_dots : Ang. Velocity
# theta_ddots : Ang. acceleration

# Parameters
n = 5
g = -9.8
dt = 1 / 100
total_time = 10
s_theta_ratio = 0

# Initial
thetas = np.array([((0.5 + -1 * s_theta_ratio) % 2) * np.pi] * n)
theta_dots = np.array([0] * n)

pendulum = Pendulum(n=n, thetas=thetas, theta_dots=theta_dots, g=g)

theta_history, theta_dot_history, coord_history = precompute_motion(
    pendulum, total_time, dt
)
time_series = np.linspace(0, total_time, len(theta_history))

###########################
#        Animation        #
###########################

# General
fig = plt.figure(figsize=(8, 12), layout="constrained")
gs = GridSpec(2, 2, figure=fig)
ax_pendulum = fig.add_subplot(gs[0, 0])
ax_phase = fig.add_subplot(gs[0, 1])
ax_time = fig.add_subplot(gs[1, :])
figure_title = "{n} Particle Simple-Pendulum".format(n=n)
fig.suptitle(figure_title, fontsize=25)
plt.get_current_fig_manager().set_window_title(
    f"{figure_title}, L={1}, g={g}, dt={'%.4f' % dt}"
)

# Time
ax_time.set_xlim(0, total_time)
ax_time.set_ylim(np.min(theta_history) - 0.1, np.max(theta_history) + 0.1)
ax_time.set_xlabel("Time (s)")
ax_time.set_ylabel("Theta (rad)")

lines_time = []
colors = ["r", "g", "b", "c", "m"]
for i in range(pendulum.n):
    (line,) = ax_time.plot(
        [], [], color=colors[i % len(colors)], label=f"Bob {i + 1}"
    )
    lines_time.append(line)

# visual
ax_pendulum.set_aspect("equal")
ax_pendulum.set_xlim(-n - 1, n + 1)
ax_pendulum.set_ylim(-n - 1, n + 1)

lines_pendulum, points_pendulum = [], []
for i in range(pendulum.n):
    (line,) = ax_pendulum.plot([], [], color="r")
    (point,) = ax_pendulum.plot([], [], "o", color=colors[i % len(colors)])
    lines_pendulum.append(line)
    points_pendulum.append(point)

trajectories = [[] for _ in range(pendulum.n)]
trajectory_lines = [
    ax_pendulum.plot([], [], color=colors[i % len(colors)], alpha=0.5)[0]
    for i in range(pendulum.n)
]

# Phase space
max_abs_th = np.max(np.abs(theta_history))
max_abs_thd = np.max(np.abs(theta_dot_history))
max_value = max(max_abs_thd, max_abs_th)

ax_phase.set_xlim(-1 * max_value - 0.1, max_value + 0.1)
ax_phase.set_ylim(-1 * max_value - 0.1, max_value + 0.1)
ax_phase.set_xlabel("Theta (rad)")
ax_phase.set_ylabel("Theta Dot (rad/s)")

lines_phase = []
for i in range(pendulum.n):
    (line,) = ax_phase.plot(
        [], [], color=colors[i % len(colors)], label=f"Bob {i + 1}"
    )
    lines_phase.append(line)


# Animation
def animate(frame):
    if frame == 0:
        for i in range(pendulum.n):
            trajectories[i] = []
        for line in trajectory_lines:
            line.set_data([], [])

    coords = coord_history[frame]
    x1, y1 = 0, 0

    for i, (x, y) in enumerate(coords):
        x2, y2 = x, y
        lines_pendulum[i].set_data([x1, x2], [y1, y2])
        points_pendulum[i].set_data([x2], [y2])
        x1, y1 = x2, y2

        trajectories[i].append((x, y))
        trajectory_lines[i].set_data(
            [pos[0] for pos in trajectories[i]],
            [pos[1] for pos in trajectories[i]],
        )

    for i, line in enumerate(lines_time):
        line.set_data(time_series[:frame], theta_history[:frame, i])

    for i, line in enumerate(lines_phase):
        line.set_data(theta_history[:frame, i], theta_dot_history[:frame, i])

    return (
        lines_pendulum
        + points_pendulum
        + lines_time
        + lines_phase
        + trajectory_lines
    )


# Animation Func.
ani = FuncAnimation(
    fig,
    animate,
    frames=len(coord_history),
    interval=dt * 1000,
    blit=True,
    repeat=True,
)
plt.show()
