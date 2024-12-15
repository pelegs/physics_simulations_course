import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# Constants
g = 9.8

# Helper functions
c = lambda n, i, j: n - max(i, j)


# Main simulation class
class PendulumSystem:
    def __init__(
        self,
        num_steps=1000,
        dt=0.01,
        theta_0=np.ones(5) * np.pi / 4,
        omega_0=np.zeros(5),
    ):
        self.num_steps = num_steps
        self.step = 0
        self.dt = dt
        self.dt_half = dt / 2
        self.dt_sixth = dt / 6
        self.time_seris = np.linspace(0, dt * num_steps, num_steps)

        self.num_pendulums = theta_0.shape[0]
        self.theta = np.zeros((self.num_steps, self.num_pendulums))
        self.theta[0] = theta_0
        self.omega = np.zeros((self.num_steps, self.num_pendulums))
        self.omega[0] = omega_0

        self.coords = np.zeros((self.num_steps, self.num_pendulums, 2))

        self.run()
        self.create_cartesian_coords()

    def A(self, theta):
        return np.array(
            [
                [
                    c(self.num_pendulums, i, j) * np.cos(theta[i] - theta[j])
                    for i in range(self.num_pendulums)
                ]
                for j in range(self.num_pendulums)
            ]
        )

    def b(self, theta, omega):
        return np.array(
            [
                np.sum(
                    [
                        c(self.num_pendulums, i, j)
                        * omega[j] ** 2
                        * np.sin(theta[i] - theta[j])
                        for j in range(self.num_pendulums)
                    ]
                )
                - g * (self.num_pendulums - i) * np.sin(theta[i])
                for i in range(self.num_pendulums)
            ]
        )

    def f(self, theta, omega):
        A = self.A(theta)
        b = self.b(theta, omega)
        return [omega, np.linalg.solve(A, b)]

    def RK4(self, theta, omega):
        k1 = self.f(theta, omega)
        k2 = self.f(
            theta + k1[0] * self.dt * 0.5, omega + k1[1] * self.dt * 0.5
        )
        k3 = self.f(
            theta + k2[0] * self.dt * 0.5, omega + k2[1] * self.dt * 0.5
        )
        k4 = self.f(theta + k2[0] * self.dt, omega + k2[1] * self.dt)

        theta_diff = self.dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        omega_diff = self.dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

        return [theta + theta_diff, omega + omega_diff]

    def next_step(self):
        self.theta[self.step + 1], self.omega[self.step + 1] = self.RK4(
            self.theta[self.step], self.omega[self.step]
        )
        self.step += 1

    def run(self):
        for _ in tqdm(self.time_seris[:-1]):
            self.next_step()

    def create_cartesian_coords(self):
        self.coords[:, :, 0] = np.sin(self.theta)
        self.coords[:, :, 1] = -np.cos(self.theta)
        self.coords = np.cumsum(self.coords, axis=1)


def draw_step(step=0):
    first_rod.set_data(
        [0, system.coords[step, 0, 0]],
        [0, system.coords[step, 0, 1]],
    )
    pendulum_figure.set_data(
        system.coords[step, :, 0],
        system.coords[step, :, 1],
    )
    return [first_rod, pendulum_figure]


if __name__ == "__main__":
    n = 6
    theta_0 = np.random.uniform(np.pi / 4 - 0.1, np.pi / 4 + 0.1, n)
    omega_0 = np.zeros(n)
    system = PendulumSystem(
        theta_0=theta_0, omega_0=omega_0, num_steps=1000, dt=1.0e-3
    )

    fig, ax = plt.subplots()
    ax.set_title(f"{n}-pendulum test")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.25 * n, 1.25 * n)
    ax.set_ylim(-1.25 * n, 1.25 * n)
    ax.set_aspect("equal", "box")
    fix = ax.plot([0, 0], [0, 0], "o", markersize=5, color="black")[0]
    first_rod = ax.plot(
        [0, system.coords[0, 0, 0]],
        [0, system.coords[0, 0, 1]],
        color="blue",
        linewidth=2,
    )[0]
    pendulum_figure = ax.plot(
        system.coords[0, :, 0],
        system.coords[0, :, 1],
        "-o",
        color="black",
        linewidth=2,
        markersize=10,
        markerfacecolor="red",
    )[0]

    # Run animation
    anim = FuncAnimation(
        fig, draw_step, frames=system.num_steps - 1, interval=0, blit=True
    )
    plt.show()
