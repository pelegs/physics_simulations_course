import matplotlib.pyplot as plt
import numpy as np

# Constants
g = 9.8  # [m/s^2]
L = 1.0  # [m]

# Parameters
t_max = 3.0  # [s]
th0 = np.pi / 20
omega = np.sqrt(g / L)


def simulation(dt):
    # Setup time
    time_series = np.arange(0, t_max, dt)
    num_steps = time_series.shape[0]
    theta = np.zeros(num_steps, dtype=np.double)
    vel = np.zeros(num_steps, dtype=np.double)

    # Initial conditions
    theta[0] = th0
    vel[0] = 0.0

    # Simulation
    for i, t in enumerate(time_series[1:], start=1):
        a_grav = -g / L * np.sin(theta[i - 1])
        vel[i] = vel[i - 1] + a_grav * dt
        theta[i] = theta[i - 1] + vel[i] * dt

    # Return x positions
    return time_series, theta


# Setup graphics
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
fig, ax = plt.subplots()
ax.set_title("Pendulum angle vs time", fontsize=20)
ax.set_xlabel(r"$t$\ [s]", fontsize=15)
ax.set_ylabel(r"$\theta$\ [rad]", fontsize=15)
ax.set_xlim(0, t_max)

if __name__ == "__main__":
    # Simulations
    for dt in [0.5, 0.25, 0.1, 0.05, 0.01]:
        time, th = simulation(dt)
        ax.plot(time, th, ".")

    # Analytical solution
    time = np.arange(0, t_max, 0.001)
    th = th0 * np.cos(omega * time)
    ax.plot(time, th)

    # Show plots
    plt.show()
