import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#############################
#        Integrators        #
#############################


def forward_euler():
    theta, omega = np.zeros((2, num_steps))
    theta[0] = theta_0
    omega[0] = omega_0
    for i, _ in enumerate(
        tqdm(time_series[1:-1], desc="Forward Euler"), start=1
    ):
        a_grav = -g / L * np.sin(theta[i - 1])
        omega[i] = omega[i - 1] + a_grav * dt
        theta[i] = theta[i - 1] + omega[i] * dt
    return theta


def middle_euler():
    theta, omega = np.zeros((2, num_steps))
    theta[0] = theta_0
    omega[0] = omega_0
    for i, _ in enumerate(
        tqdm(time_series[1:-1], desc="Forward Euler"), start=1
    ):
        a_grav = -g / L * np.sin(theta[i - 1])
        pass
    return theta


############################
#        Simultaion        #
############################

# Constants
g = 9.8  # [m/s^2]
L = 1.0  # [m]
vel_res = np.sqrt(g / L)  # [rad/s]

# Parameters
t_max = 15.0  # [s]
dt = 0.1  # [s]

# Variables
time_series = np.arange(0, t_max, dt)
num_steps = time_series.shape[0]

# Initial conditions
theta_0 = np.pi / 10
omega_0 = 0.0

# Run simulation
theta_euler_1st = forward_euler()

# Analytical solution (small angles approx)
theta_analytic = theta_0 * np.cos(time_series * vel_res)


###########################
#        Animation        #
###########################

# General
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
fig, ax = plt.subplots(figsize=(10, 9), layout="constrained", dpi=200)
fig.suptitle("Simple pendulum, different integrators", fontsize=25)

# Time plot
ax.set_title("Time plot", fontsize=20)
ax.set_xlabel(r"$t$\ [s]", fontsize=15)
ax.set_ylabel(r"$\theta$\ [rad]", fontsize=15, rotation=0)
ax.yaxis.set_label_coords(-0.075, 0.5)
ax.set_xlim(0, t_max)

# Plot
ax.plot(time_series, theta_euler_1st, "o", color="red")[0]
ax.plot(time_series, theta_analytic, color="black")[0]

# Save
fig_filename = f"simple_pendulum_different_integrators.png"
plt.savefig(f"figures/{fig_filename}", bbox_inches="tight")
