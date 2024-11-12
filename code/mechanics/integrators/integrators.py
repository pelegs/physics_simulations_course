import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

############################
#        Parameters        #
############################

# General params
A = 1.00
phi = 0.0
m = 3.0
k = 2
omega = np.sqrt(k / m)

# Time related
dt = 0.2
num_steps = 1000
L = 1
times_series = np.linspace(0, num_steps * dt, num_steps)

# Starting conditions
x0 = A
v0 = 0.0

#############################
#        Integrators        #
#############################


def exact_solution():
    xs = A * np.cos(omega * times_series + phi)
    vs = -A * omega * np.sin(omega * times_series + phi)
    return xs, vs


def euler_explicit():
    xs, vs = np.zeros((2, num_steps))
    xs[0], vs[0] = x0, v0
    for i, _ in enumerate(
        tqdm(times_series[1:], desc="Forward Euler"), start=1
    ):
        a = -(k / m) * xs[i - 1]
        vs[i] = vs[i - 1] + a * dt
        xs[i] = xs[i - 1] + vs[i] * dt

    return xs, vs


def euler_implicit():
    A = 1 + (dt**2 * k / m)
    xs, vs = np.zeros((2, num_steps))

    for i, _ in enumerate(tqdm(times_series[1:], desc="Forward Euler")):
        xs[i] = (xs[i - 1] + dt * vs[i - 1]) / A
        vs[i] = vs[i - 1] + dt * (-k / m * xs[i])

    return xs, vs


#############################
#        Simulations        #
#############################

theta_exact, omega_exact = exact_solution()
theta_euler_explicit, omega_euler_explicit = euler_explicit()
theta_euler_implicit, omega_euler_implicit = euler_implicit()

##########################
#        Graphics        #
##########################

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

fig, ax = plt.subplots(figsize=(10, 9), layout="constrained", dpi=200)
ax.set_title("Simple harmonic oscillator, different integration methods")
ax.set_xlabel(r"$t$\ [sec]")
ax.set_ylabel(r"$\theta$\ [rad]")
ax.set_xlim(0, times_series[-1])

# Plots
ax.plot(
    times_series[:-1],
    theta_euler_explicit[:-1],
    "o",
    markersize=2,
    color="red",
)
ax.plot(times_series[:-1], theta_exact[:-1], "black")

plt.show()
