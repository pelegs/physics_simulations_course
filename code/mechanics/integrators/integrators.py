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
dt = 0.1
dt_half = dt / 2
dt_sixth = dt / 6
num_steps = 1000
L = 1
times_series = np.linspace(0, num_steps * dt, num_steps)

# Starting conditions
x0 = A
v0 = 0.0

#############################
#        Integrators        #
#############################


def create_blank_data():
    xs, vs = np.zeros((2, num_steps))
    xs[0], vs[0] = x0, v0
    return xs, vs


def derivatives(x, v):
    return v, -(k / m) * x


def exact_solution():
    xs = A * np.cos(omega * times_series + phi)
    vs = -A * omega * np.sin(omega * times_series + phi)
    return xs, vs


def euler_explicit():
    xs, vs = create_blank_data()
    for i, _ in enumerate(
        tqdm(times_series[1:], desc="Explicit Euler"), start=1
    ):
        a = -(k / m) * xs[i - 1]
        vs[i] = vs[i - 1] + a * dt
        xs[i] = xs[i - 1] + vs[i - 1] * dt

    return xs, vs


def euler_implicit():
    c1 = 1 + (dt**2 * k / m)
    c2 = dt * k / m
    xs, vs = create_blank_data()
    for i, _ in enumerate(
        tqdm(times_series[1:], desc="Implicit Euler"), start=1
    ):
        xs[i] = (xs[i - 1] + dt * vs[i - 1]) / c1
        vs[i] = vs[i - 1] - c2 * xs[i]

    return xs, vs


def RK2():
    xs, vs = create_blank_data()
    for i, _ in enumerate(
        tqdm(times_series[1:], desc="Forward Euler"), start=1
    ):
        k1x, k1v = derivatives(xs[i - 1], vs[i - 1])
        k2x, k2v = derivatives(xs[i - 1] + dt * k1x, vs[i - 1] + dt * k1v)
        xs[i] = xs[i - 1] + (dt_half) * (k1x + k2x)
        vs[i] = vs[i - 1] + (dt_half) * (k1v + k2v)
    return xs, vs


def RK4():
    xs, vs = create_blank_data()
    for i, _ in enumerate(
        tqdm(times_series[1:], desc="Forward Euler"), start=1
    ):
        k1x, k1v = derivatives(xs[i - 1], vs[i - 1])
        k2x, k2v = derivatives(
            xs[i - 1] + 0.5 * dt * k1x, vs[i - 1] + 0.5 * dt * k1v
        )
        k3x, k3v = derivatives(
            xs[i - 1] + 0.5 * dt * k2x, vs[i - 1] + 0.5 * dt * k2v
        )
        k4x, k4v = derivatives(xs[i - 1] + dt * k3x, vs[i - 1] + dt * k3v)

        xs[i] = xs[i - 1] + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
        vs[i] = vs[i - 1] + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return xs, vs


#############################
#        Simulations        #
#############################

xs_exact, vs_exact = exact_solution()
xs_euler_explicit, vs_euler_explicit = euler_explicit()
xs_euler_implicit, vs_euler_implicit = euler_implicit()
xs_RK2, vs_RK2 = RK2()
xs_RK4, vs_RK4 = RK4()

##########################
#        Graphics        #
##########################

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

# Positions vs time
fig_xs, ax_xs = plt.subplots(figsize=(10, 9), layout="constrained", dpi=200)
ax_xs.set_title("Simple harmonic oscillator, different integration methods")
ax_xs.set_xlabel(r"$t$\ [sec]")
ax_xs.set_ylabel(r"$x$\ [m]")
ax_xs.set_xlim(0, times_series[-1])
ax_xs.plot(
    times_series[:-1],
    xs_euler_explicit[:-1],
    "o",
    markersize=2,
    color="red",
)
ax_xs.plot(
    times_series[:-1],
    xs_euler_implicit[:-1],
    "o",
    markersize=2,
    color="blue",
)
ax_xs.plot(
    times_series[:-1],
    xs_RK2[:-1],
    "o",
    markersize=2,
    color="green",
)
ax_xs.plot(
    times_series[:-1],
    xs_RK4[:-1],
    "o",
    markersize=2,
    color="purple",
)
ax_xs.plot(times_series[:-1], xs_exact[:-1], "black")
plt.savefig("pos_vs_time_full.png", bbox_inches="tight")
plt.close()

# Velocities vs time
fig_vs, ax_vs = plt.subplots(figsize=(10, 9), layout="constrained", dpi=200)
ax_vs.set_title("Simple harmonic oscillator, different integration methods")
ax_vs.set_xlabel(r"$t$\ [sec]")
ax_vs.set_ylabel(r"$v$\ [m/s]")
ax_vs.set_xlim(0, times_series[-1])
ax_vs.plot(
    times_series[:-1],
    vs_euler_explicit[:-1],
    "o",
    markersize=2,
    color="red",
)
ax_vs.plot(
    times_series[:-1],
    vs_euler_implicit[:-1],
    "o",
    markersize=2,
    color="blue",
)
ax_vs.plot(
    times_series[:-1],
    vs_RK2[:-1],
    "o",
    markersize=2,
    color="green",
)
ax_vs.plot(
    times_series[:-1],
    vs_RK4[:-1],
    "o",
    markersize=2,
    color="purple",
)
ax_vs.plot(times_series[:-1], vs_exact[:-1], "black")
plt.savefig("vel_vs_time_full.png", bbox_inches="tight")
plt.close()

# Positions vs time, partial
fig_xs, ax_xs = plt.subplots(figsize=(10, 9), layout="constrained", dpi=200)
ax_xs.set_title("Simple harmonic oscillator, different integration methods")
ax_xs.set_xlabel(r"$t$\ [sec]")
ax_xs.set_ylabel(r"$x$\ [m]")

i_max = 100
ax_xs.set_xlim(0, times_series[i_max])
ax_xs.plot(
    times_series[:i_max],
    xs_euler_explicit[:i_max],
    "o",
    markersize=2,
    color="red",
)
ax_xs.plot(
    times_series[:i_max],
    xs_euler_implicit[:i_max],
    "o",
    markersize=2,
    color="blue",
)
ax_xs.plot(
    times_series[:i_max],
    xs_RK2[:i_max],
    "o",
    markersize=2,
    color="green",
)
ax_xs.plot(
    times_series[:i_max],
    xs_RK4[:i_max],
    "o",
    markersize=2,
    color="purple",
)
ax_xs.plot(times_series[:i_max], xs_exact[:i_max], "black")
plt.savefig("pos_vs_time_partial.png", bbox_inches="tight")
plt.close()

# Velocity vs time, partial
fig_vs, ax_vs = plt.subplots(figsize=(10, 9), layout="constrained", dpi=200)
ax_vs.set_title("Simple harmonic oscillator, different integration methods")
ax_vs.set_xlabel(r"$t$\ [sec]")
ax_vs.set_ylabel(r"$v$\ [m]")

i_max = 100
ax_vs.set_xlim(0, times_series[i_max])
ax_vs.plot(
    times_series[:i_max],
    vs_euler_explicit[:i_max],
    "o",
    markersize=2,
    color="red",
)
ax_vs.plot(
    times_series[:i_max],
    vs_euler_implicit[:i_max],
    "o",
    markersize=2,
    color="blue",
)
ax_vs.plot(
    times_series[:i_max],
    vs_RK2[:i_max],
    "o",
    markersize=2,
    color="green",
)
ax_vs.plot(
    times_series[:i_max],
    vs_RK4[:i_max],
    "o",
    markersize=2,
    color="purple",
)
ax_vs.plot(times_series[:i_max], vs_exact[:i_max], "black")
plt.savefig("vel_vs_time_partial.png", bbox_inches="tight")
plt.close()
