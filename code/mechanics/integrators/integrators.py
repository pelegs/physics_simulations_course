import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

############################
#        Parameters        #
############################

# Indexing
X, V = 0, 1

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


def derivatives(x, v):
    return v, -(k / m) * x


def create_blank_data():
    data = np.zeros((2, num_steps))
    data[:, 0] = x0, v0
    return data


def exact_solution():
    data = create_blank_data()
    data[X] = A * np.cos(omega * times_series + phi)
    data[V] = -A * omega * np.sin(omega * times_series + phi)
    return data


def euler_explicit():
    data = create_blank_data()
    for i, _ in enumerate(
        tqdm(times_series[1:], desc="Explicit Euler"), start=1
    ):
        a = -(k / m) * data[X, i - 1]
        data[V, i] = data[V, i - 1] + a * dt
        data[X, i] = data[X, i - 1] + data[V, i - 1] * dt

    return data


def euler_implicit():
    c1 = 1 + (dt**2 * k / m)
    c2 = dt * k / m
    data = create_blank_data()
    for i, _ in enumerate(
        tqdm(times_series[1:], desc="Implicit Euler"), start=1
    ):
        data[X, i] = (data[X, i - 1] + dt * data[V, i - 1]) / c1
        data[V, i] = data[V, i - 1] - c2 * data[X, i]

    return data


def RK2():
    data = create_blank_data()
    for i, _ in enumerate(
        tqdm(times_series[1:], desc="Forward Euler"), start=1
    ):
        k1x, k1v = derivatives(data[X, i - 1], data[V, i - 1])
        k2x, k2v = derivatives(
            data[X, i - 1] + dt * k1x, data[V, i - 1] + dt * k1v
        )
        data[X, i] = data[X, i - 1] + (dt_half) * (k1x + k2x)
        data[V, i] = data[V, i - 1] + (dt_half) * (k1v + k2v)
    return data


def RK4():
    data = create_blank_data()
    for i, _ in enumerate(
        tqdm(times_series[1:], desc="Forward Euler"), start=1
    ):
        k1x, k1v = derivatives(data[X, i - 1], data[V, i - 1])
        k2x, k2v = derivatives(
            data[X, i - 1] + 0.5 * dt * k1x, data[V, i - 1] + 0.5 * dt * k1v
        )
        k3x, k3v = derivatives(
            data[X, i - 1] + 0.5 * dt * k2x, data[V, i - 1] + 0.5 * dt * k2v
        )
        k4x, k4v = derivatives(
            data[X, i - 1] + dt * k3x, data[V, i - 1] + dt * k3v
        )

        data[X, i] = data[X, i - 1] + (dt / 6) * (
            k1x + 2 * k2x + 2 * k3x + k4x
        )
        data[V, i] = data[V, i - 1] + (dt / 6) * (
            k1v + 2 * k2v + 2 * k3v + k4v
        )
    return data


#############################
#        Simulations        #
#############################

data_exact = exact_solution()
data_euler_explicit = euler_explicit()
data_euler_implicit = euler_implicit()
data_RK2 = RK2()
data_RK4 = RK4()

########################
#        Errors        #
########################


def error(method, exact):
    diff = np.abs(method - exact)
    global_error = np.sum(diff, axis=1)
    local_error = np.mean(diff, axis=1)
    return global_error, local_error


global_error_euler_explicit, local_error_euler_explicit = error(
    data_euler_explicit, data_exact
)


##########################
#        Graphics        #
##########################

extra_info_filename = lambda i: f"_first_{i}_steps_only" if i != -1 else ""
var_params = {
    X: {
        "title": "Simple harmonic oscillator, different integration methods, "
        "position vs. time",
        "xlabel": r"$t$\ [s]",
        "ylabel": r"$x$\ [m]",
        "filename": "pos_vs_time_different_integrators",
    },
    V: {
        "title": "Simple harmonic oscillator, different integration methods, "
        "velocity vs. time",
        "xlabel": r"$t$\ [s]",
        "ylabel": r"$v$\ [m/s]",
        "filename": "vel_vs_time_different_integrators",
    },
}

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})


def generate_figure(var, last_step=-1):
    param_dict = var_params[var]
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained", dpi=200)
    ax.set_title(param_dict["title"])
    ax.set_xlabel(param_dict["xlabel"])
    ax.set_ylabel(param_dict["ylabel"])
    ax.set_xlim(0, times_series[last_step])
    ax.plot(
        times_series[:last_step],
        data_exact[var, :last_step],
        "#AAAAAA",
        label="Precise solution",
    )
    ax.plot(
        times_series[:last_step],
        data_euler_explicit[var, :last_step],
        "o",
        markersize=1,
        color="red",
        label="Explicit (forward) Euler",
    )
    ax.plot(
        times_series[:last_step],
        data_euler_implicit[var, :last_step],
        "o",
        markersize=1,
        color="blue",
        label="Implicit (backwards) Euler",
    )
    ax.plot(
        times_series[:last_step],
        data_RK2[var, :last_step],
        "o",
        markersize=1,
        color="green",
        label="Runge-Kutta 2 (explicit)",
    )
    ax.plot(
        times_series[:last_step],
        data_RK4[var, :last_step],
        "o",
        markersize=1,
        color="purple",
        label="Runge-Kutta 4 (explicit)",
    )
    plt.legend()
    plt.savefig(
        f"{param_dict['filename']}{extra_info_filename(last_step)}.png",
        bbox_inches="tight",
    )
    plt.close()


######################
#        MAIN        #
######################

generate_figure(var=X)
generate_figure(var=V)
generate_figure(var=X, last_step=100)
generate_figure(var=V, last_step=100)
