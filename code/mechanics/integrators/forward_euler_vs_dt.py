import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#########################
#        General        #
#########################

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

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

# Starting conditions
x0 = A
v0 = 0.0

# Time related
max_t = 100.0

#############################
#        Integrators        #
#############################


def derivatives(x, v):
    return v, -(k / m) * x


def create_blank_data(dt):
    time_series = np.arange(0, max_t, dt)
    num_steps = time_series.shape[0]
    data = np.zeros((2, num_steps))
    data[:, 0] = x0, v0
    return time_series, data


def exact_solution(dt):
    time_series, data = create_blank_data(dt)
    data[X] = A * np.cos(omega * time_series + phi)
    data[V] = -A * omega * np.sin(omega * time_series + phi)
    return time_series, data


def euler_explicit(dt):
    time_series, data = create_blank_data(dt)
    for i, _ in enumerate(tqdm(time_series[1:], desc=f"dt={dt}"), start=1):
        a = -(k / m) * data[X, i - 1]
        data[V, i] = data[V, i - 1] + a * dt
        data[X, i] = data[X, i - 1] + data[V, i - 1] * dt

    return time_series, data


if __name__ == "__main__":
    dt_list = [0.1, 0.05, 0.01, 0.005, 0.001]
    all_times = list()
    all_xs = list()
    for dt in dt_list:
        time, xs = euler_explicit(dt)
        all_times.append(time)
        all_xs.append(xs)
    exact_time, exact_xs = exact_solution(dt_list[-1])

    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained", dpi=200)
    ax.set_title(r"Forward (excplicit) Euler for different $\Delta t$ values")
    ax.set_xlabel(r"$t$\ [s]")
    ax.set_ylabel(r"$x$\ [m]", rotation=0)
    ax.set_xlim(0, exact_time[-1])

    for dt, ts, xs in zip(dt_list, all_times, all_xs):
        ax.plot(
            ts,
            xs[X],
            "o",
            markersize=2,
            label=rf"${dt}$",
        )
    ax.plot(
        exact_time,
        exact_xs[X],
        "#AAAAAA",
        label="Precise solution",
    )

    ax.legend(
        title=r"$\Delta t$\ [s]",
        loc=2,
        fancybox=True,
    )

    plt.savefig(
        "explicit_euler_different_dt.png",
        bbox_inches="tight",
    )
