# TEMP code to test some stuff

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Physical constants
L = 1.0  # Length in meters
dx = 0.02  # Distance between points
c = 1.0  # Wave speed
dt = 1.0 * dx / c  # CFL
w = 0.1

# Set up points
x = np.arange(0, L * (1 + dx), dx)
num_pts = x.shape[0]
# print(num_pts)
# exit()
# num_steps = 200
y = np.zeros((num_pts, 3))
y[:, 0] = np.sin(2 * np.pi * x / L)
y[:, 0] = 0.75 * np.sin(2 * np.pi * x / L) + 0.25 * np.sin(3 * np.pi * x / L)
y[:, 0] = np.exp(-((x - L / 2) ** 2) / w**2)
y[1:-1, 1] = y[1:-1, 0] + 0.5 * (c * dt / dx) ** 2 * (
    y[2:, 0] + y[:-2, 0] - 2 * y[1:-1, 0]
)

# Set up plot
fig = plt.figure()
axis = plt.axes(xlim=[0, L], ylim=[-2, 2])
(line,) = axis.plot([], [], "-*", lw=2)


# Main loop
def simulate(t):
    y[1:-1, 2] = (
        2 * y[1:-1, 1]
        - y[1:-1, 0]
        + (c * dt / dx) ** 2 * (y[2:, 1] + y[:-2, 1] - 2 * y[1:-1, 1])
    )
    y[1:-1, 0] = y[1:-1, 1]
    y[1:-1, 1] = y[1:-1, 2]

    global line
    line.set_data(x, y[:, 2])

    return (line,)


anim = animation.FuncAnimation(fig, simulate, frames=1, interval=0)
plt.show()
