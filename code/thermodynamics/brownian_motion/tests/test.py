from sys import argv

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["figure.constrained_layout.use"] = True

num_steps = 1000
num_particles = 5

pos = np.zeros((num_steps, num_particles))
vel = np.random.normal(size=(num_steps - 1, num_particles))
pos[1:, :] = np.cumsum(vel, axis=0)

time = np.linspace(0, 1, num_steps)
MSD = np.zeros(num_steps - 1)
for duration in range(1, num_steps):
    num_windows = num_steps - duration + 1
    SD = np.zeros((num_windows, num_particles))
    for t0 in range(num_windows - 1):
        SD[t0] = (pos[t0 + duration, :] - pos[t0, :]) ** 2
    MSD[duration - 1] = np.mean(np.mean(SD, axis=1), axis=0)

fig, ax = plt.subplots()
ax.set_title("Test", fontsize=25)
ax.set_xlabel("Simulation duration", fontsize=20)
ax.set_ylabel("MSD", fontsize=20)
ax.plot(np.arange(0, num_steps - 1), MSD)
plt.show()
