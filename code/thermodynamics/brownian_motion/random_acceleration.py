import matplotlib.pyplot as plt
import numpy as np

num_steps = 1000
time_series = np.linspace(0, 1, num_steps)
dt = 1.0e-2

pos = np.zeros((num_steps, 2))
vel = np.zeros((num_steps, 2))
acc = np.random.normal(size=(num_steps - 1, 2))

for step, time in enumerate(time_series[1:], start=1):
    vel[step] = vel[step - 1] + dt * acc[step - 1]
    pos[step] = pos[step - 1] + dt * vel[step - 1]


fig, ax = plt.subplots()
ax.set_title("Brownian acceleration")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot(pos[:, 0], pos[:, 1])
ax.grid()
plt.show()
