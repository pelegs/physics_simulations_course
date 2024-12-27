import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# Time stuff
dt = 0.1
num_steps = 1000
time_series = np.linspace(0, dt * num_steps, num_steps)

# Simulation boundaries
sim_size = 600.0

# Particle stuff
num_particles = 1000
pcolors = np.arange(0, num_particles)
pos = np.zeros((num_steps, num_particles, 2))
pos[0] = np.random.uniform(10, sim_size - 10, (num_particles, 2))
vel = np.zeros((num_steps, num_particles, 2))
acc_brown = np.random.normal(size=(num_steps - 1, num_particles, 2))

# Calculation
for step, time in enumerate(tqdm(time_series[1:]), start=1):
    vel[step] = vel[step - 1] + dt * acc_brown[step - 1]
    pos[step] = (pos[step - 1] + dt * vel[step]) % sim_size


# Graphics
def animate_step(step):
    scatter_plot.set_offsets(pos[step, :])
    return [scatter_plot]


fig, ax = plt.subplots()
ax.set_title("Brownian motion with inertia test")
ax.set_xlabel("x")
ax.set_ylabel("y")
# ax.grid()
ax.set_xlim(0, sim_size)
ax.set_ylim(0, sim_size)
scatter_plot = ax.scatter(
    pos[0, :, 0],
    pos[0, :, 1],
    s=16,
    c=pcolors,
    cmap="prism",
    edgecolors="black",
)
animation = FuncAnimation(
    fig=fig, func=animate_step, frames=num_steps, interval=0, blit=True
)
plt.show()
