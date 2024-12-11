import matplotlib.pyplot as plt
import numpy as np
from ideal_gas import Container, Particle, Simulation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from tqdm import tqdm

# Container
L = 500
dL = 50
container = Container(np.array([L, L, L]))

# Particles
N = 15
x = np.linspace(dL, L - dL, N)
y = np.linspace(dL, L - dL, N)
xv, yv = np.meshgrid(x, y)
coordinates = np.column_stack((xv.ravel(), yv.ravel()))
hlid = N**2 // 2
particle_list: list[Particle] = [
    Particle(
        id=id,
        container=container,
        pos=np.append(xy, L / 2),
        rad=5,
        color="#00AAFF",
    )
    for id, xy in enumerate(coordinates)
]
particle_list[hlid].color = "#FF0000"
particle_list[hlid].vel = np.array([500, 300, 0.0])

# Simulation
max_t = 100.0
dt = 0.025
simulation = Simulation(container, particle_list, dt, max_t)
simulation.run()

# Graphics
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")
frames_label = ax.annotate(
    f"frame: 0/{simulation.num_steps:04d}", xy=(10, L - 10)
)
circles = [
    Circle(
        pos[:2],
        particle.rad,
        facecolor=particle.color,
        edgecolor="black",
        lw=1,
    )
    for particle, pos in zip(particle_list, simulation.pos_matrix[0])
]
for circle in tqdm(circles, desc="Drawing first frame"):
    ax.add_patch(circle)


# Create animation
def update_sphere_animation(frame):
    for pos, circle in zip(simulation.pos_matrix[frame], circles):
        circle.set_center(pos[:2])
    frames_label.set_text(f"frame: {frame:04d}/{simulation.num_steps:04d}")
    return circles + [frames_label]


animation = FuncAnimation(
    fig=fig,
    func=update_sphere_animation,
    frames=simulation.num_steps,
    interval=0,
)

plt.show()

# Average kintetic energy over time
speeds = np.linalg.norm(simulation.vel_matrix, axis=2)
Ek = 0.5 * np.mean(speeds, axis=1) ** 2
plt.rcParams["text.usetex"] = True
fig, ax = plt.subplots()
ax.set_xlim(0, max_t)
ax.set_title("Average kinetic energy over time", fontsize=30)
ax.set_xlabel("Time", fontsize=20)
ax.set_ylabel(r"$\langle E_{k}\rangle$", rotation=0, fontsize=20)
ax.plot(simulation.time_series, Ek)
plt.show()

# Speeds histogram over time
num_bins = 20
histograms = np.zeros((simulation.num_steps, num_bins))
bin_edges = np.arange(0, num_bins)
for step, time in enumerate(simulation.time_series):
    histograms[step], bin_edges = np.histogram(
        speeds[step], bins=num_bins, density=True
    )

fig, ax = plt.subplots()
ax.set_xlim(0, bin_edges[-1])
ax.set_ylim(0, 0.025)
ax.set_xlabel("Speeds")
ax.set_ylabel("Frequency")
hist_bars = ax.bar(bin_edges[:-1], histograms[0], width=3)


def update_histogram_animation(frame):
    for i, rect in enumerate(hist_bars):
        rect.set_height(histograms[frame][i])
    return hist_bars


animation = FuncAnimation(
    fig=fig,
    func=update_histogram_animation,
    frames=simulation.num_steps,
    interval=0,
)

plt.show()
