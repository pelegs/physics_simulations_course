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
N = 20
x = np.linspace(dL, L - dL, N)
y = np.linspace(dL, L - dL, N)
xv, yv = np.meshgrid(x, y)
coordinates = np.column_stack((xv.ravel(), yv.ravel()))
hlid = (N**2 + N) // 2
particle_list: list[Particle] = [
    Particle(
        id=id,
        container=container,
        pos=np.append(xy, L / 2),
        vel=np.append(np.random.uniform(-50, 50, 2), 0),
        rad=3,
        color="#00AAFF",
    )
    for id, xy in enumerate(coordinates)
]
particle_list[hlid].vel = np.zeros(3)
particle_list[hlid].mass = 25.0
particle_list[hlid].rad = 10.0
particle_list[hlid].color = "#FF0000"

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
def update_animation(frame):
    for pos, circle in zip(simulation.pos_matrix[frame], circles):
        circle.set_center(pos[:2])
    frames_label.set_text(f"frame: {frame:04d}/{simulation.num_steps:04d}")
    return circles + [frames_label]


animation = FuncAnimation(
    fig=fig, func=update_animation, frames=simulation.num_steps, interval=0
)

plt.show()
