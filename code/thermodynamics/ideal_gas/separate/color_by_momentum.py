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
masses = np.random.uniform(1.0, 10.0, N**2)
particle_list: list[Particle] = [
    Particle(
        id=id,
        container=container,
        pos=np.append(xy, L / 2),
        mass=masses[id],
        rad=masses[id],
        color="#00AAFF",
    )
    for id, xy in enumerate(coordinates)
]
particle_list[hlid].color = "#FF0000"
particle_list[hlid].vel = np.array([500, 300, 0.0])

# Simulation
max_t = 50.0
dt = 0.05
simulation = Simulation(container, particle_list, dt, max_t)
simulation.run()

# Speed data
speeds = np.linalg.norm(simulation.vel_matrix, axis=2)
max_speed = np.max(speeds)
cmap = plt.get_cmap("turbo")
norm = plt.Normalize(0, max_speed)

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
    for pos, circle, speed, mass in zip(
        simulation.pos_matrix[frame],
        circles,
        speeds[frame],
        [p.mass for p in particle_list],
    ):
        circle.set_center(pos[:2])
        circle.set_color(cmap(norm(mass * speed)))
    frames_label.set_text(f"frame: {frame:04d}/{simulation.num_steps:04d}")
    return circles + [frames_label]


animation = FuncAnimation(
    fig=fig,
    func=update_sphere_animation,
    frames=simulation.num_steps,
    interval=0,
    blit=True,
)

plt.show()
