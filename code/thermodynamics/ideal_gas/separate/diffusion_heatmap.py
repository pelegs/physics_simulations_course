import matplotlib.pyplot as plt
import numpy as np
from ideal_gas import Container, Particle, Simulation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from tqdm import tqdm

# Container
L = 2500
dL = 50
container = Container(np.array([L, L, L]))

# Particles
N = 25
x = np.linspace(dL, L / 5 - dL, N)
y = np.linspace(dL, L / 5 - dL, N)
xv, yv = np.meshgrid(x, y)
coordinates = np.column_stack((xv.ravel(), yv.ravel()))
particle_list: list[Particle] = [
    Particle(
        id=id,
        container=container,
        pos=np.append(xy, L / 2),
        vel=np.append(np.random.uniform(-500, 500, 2), 0),
        rad=2,
        color="#00AAFF",
    )
    for id, xy in enumerate(coordinates)
]

# Simulation
max_t = 5.0
dt = 0.025
simulation = Simulation(container, particle_list, dt, max_t)
simulation.run()

# Graphics
visualize = False
heatmap = True

# Visualization
if visualize:
    fig, ax = plt.subplots()
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    frames_label = ax.annotate(
        f"frame: 0/{simulation.num_steps:04d}", xy=(10, L - 50)
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

# Diffusion heatmap
if heatmap:
    num_bins = 7
    histograms = np.zeros((simulation.num_steps, num_bins, num_bins))
    edges = np.zeros(num_bins)
    for step, time in enumerate(simulation.time_series):
        histograms[step], edges, _ = np.histogram2d(
            simulation.pos_matrix[step, :, 0],
            simulation.pos_matrix[step, :, 1],
            bins=num_bins,
        )

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    heatmap = ax.imshow(histograms[0], cmap="turbo")

    def update_heatmap_animation(frame):
        heatmap.set_data(histograms[frame])
        return [heatmap]

    animation = FuncAnimation(
        fig=fig,
        func=update_heatmap_animation,
        frames=simulation.num_steps,
        interval=0,
    )

    plt.show()
