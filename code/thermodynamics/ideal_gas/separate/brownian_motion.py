import matplotlib.pyplot as plt
import numpy as np
from ideal_gas import Container, Particle, Simulation
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle
from tqdm import tqdm

# Container
L = 500
dL = 50
container = Container(np.array([L, L, L]))

# Particles
N = 34
x = np.linspace(dL, L - dL, N)
y = np.linspace(dL, L - dL, N)
xv, yv = np.meshgrid(x, y)
coordinates = np.column_stack((xv.ravel(), yv.ravel()))
bparticle_id = (N**2 + N) // 2
particle_list: list[Particle] = [
    Particle(
        id=id,
        container=container,
        pos=np.append(xy, L / 2),
        vel=np.append(np.random.uniform(-50, 50, 2), 0),
        rad=1,
        color="#00AAFF",
    )
    for id, xy in enumerate(coordinates)
]
particle_list[bparticle_id].vel = np.zeros(3)
particle_list[bparticle_id].mass = 25.0
particle_list[bparticle_id].rad = 10.0
particle_list[bparticle_id].color = "#FF7799"

# Simulation
max_t = 20.0
dt = 0.05
simulation = Simulation(container, particle_list, dt, max_t)
simulation.run()

# Booleans
plot_all_particles = True
show_all_particles = True
save_all_particles_video = True
plot_brownian_particle = False

# Graphics
if plot_all_particles:
    fig, ax = plt.subplots()
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    motion_line = ax.plot(
        simulation.pos_matrix[0, bparticle_id, 0],
        simulation.pos_matrix[0, bparticle_id, 1],
        color="red",
    )[0]
    frames_label = ax.annotate(
        f"frame: 0/{simulation.num_steps:04d}", xy=(10, L - 30)
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
            motion_line.set_data(
                simulation.pos_matrix[:frame, bparticle_id, 0],
                simulation.pos_matrix[:frame, bparticle_id, 1],
            )
        frames_label.set_text(f"frame: {frame:04d}/{simulation.num_steps:04d}")
        return circles + [frames_label] + [motion_line]

    animation = FuncAnimation(
        fig=fig, func=update_animation, frames=simulation.num_steps, interval=1
    )
    if show_all_particles:
        plt.show()

    if save_all_particles_video:
        writervideo = FFMpegWriter(fps=30)
        save_pbar = lambda _i, _n: progress_bar.update(1)
        with tqdm(
            total=simulation.num_steps, desc="Saving video"
        ) as progress_bar:
            animation.save(
                "figures/brownian_motion_2.mp4",
                writer=writervideo,
                dpi=150,
                progress_callback=save_pbar,
            )

# Plot just the Brownian particle's motion
if plot_brownian_particle:
    fig, ax = plt.subplots()
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.plot(
        simulation.pos_matrix[:, bparticle_id, 0],
        simulation.pos_matrix[:, bparticle_id, 1],
        lw=2,
    )
    plt.show()
