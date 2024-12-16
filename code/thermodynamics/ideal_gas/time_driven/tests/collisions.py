import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from lib.Particle import Particle
from lib.Simulation import Boundary, Simulation

if __name__ == "__main__":
    print("Testing Simulation class")
    L = 500.0
    simulation = Simulation(
        dt=0.05,
        max_t=50.0,
        sides=[L, L, L],
        boundaries=[Boundary.WALL, Boundary.WALL, Boundary.EMPTY],
    )

    num_paricles: int = 100
    particles = [
        Particle(
            vel=np.append(np.random.uniform(-20, 20, 2), 0),
            rad=2.5,
            mass=1,
        )
        for _ in range(num_paricles)
    ]
    for i, particle in enumerate(particles):
        simulation.add_object(particle)
        if i % 5 == 0:
            particle.rad = 5
            particle.mass = 20
            particle.color = "#00FF00"
    simulation.setup_system()
    simulation.put_particles_on_grid(
        Ns=np.array([10, 10, 1]), dL=np.array([20.0, 20.0, 0.0])
    )

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
        for particle, pos in zip(
            simulation.particle_list, simulation.pos_matrix[0]
        )
    ]
    for circle in circles:
        ax.add_patch(circle)

    # Create animation
    def update_sphere_animation(frame):
        for pos, circle in zip(
            simulation.pos_matrix[frame],
            circles,
        ):
            circle.set_center(pos[:2])
        frames_label.set_text(f"frame: {frame:04d}/{simulation.num_steps:04d}")
        return [frames_label]

    # Run simulation and draw graphics
    simulation.run()
    animation = FuncAnimation(
        fig=fig,
        func=update_sphere_animation,
        frames=simulation.num_steps,
        interval=1,
    )
    plt.show()
