import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from lib.constants import Axes
from lib.Particle import Particle
from lib.Simulation import Boundary, Simulation

if __name__ == "__main__":
    print("Testing Simulation class")
    L = 500.0
    simulation = Simulation(
        dt=0.05,
        max_t=100.0,
        sides=[L, L, L],
        boundaries=[Boundary.WALL, Boundary.WALL, Boundary.EMPTY],
    )

    num_paricles: int = 10
    particles = [
        Particle(
            pos=np.append(np.random.uniform(20, L - 20, 2), 0),
            vel=np.append(np.random.uniform(-20, 20, 2), 0),
            rad=np.random.randint(10, 20),
        )
        for _ in range(num_paricles)
    ]
    for particle in particles:
        simulation.add_object(particle)
    simulation.setup_system()

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
    rectangles = [
        Rectangle(
            obj.bbox.pts[0],
            obj.bbox.sides[Axes.X],
            obj.bbox.sides[Axes.Y],
            lw=2,
            edgecolor="black",
            facecolor="none",
        )
        for obj in simulation.objects_list
    ]
    for circle, rectangle in zip(circles, rectangles):
        ax.add_patch(circle)
        ax.add_patch(rectangle)
    labels = [
        ax.annotate(f"{i}", obj.pos[:2])
        for i, obj in enumerate(simulation.objects_list)
    ]
    overlaps_label = ax.annotate("overlaps: []", (20, 20))

    # Create animation
    def update_sphere_animation(frame):
        for pos, circle, rectangle, label, particle in zip(
            simulation.pos_matrix[frame],
            circles,
            rectangles,
            labels,
            simulation.particle_list,
        ):
            circle.set_center(pos[:2])
            rectangle.set_xy(pos[:2] - particle.rad)
            label.set_position(pos[:2])
        frames_label.set_text(f"frame: {frame:04d}/{simulation.num_steps:04d}")
        overlaps_label.set_text(
            f"overlaps:\n[{simulation.AABBs_overlaps[frame]}]"
        )
        return [frames_label, overlaps_label]

    # Run simulation and draw graphics
    simulation.run()
    animation = FuncAnimation(
        fig=fig,
        func=update_sphere_animation,
        frames=simulation.num_steps,
        interval=1,
    )
    plt.show()
