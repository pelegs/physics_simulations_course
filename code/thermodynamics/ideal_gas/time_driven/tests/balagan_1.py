import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from lib.constants import Axes, npdarr
from lib.functions import distance
from lib.Particle import Particle
from lib.Simulation import Boundary, Simulation
from lib.visualize import Visualization2D

if __name__ == "__main__":
    print("Testing Simulation class")
    L = 500.0
    simulation = Simulation(
        dt=0.05,
        max_t=10.0,
        sides=[L, L, L],
        boundaries=[Boundary.WALL, Boundary.WALL, Boundary.EMPTY],
    )

    N = 20
    num_paricles: int = N**2
    particles: list[Particle] = list()
    i = 1
    while i < num_paricles:
        add: bool = True
        new_particle: Particle = Particle(
            pos=np.append(
                np.random.uniform(
                    (25, 25),
                    (
                        simulation.sides[Axes.X] - 25,
                        simulation.sides[Axes.Y] - 25,
                    ),
                    size=2,
                ),
                0,
            ),
            vel=np.append(np.random.uniform(-10, 10, 2), 0),
            rad=np.random.uniform(2.5, 5),
            mass=1,
            color=f"#{os.urandom(3).hex().upper()}",
        )
        for p2 in particles:
            if distance(new_particle.pos, p2.pos) < (
                new_particle.rad + p2.rad
            ):
                add = False
                break
        if add:
            particles.append(new_particle)
            i += 1

    simulation.add_objects(particles)
    simulation.setup_system()
    simulation.run()

    visualization: Visualization2D = Visualization2D(simulation)
    visualization.create_animation()
    plt.show()
    visualization.save_animation("videos/balagan_1.mp4")
