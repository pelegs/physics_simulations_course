import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from lib.Particle import Particle
from lib.Simulation import Boundary, Simulation
from lib.visualize import Visualization2D

if __name__ == "__main__":
    print("Testing Simulation class")
    L = 500.0
    simulation = Simulation(
        dt=0.05,
        max_t=100.0,
        sides=[L, L, L],
        boundaries=[Boundary.WALL, Boundary.WALL, Boundary.EMPTY],
    )

    num_paricles: int = 100
    masses = np.ones(num_paricles)
    masses[::5] = 10
    radii = np.ones(num_paricles) * 5
    radii[::5] = 10
    colors = ["#00FF00"] * num_paricles
    red_colors = ["#FF0000"] * (num_paricles // 5)
    colors[::5] = red_colors
    particles = [
        Particle(
            vel=np.append(np.random.uniform(-20, 20, 2), 0),
            rad=rad,
            mass=mass,
            color=color,
        )
        for mass, rad, color in zip(masses, radii, colors)
    ]
    simulation.add_objects(particles)
    simulation.setup_system()
    simulation.put_particles_on_grid(
        Ns=np.array([10, 10, 1]), dL=np.array([20.0, 20.0, 0.0])
    )
    simulation.run()

    visualization: Visualization2D = Visualization2D(simulation)
    visualization.create_animation()
    plt.show()
    visualization.save_animation("videos/simple_test_1.mp4")
