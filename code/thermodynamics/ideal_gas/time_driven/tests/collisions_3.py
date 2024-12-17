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
        dt=0.1,
        max_t=100.0,
        sides=[L, L, L],
        boundaries=[Boundary.WALL, Boundary.WALL, Boundary.EMPTY],
    )

    num_paricles: int = 100
    particles_1 = [
        Particle(
            vel=np.append(np.random.uniform(-10, 10, 2), 0),
            rad=3,
            mass=1.0,
            color="#00AAAA",
        )
        for _ in range(num_paricles // 2)
    ]
    particles_2 = [
        Particle(
            vel=np.append(np.random.uniform(-10, 10, 2), 0),
            rad=7,
            mass=5.0,
            color="#AA0099",
        )
        for _ in range(num_paricles // 2)
    ]
    simulation.add_objects(particles_1 + particles_2)
    simulation.setup_system()
    simulation.put_particles_on_grid(
        Ns=np.array([10, 10, 1]), dL=np.array([10.0, 10.0, 0.0])
    )
    simulation.run()

    visualization: Visualization2D = Visualization2D(simulation)
    visualization.create_animation()
    plt.show()
    # visualization.save_animation("videos/simple_test_3.mp4")
