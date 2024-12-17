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
        boundaries=[Boundary.PERIODIC, Boundary.PERIODIC, Boundary.EMPTY],
    )

    N: int = 30
    num_paricles: int = N**2
    particles = [
        Particle(
            vel=np.append(np.random.uniform(-10, 10, 2), 0),
            rad=1,
            mass=1.0,
            color="#00AAAA",
        )
        for _ in range(num_paricles)
    ]
    center_idx = N * (N + 1) // 2
    bpos = np.copy(particles[center_idx].pos)
    brownian_particle = Particle(pos=bpos, mass=25, rad=10, color="#FF0077")
    particles[center_idx] = brownian_particle
    simulation.add_objects(particles)
    simulation.setup_system()
    simulation.put_particles_on_grid(
        Ns=np.array([N, N, 1]), dL=np.array([10.0, 10.0, 0.0])
    )
    simulation.run()

    visualization: Visualization2D = Visualization2D(simulation)
    visualization.create_animation()
    # plt.show()
    visualization.save_animation("videos/brownian_motion_1.mp4")
