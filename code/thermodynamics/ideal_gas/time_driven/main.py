import numpy as np
from lib.Particle import Particle
from lib.Simulation import Simulation

if __name__ == "__main__":
    print("Testing Simulation class")
    L = 500.0
    simulation = Simulation(dt=0.05, max_t=25.0, sides=[L, L, L])

    num_paricles: int = 4
    particles = [
        Particle(
            pos=np.random.randint(0, int(L), size=3).astype(float),
            rad=np.random.randint(1, 10),
        )
        for _ in range(num_paricles)
    ]
    for particle in particles:
        simulation.add_object(particle)
    simulation.setup_system()

    for object in simulation.objects_list:
        print(object)
