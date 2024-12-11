import numpy as np
from ideal_gas import Container, Particle, Simulation, visualize_3D

# Container
L = 500
dL = 50
container = Container(np.array([L, L, L]))

# Particles
N = 5
x = np.linspace(dL, L - dL, N)
y = np.linspace(dL, L - dL, N)
z = np.linspace(dL, L - dL, N)
xv, yv, zv = np.meshgrid(x, y, z)
coordinates = np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))
particle_list: list[Particle] = [
    Particle(
        id=id,
        container=container,
        pos=xyz,
        vel=np.random.uniform(-50, 50, 3),
        rad=20,
        color="#FF0000",
    )
    for id, xyz in enumerate(coordinates)
]

# Simulation
max_t = 100.0
dt = 0.1
simulation = Simulation(container, particle_list, dt, max_t)
simulation.run()

# Show
cpos = [
    (4 * L, 6 * L, 4000.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
]
visualize_3D(simulation, cpos)
