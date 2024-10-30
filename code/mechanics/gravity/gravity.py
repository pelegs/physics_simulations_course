import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import XKCD_COLORS
from tqdm import tqdm

# Time-related constants, variables and parameters
num_steps = 15000
dt = 0.01
max_t = num_steps * dt
time_series = np.arange(0, max_t, dt)
current_step = 0

# Other constants, variables and parameters
G = 1.0
ZERO_VEC = np.zeros(3)
X_, Y_, Z_ = np.identity(3)
ZERO_ARRAY = np.zeros((num_steps, 3))
EMPTY_ARRAY = np.empty((num_steps, 3))
EMPTY_ARRAY[:] = np.nan

# Colors
Colors = {name[5:]: HTML_val for name, HTML_val in XKCD_COLORS.items()}


# Math and physics functions
def normalize(vec):
    L = np.linalg.norm(vec)
    if L == 0:
        raise ValueError("Can't normalize the zero vector")
    return vec / L


def scale(vec, norm):
    return norm * normalize(vec)


def look_at(v1, v2):
    """Returns a unit vector in the direction from v1 to v2."""
    return normalize(v2 - v1)


def sqr_dist(p1, p2):
    return np.dot(p2 - p1, p2 - p1)


def gravity(p1, p2):
    r_vec = p2.pos[current_step] - p1.pos[current_step]
    r_dir = normalize(r_vec)
    r2 = np.dot(r_vec, r_vec)
    F = 1.0 * (G * p1.mass * p2.mass / r2) * r_dir
    return F


# Objects with mass
class Particle:
    def __init__(
        self,
        pos_0=np.copy(ZERO_VEC),
        vel_0=np.copy(ZERO_VEC),
        mass=1.0,
        rad=1.0,
        color=Colors["light red"],
        id=-1,
    ):
        self.pos = np.copy(EMPTY_ARRAY)
        self.pos[0] = pos_0

        self.vel = np.copy(ZERO_ARRAY)
        self.vel[0] = vel_0

        self.acc = np.copy(ZERO_ARRAY)
        self.force = np.copy(ZERO_ARRAY)

        self.mass = mass
        self.mass_inv = 1.0 / mass
        self.rad = rad

        self.color = color
        self.id = id

    def add_force(self, force):
        self.force[current_step] += force

    def verlet_1(self):
        if current_step >= 1:
            self.pos[current_step] = (
                self.pos[current_step - 1]
                + self.vel[current_step - 1] * dt
                + 0.5 * self.acc[current_step - 1] * dt**2
            )

    def verlet_2(self):
        if current_step >= 1:
            self.acc[current_step] = self.force[current_step] * self.mass_inv
            self.vel[current_step] = (
                self.vel[current_step - 1]
                + 0.5
                * (self.acc[current_step - 1] + self.acc[current_step])
                * dt
            )


def apply_forces():
    for p1_idx, particle_1 in enumerate(particles):
        for particle_2 in particles[p1_idx + 1 :]:
            grav = gravity(particle_1, particle_2)
            particle_1.add_force(grav)
            particle_2.add_force(-1.0 * grav)


def verlet_1():
    for particle in particles:
        particle.verlet_1()


def verlet_2():
    for particle in particles:
        particle.verlet_2()


def run():
    global current_step
    for current_step, time in enumerate(tqdm(time_series)):
        verlet_1()
        apply_forces()
        verlet_2()


def prepare_graphics():
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title("Gravity test")
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    for particle in particles:
        ax.plot(particle.pos[:, 0], particle.pos[:, 1], c=particle.color)
    plt.show()


if __name__ == "__main__":
    star = Particle(id=0, mass=1.0e3, rad=3.0, color=Colors["orange"])
    planet = Particle(
        id=1,
        pos_0=100 * X_,
        vel_0=0.7 * np.sqrt(G * star.mass / 100) * Y_,
        mass=1.0e-4,
        rad=3.0,
        color=Colors["blue"],
    )
    particles = [star, planet]

    # num_particles = 10
    # particles = [
    #     Particle(
    #         id=id,
    #         pos_0=np.append(np.random.uniform(low=-500, high=500, size=2), 0),
    #         mass=np.random.uniform(1.0e2, 1.0e6),
    #     )
    #     for id in range(num_particles)
    # ]

    run()
    prepare_graphics()
