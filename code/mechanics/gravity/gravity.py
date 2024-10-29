import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import XKCD_COLORS
from tqdm import tqdm

# Time-related constants, variables and parameters
num_steps = 5000
dt = 0.01
max_t = num_steps * dt
time_series = np.arange(0, max_t, dt)

# Other constants, variables and parameters
G = 1.0
ZERO_VEC = np.zeros(3)
X_, Y_, Z_ = np.identity(3)

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
    r_vec = p2.pos[-1] - p1.pos[-1]
    r_dir = normalize(r_vec)
    r2 = np.dot(r_vec, r_vec)
    return 1.0 * (G * p1.mass * p2.mass / r2) * r_dir


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
        self.pos = np.zeros((1, 3))
        self.pos[0] = pos_0

        self.vel = np.zeros((1, 3))
        self.vel[0] = vel_0

        self.acc = np.zeros((1, 3))
        self.force = np.zeros((1, 3))

        self.mass = mass
        self.mass_inv = 1.0 / mass
        self.rad = rad

        self.color = color
        self.id = id

    def prepare_next_step(self):
        self.pos = np.vstack((self.pos, self.pos[-1]))
        self.vel = np.vstack((self.vel, self.vel[-1]))
        self.acc = np.vstack((self.acc, np.copy(ZERO_VEC)))
        self.force = np.vstack((self.force, np.copy(ZERO_VEC)))

    def add_force(self, force):
        self.force[-1] += force

    def verlet_1(self):
        self.pos[-1] = (
            self.pos[-2] + self.vel[-2] * dt + 0.5 * self.acc[-2] * dt**2
        )

    def verlet_2(self):
        self.acc[-1] = self.force[-1] * self.mass_inv
        self.vel[-1] = self.vel[-2] + 0.5 * (self.acc[-2] + self.acc[-1]) * dt


def prepare_next_step():
    for particle in particles:
        particle.prepare_next_step()


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
    for t in tqdm(time_series):
        prepare_next_step()
        apply_forces()
        verlet_1()
        verlet_2()


def prepare_graphics():
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title("Gravity test")
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.grid(True)
    ax.plot(
        particles[0].pos[:, 0], particles[0].pos[:, 1], c=particles[0].color
    )
    ax.plot(
        particles[1].pos[:, 0], particles[1].pos[:, 1], c=particles[1].color
    )
    plt.show()


if __name__ == "__main__":
    star = Particle(id=0, mass=1.0e3, rad=3.0, color=Colors["orange"])
    planet = Particle(
        id=1,
        pos_0=100 * X_,
        vel_0=2 * Y_,
        mass=1.0e3,
        rad=3.0,
        color=Colors["blue"],
    )
    particles = [star, planet]

    run()
    prepare_graphics()
    # print(particles[1].pos)
