# from random import choice as random_choice
from sys import argv

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import XKCD_COLORS
from matplotlib.patches import Circle
from tqdm import tqdm

# Time-related constants, variables and parameters
num_steps = 10000
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
colors = {name[5:]: HTML_val for name, HTML_val in XKCD_COLORS.items()}
colors_list = list(colors.values())


# Math and physics functions
def minmax(lst):
    min = np.min(lst)
    max = np.max(lst)
    return np.array([min, max])


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


def distance_sqr(p1, p2):
    return np.dot(p2 - p1, p2 - p1)


def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def angle_between(v1, v2):
    return np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )


def rotation_xy(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotate(vec, th):
    return np.dot(rotation_xy(th), vec)


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
        color=colors["light red"],
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

    def calc_ecc_vec(self, massive_object, idx=0):
        mu = G * (massive_object.mass + self.mass)
        r = self.pos[idx] - massive_object.pos[idx]
        v = self.vel[idx] - massive_object.vel[idx]
        h = np.cross(r, v)
        self.ecc_vec = np.cross(v, h) / mu - normalize(r)

    def save_orbital_data(self, filename, massive_object, idx=0):
        global_params = np.array([G])
        own_params = np.array([self.mass, self.rad])
        massive_obj_params = np.array(
            [massive_object.mass, massive_object.rad]
        )
        pos = self.pos - massive_object.pos[idx]
        vel = self.vel - massive_object.vel[idx]
        np.savez(
            filename,
            global_params=global_params,
            own_params=own_params,
            massive_obj_params=massive_obj_params,
            pos=pos,
            vel=vel,
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


def set_filename(ecc):
    if argv[1] == "":
        filename = f"orbit_e_{ecc:0.3f}"
    else:
        filename = argv[1]
    return f"data/{filename}.npz"


def run():
    global current_step
    for current_step, time in enumerate(tqdm(time_series)):
        verlet_1()
        apply_forces()
        verlet_2()


if __name__ == "__main__":
    star = Particle(id=0, mass=1.0e6)
    planet = Particle(
        id=1,
        pos_0=np.random.uniform(-1000, 1000, 3),
        vel_0=np.random.uniform(-30, 30, 3),
    )

    particles = [star, planet]

    # num_particles = 10
    # particles = [
    #     Particle(
    #         id=id,
    #         pos_0=np.append(np.random.uniform(low=-500, high=500, size=2), 0),
    #         vel_0=np.append(np.random.uniform(low=-50, high=50, size=2), 0),
    #         mass=np.random.uniform(1.0e-4, 1.0e-3),
    #         color=random_choice(colors_list),
    #     )
    #     for id in range(num_particles)
    # ]
    # particles.append(
    #     Particle(
    #         mass=1.0e6,
    #         color=colors["red"],
    #     )
    # )

    run()

    planet.calc_ecc_vec(star)
    ecc = np.linalg.norm(planet.ecc_vec)
    cont = input(
        f"Calculated eccentricity is {ecc:0.3f}. Continue? (default=yes) "
    )
    if cont in ["y", "yes", ""]:
        filename = set_filename(ecc)
        planet.save_orbital_data(filename, star)
        print(f"saved data to {filename}")
    else:
        print("data discarded")
