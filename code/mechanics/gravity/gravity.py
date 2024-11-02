from random import choice as random_choice

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import XKCD_COLORS
from tqdm import tqdm

# Time-related constants, variables and parameters
num_steps = 50000
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


def cross2D(a, b):
    return a[0] * b[1] - a[1] * b[0]


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


def setup_graphics(xs=[-500, 500], ys=[-500, 500]):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title("Gravity test")
    ax.set_xlim(*xs)
    ax.set_ylim(*ys)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    lines = [
        ax.plot(particle.pos[0, 0], particle.pos[0, 1], c=particle.color)[0]
        for particle in particles
    ]
    frames_label = plt.text(
        xs[0],
        ys[1],
        f"frame {0:05d}/{num_steps:05d}",
        fontsize=20,
    )
    return [fig, frames_label] + lines


def animate(frame):
    step = frame * steps_per_frame
    frames_label.set_text(f"frame {step:05d}/{num_steps:05d}")
    for particle, line in zip(particles, lines):
        line.set_xdata(particle.pos[:step, 0])
        line.set_ydata(particle.pos[:step, 1])
    return lines + [frames_label]


def ecc_vec(massive_obj, particle):
    r_vec = particle.pos[current_step] - massive_obj.pos[current_step]
    v_vec = particle.vel[current_step]
    mu = G * massive_obj.mass
    h_vec = np.cross(r_vec, v_vec)
    return np.cross(v_vec, h_vec) / mu - normalize(r_vec)


if __name__ == "__main__":
    star = Particle(id=0, mass=1.0e7, rad=3.0, color=colors["orange"])
    planet = Particle(
        id=1,
        pos_0=100 * X_,
        vel_0=1.0 * np.sqrt(G * star.mass / 100) * Y_,
        mass=1.0e-5,
        rad=3.0,
        color=colors["blue"],
    )
    particles = [star, planet]
    e_vec = ecc_vec(star, planet)
    e = np.linalg.norm(e_vec)
    print(e_vec, e)

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

    # run()

    # fig, frames_label, *lines = setup_graphics(
    #     minmax(planet.pos[:, 0]), minmax(planet.pos[:, 1])
    # )
    # num_frames = 50
    # steps_per_frame = num_steps // num_frames
    # ani = animation.FuncAnimation(fig=fig, func=animate, frames=num_frames)
    # plt.show()
