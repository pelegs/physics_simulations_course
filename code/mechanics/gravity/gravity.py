from random import choice as random_choice

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import XKCD_COLORS
from matplotlib.patches import Circle
from sympy import lambdify, symbols
from tqdm import tqdm

# Time-related constants, variables and parameters
num_steps = 10000
dt = 0.001
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


def rotation_matrix(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def rotate(vec, th):
    return np.dot(rotation_matrix(th), vec)


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

    def calc_ecc_vec(self, massive_obj):
        self.e_vec = ecc_vec(massive_obj, self)
        self.e_hat = normalize(self.e_vec)
        self.e = np.linalg.norm(self.e_vec)

    def calc_semi_axes(self, massive_obj):
        r_vec = self.pos[current_step] - massive_obj.pos[current_step]
        r = np.linalg.norm(r_vec)
        th = angle_between(r_vec, self.e_vec)
        self.a = r * (1 + self.e * np.cos(th)) / (1 - self.e**2)
        self.b = self.a * np.sqrt(1 - self.e**2)
        self.rp = 2 * self.a / ((1 + self.e) / (1 - self.e) + 1)

    def calc_orbital_points(self, massive_obj):
        self.p1 = massive_obj.pos[current_step] + self.e_hat * self.rp
        self.p2 = self.p1 - 2 * self.e_hat * self.a
        self.c = (self.p1 + self.p2) / 2
        b_hat = np.append(rotate(self.e_hat[:2], np.pi / 2), 0)
        self.p3 = self.c + b_hat * self.b
        self.p4 = self.c - b_hat * self.b

    def get_orbital_points(self, massive_obj):
        self.calc_ecc_vec(massive_obj)
        self.calc_semi_axes(massive_obj)
        self.calc_orbital_points(massive_obj)
        return np.array(
            [
                self.pos[current_step],
                self.p1,
                self.p2,
                self.p3,
                self.p4,
                self.c,
            ]
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


def setup_graphics(xs=[-500, 500], ys=[-500, 500], orbital_pts=None):
    global fig, ax, frames_label, paths, circles, orbital_pts_scatter, contour

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(xs[0] - 10, xs[1] + 10)
    ax.set_ylim(ys[0] - 10, ys[1] + 10)
    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("y", rotation=0, fontsize=15)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.2)

    paths = [
        ax.plot(particle.pos[0, 0], particle.pos[0, 1], c=particle.color)[0]
        for particle in particles
    ]

    circles = [
        Circle(xy=particle.pos[0], radius=particle.rad, color=particle.color)
        for particle in particles
    ]
    for circle in circles:
        ax.add_patch(circle)

    frames_label = plt.text(
        0.05,
        0.95,
        f"frame {0:05d}/{num_steps:05d}",
        fontsize=10,
        transform=ax.transAxes,
    )

    if orbital_pts is not None:
        num_pts = 400
        xrange = np.linspace(xs[0], xs[1], num_pts)
        yrange = np.linspace(ys[0], ys[1], num_pts)
        X, Y = np.meshgrid(xrange, yrange)
        conic = A * X**2 + B * X * Y + C * Y**2 + D * X + E * Y
        contour = ax.contour(X, Y, (conic), [1], alpha=0.1)
        orbital_pts_scatter = ax.scatter(
            orbital_pts[:, 0],
            orbital_pts[:, 1],
            c=colors["light red"],
            s=25,
            alpha=0.25,
        )
        ax.plot(
            orbital_pts[1:3, 0],
            orbital_pts[1:3, 1],
            c=colors["brick"],
            alpha=0.25,
        )
        ax.plot(
            orbital_pts[3:5, 0],
            orbital_pts[3:5, 1],
            c=colors["sage"],
            alpha=0.25,
        )


def animate(frame):
    step = frame * steps_per_frame
    frames_label.set_text(f"frame {step:05d}/{num_steps:05d}")
    for particle, circle, path in zip(particles, circles, paths):
        path.set_xdata(particle.pos[:step, 0])
        path.set_ydata(particle.pos[:step, 1])
        circle.center = particle.pos[step]
    return paths + circles + [frames_label]


def ecc_vec(massive_obj, particle):
    r_vec = particle.pos[current_step] - massive_obj.pos[current_step]
    v_vec = particle.vel[current_step]
    mu = G * massive_obj.mass
    h_vec = np.cross(r_vec, v_vec)
    return np.cross(v_vec, h_vec) / mu - normalize(r_vec)


def get_conic_coeffs(pts):
    x = pts[:, 0]
    y = pts[:, 1]
    M = np.vstack([x**2, x * y, y**2, x, y]).T
    return np.linalg.lstsq(M, np.ones(5), rcond=-1)[0]


if __name__ == "__main__":
    star = Particle(id=0, mass=1.0e7, rad=8.0, color=colors["orange"])
    planet = Particle(
        id=1,
        pos_0=100 * X_,
        vel_0=np.sqrt(G * star.mass / 100) * Y_
        + np.append(np.random.normal(0, 100, 2), 0),
        mass=1.0e-5,
        rad=3.0,
        color=colors["blue"],
    )
    particles = [star, planet]
    orbital_pts = planet.get_orbital_points(star)
    print(f"orbital eccentricity = {planet.e}")
    A, B, C, D, E = get_conic_coeffs(orbital_pts[:-1])

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

    setup_graphics(
        minmax(planet.pos[:, 0]),
        minmax(planet.pos[:, 1]),
        orbital_pts=orbital_pts,
    )
    num_frames = num_steps // 5
    steps_per_frame = num_steps // num_frames
    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        frames=num_frames,
        interval=0,
        blit=True,
    )
    plt.show()
