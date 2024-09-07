from random import choice
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# Types (for hints)
npdarr = npt.NDArray[np.float64]

# Useful constants
X: int = 0
Y: int = 1
ZERO_VEC: npdarr = np.zeros(2)
X_DIR: npdarr = np.array([1, 0])
Y_DIR: npdarr = np.array([0, 1])
LL: int = 0
UR: int = 1

# Useful variables
colors = [
    "red",
    "green",
    "blue",
    "purple",
    "orange",
    "black",
    "grey",
    "pink",
    "cyan",
]

# Simulation constants
# dt: float = 0.01


# Helper functions?
def normalize(v: npdarr) -> npdarr:
    n: np.float64 = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Can't normalize zero vector")
    return v / n


def dist_sqr(v1: npdarr, v2: npdarr) -> float:
    return np.dot(v1 - v2, v1 - v2)


class Container:
    """
    A container holding all the particles in the simulation.
    It has a width and a height, and it is assumed that its bottom-left corner
    is at (0,0).

    Attributes:
        width: Width of the container.
        height: Height of the container.
    """

    def __init__(self, width: float = 1000.0, height: float = 1000.0) -> None:
        self.width: float = width
        self.height: float = height

    def __repr__(self) -> str:
        return f"width: {self.width}, height: {self.height}"


class Particle:
    """
    An ellastic, perfectly spherical particle.

    Attributes:
        id: Particle's unique identification number.
        container: A reference to the container in which the particle exists.
        pos: Position of the particle in (x,y) format (numpt ndarr, double).
        vel: Velocity of the particle in (x,y) format (numpt ndarr, double).
        rad: Radius of the particle.
        mass: Mass of the particle.
        bbox: Bounding box of the particle. Represented as a 2x2 ndarray where the first row
            is the coordinates of the lower left corner of the bbox, and the second row is the
            coordinates of the upper right corner of the bbox.
        overlaps: List of all references to all particles which have overlapping bboxes to
            this particle.
    """

    def __init__(
        self,
        id: int = -1,
        container: Container = Container(),
        pos: npdarr = ZERO_VEC,
        vel: npdarr = ZERO_VEC,
        rad: float = 1.0,
        mass: float = 1.0,
        color: str = "blue",
    ) -> None:
        # Setting argument variables
        self.id: int = id
        self.container: Container = container
        self.pos: npdarr = pos
        self.vel: npdarr = vel
        self.rad: float = rad
        self.mass: float = mass
        self.color = color

        # Setting non-argument variables
        self.bbox: npdarr = np.zeros((2, 2))
        self.set_bbox()
        self.overlaps: set[Particle] = set()

    def __repr__(self) -> str:
        return (
            f"id: {self.id}, position: {self.pos}, velocity: {self.vel}, "
            f"radius: {self.rad}, mass: {self.mass}, color: {self.color}, "
            f"bbox: {self.bbox}"
        )

    def set_bbox(self):
        self.bbox[LL] = self.pos - self.rad
        self.bbox[UR] = self.pos + self.rad

    def bounce_wall(self, direction: int) -> None:
        self.vel[direction] *= -1.0

    def resolve_wall_collisions(self) -> None:
        if (self.bbox[LL, X] < 0.0) or (
            self.bbox[UR, X] > self.container.width
        ):
            self.bounce_wall(X)
        if (self.bbox[LL, Y] < 0.0) or (
            self.bbox[UR, Y] > self.container.height
        ):
            self.bounce_wall(Y)

    def move(self, dt: float) -> None:
        """
        Advances the particle by its velocity in the given time step.
        If the particle hits a wall its velocity changes accordingly.
        Lastly, we reset its bbox so it moves with the particle.
        """
        self.pos += self.vel * dt
        self.set_bbox()
        self.resolve_wall_collisions()

    def reset_overlaps(self):
        self.overlaps = set()

    def add_overlap(self, particle: Self):
        self.overlaps.add(particle)


class Simulation:
    """docstring for Simulation."""

    def __init__(
        self,
        container: Container,
        particle_list: list[Particle],
        dt: float = 0.01,
        max_t: float = 100.0,
    ) -> None:
        # Setting argument variables
        self.container: Container = container
        self.particle_list: list[Particle] = particle_list
        self.dt: float = dt
        self.max_t: float = max_t

        # Setting non-argument variables
        self.time_series: np.ndarray = np.arange(0, max_t, dt)
        self.num_steps: int = len(self.time_series)
        self.num_particles: int = len(self.particle_list)
        self.sorted_bboxes: npdarr = np.zeros((2, num_particles))
        self.pos_matrix: npdarr = np.zeros(
            (self.num_steps, self.num_particles, 2)
        )
        self.rad_list: npdarr = np.array(
            [4 * p.rad**2 for p in self.particle_list]
        )
        self.colors_list = [p.color for p in self.particle_list]

    def check_elastic_collisions(self, p_idx: int) -> None:
        particle: Particle = self.particle_list[p_idx]
        for p2 in self.particle_list[p_idx + 1 : -1]:
            if dist_sqr(particle.pos, p2.pos) <= (particle.rad + p2.rad) ** 2:
                particle.vel, p2.vel = elastic_collision(particle, p2)

    def advance_step(self, t: int) -> None:
        for p, particle in enumerate(self.particle_list):
            particle.move(self.dt)
            self.pos_matrix[t, p] = particle.pos
            self.check_elastic_collisions(p)

    def run(self):
        for t, _ in enumerate(
            tqdm(self.time_series, desc="Running simulation")
        ):
            self.advance_step(t)


def elastic_collision(p1: Particle, p2: Particle) -> npdarr:
    vels_after: npdarr = np.zeros((2, 2))

    m1: float = p1.mass
    m2: float = p2.mass
    M: float = 2 / (m1 + m2)

    x1: npdarr = p1.pos
    x2: npdarr = p2.pos
    dx: npdarr = x1 - x2
    dx_2: float = np.dot(dx, dx)

    v1: npdarr = p1.vel
    v2: npdarr = p2.vel
    dv: npdarr = v1 - v2

    K: npdarr = M * np.dot(dv, dx) / dx_2 * dx

    vels_after[0] = v1 - K * m2
    vels_after[1] = v2 + K * m1

    return vels_after


def animate(t: int = 0):
    frame_count_label.set_text(f"Frame: {t}/{simulation.num_steps}")
    scat.set_offsets(simulation.pos_matrix[t])
    return [scat, frame_count_label]


if __name__ == "__main__":
    w: float = 500.0
    h: float = 500.0
    container: Container = Container(width=w, height=h)
    num_particles: int = 15
    particles: list[Particle] = [
        Particle(
            id=id,
            pos=np.random.uniform((50, 50), (w - 50, h - 50), size=2),
            vel=np.random.uniform(-400, 400, size=2),
            rad=15,
            container=container,
            # color=choice(colors),
        )
        for id in range(num_particles)
    ]
    particles[0].color = "red"

    simulation = Simulation(container, particles, dt=0.005, max_t=10.0)

    # Main simulation run
    simulation.run()

    ##########################
    #        Graphics        #
    ##########################

    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    frame_count_label = ax.annotate(
        f"Frame: 0/{simulation.num_steps}",
        xy=(10, h - 15),
    )
    scat = ax.scatter(
        simulation.pos_matrix[0, :, X],
        simulation.pos_matrix[0, :, Y],
        s=simulation.rad_list,
        c=simulation.colors_list,
    )

    anim = FuncAnimation(
        fig, animate, frames=simulation.num_steps, interval=0, blit=True
    )
    plt.show()
