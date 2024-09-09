from copy import deepcopy
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
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
POINTS_WHOLE_AX: float = 5 * 0.8 * 72


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


def dist_sqr(vec1: npdarr, vec2: npdarr) -> np.float64:
    return np.dot(vec1 - vec2, vec1 - vec2)


def distance(vec1: npdarr, vec2: npdarr) -> np.float64:
    return np.linalg.norm(vec1 - vec2)


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
        self.color: str = color

        # Setting non-argument variables
        self.bbox: npdarr = np.zeros((2, 2))
        self.set_bbox()
        self.axis_overlaps: list[set[Particle]] = [set(), set()]
        self.full_overlaps: set[Particle] = set()

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

    def resolve_elastic_collisions(self):
        for p2 in self.full_overlaps:
            if distance(self.pos, p2.pos) <= self.rad + p2.rad:
                self.vel, p2.vel = elastic_collision(self, p2)

    def move(self, dt: float) -> None:
        """
        Advances the particle by its velocity in the given time step.
        If the particle hits a wall its velocity changes accordingly.
        Lastly, we reset its bbox so it moves with the particle.
        """
        self.pos += self.vel * dt
        self.set_bbox()

    def reset_overlaps(self):
        self.axis_overlaps = [set(), set()]
        self.full_overlaps = set()

    def add_axis_overlap(self, axis: int, particle: Self):
        self.axis_overlaps[axis].add(particle)

    def set_full_overlaps(self):
        self.full_overlaps = set(
            p for p in self.axis_overlaps[X] if p in self.axis_overlaps[Y]
        )


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
        self.sorted_by_bboxes: list[list[Particle]] = [
            deepcopy(self.particle_list),
            deepcopy(self.particle_list),
        ]
        self.pos_matrix: npdarr = np.zeros(
            (self.num_steps, self.num_particles, 2)
        )
        self.energy_list: npdarr = np.zeros(self.num_steps)

    def sort_particles(self):
        def order_x(particle: Particle):
            return particle.bbox[LL, X]

        def order_y(particle: Particle):
            return particle.bbox[LL, Y]

        self.sorted_by_bboxes[X] = sorted(self.particle_list, key=order_x)
        self.sorted_by_bboxes[Y] = sorted(self.particle_list, key=order_y)

    def check_axis_overlaps(self, axis: int) -> None:
        for p1_idx, p1 in enumerate(self.sorted_by_bboxes[axis]):
            for p2 in self.sorted_by_bboxes[axis][p1_idx + 1 :]:
                if p1.bbox[UR, axis] >= p2.bbox[LL, axis]:
                    p1.add_axis_overlap(axis, p2)
                else:
                    break

    def set_full_overlaps(self):
        for particle in self.particle_list:
            particle.set_full_overlaps()

    def reset_overlaps(self):
        for particle in self.particle_list:
            particle.reset_overlaps()

    def resolve_elastic_collisions(self):
        for particle in self.particle_list:
            particle.resolve_elastic_collisions()

    def resolve_wall_collisions(self):
        for particle in self.particle_list:
            particle.resolve_wall_collisions()

    def advance_step(self, t: int) -> None:
        for p, particle in enumerate(self.particle_list):
            particle.move(self.dt)
            self.pos_matrix[t, p] = particle.pos

    def calc_total_energy(self, i: int):
        self.energy_list[i] = 0.5 * np.sum(
            [
                particle.mass * np.dot(particle.vel, particle.vel)
                for particle in self.particle_list
            ]
        )

    def run(self):
        for i, _ in enumerate(
            tqdm(self.time_series, desc="Running simulation")
        ):
            self.reset_overlaps()
            self.sort_particles()
            self.check_axis_overlaps(axis=X)
            self.check_axis_overlaps(axis=Y)
            self.set_full_overlaps()
            self.resolve_elastic_collisions()
            self.resolve_wall_collisions()
            self.advance_step(i)
            self.calc_total_energy(i)


def elastic_collision(p1: Particle, p2: Particle) -> npdarr:
    m1: float = p1.mass
    m2: float = p2.mass

    x1: npdarr = p1.pos
    x2: npdarr = p2.pos
    n: npdarr = normalize(x1 - x2)

    v1: npdarr = p1.vel
    v2: npdarr = p2.vel

    K: npdarr = 2 / (m1 + m2) * np.dot(v1 - v2, n) * n

    vels_after: npdarr = np.zeros((2, 2))
    vels_after[0] = v1 - K * m2
    vels_after[1] = v2 + K * m1

    return vels_after


def init_plot():
    for patch in patches:
        ax.add_patch(patch)
    return patches


def animate(frame: int = 0):
    frame_count_label.set_text(f"Frame: {frame:5d}")
    total_energy_label.set_text(
        f"Energy: {simulation.energy_list[frame]:0.3f}"
    )
    for p in range(num_particles):
        patches[p].center = simulation.pos_matrix[frame, p]
    return patches + [frame_count_label, total_energy_label]


if __name__ == "__main__":
    w: float = 1000.0
    h: float = 1000.0
    container: Container = Container(width=w, height=h)
    num_particles: int = 100
    particles: list[Particle] = [
        Particle(
            id=id,
            pos=np.random.uniform((50, 50), (w - 50, h - 50), size=2),
            vel=np.random.uniform(-400, 400, size=2),
            # pos=np.array([w + 0.5, h + 0.5]) / 2.0,
            # vel=np.array([100, 100]),
            rad=5.0,
            container=container,
            # color=choice(colors),
        )
        for id in range(num_particles)
    ]
    particles[0].color = "red"
    patches = [
        Circle(particle.pos.tolist(), particle.rad, fc=particle.color)
        for particle in particles
    ]

    simulation = Simulation(container, particles, dt=0.005, max_t=5.0)

    # Main simulation run
    simulation.run()

    ##########################
    #        Graphics        #
    ##########################

    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.grid()
    frame_count_label = ax.annotate(
        f"Frame: {0:5d}",
        xy=(10, h - 20),
    )
    total_energy_label = ax.annotate(
        f"Energy: {0:0.3f}",
        xy=(10, h - 40),
    )

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init_plot,
        frames=simulation.num_steps,
        interval=10,
        blit=False,
    )
    plt.show()

    print(simulation.pos_matrix.shape)
