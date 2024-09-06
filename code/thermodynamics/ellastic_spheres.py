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
    ) -> None:
        self.id: int = id
        self.container: Container = container
        self.pos: npdarr = pos
        self.vel: npdarr = vel
        self.rad: float = rad
        self.mass: float = mass
        self.bbox: npdarr = np.zeros((2, 2))
        self.set_bbox()
        self.overlaps: set[Particle] = set()

    def __repr__(self) -> str:
        return (
            f"id: {self.id}, position: {self.pos}, velocity: {self.vel}, "
            f"radius: {self.rad}, mass: {self.mass}, bbox: {self.bbox}"
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
    frame_count_label.set_text(f"Frame: {t}/{num_steps}")
    scat.set_offsets(pos_matrix[t])
    return [scat, frame_count_label]


if __name__ == "__main__":
    w: float = 500.0
    h: float = 500.0
    container: Container = Container(width=w, height=h)
    num_particles: int = 10
    particles: list[Particle] = [
        Particle(
            id=id,
            pos=np.random.uniform((50, 50), (w - 50, h - 50), size=2),
            vel=np.random.uniform(-400, 400, size=2),
            rad=5,
            container=container,
        )
        for id in range(num_particles)
    ]

    dt: float = 0.01
    max_t: float = 100.0
    time_series = np.arange(0.0, max_t, dt)
    num_steps = time_series.shape[0]

    pos_matrix: npdarr = np.zeros((num_steps, num_particles, 2))
    rad_list: npdarr = np.array([p.rad**2 for p in particles])

    for t, time in enumerate(tqdm(time_series)):
        for p, particle in enumerate(particles):
            particle.move(dt)
            pos_matrix[t, p] = particle.pos
            for p2 in particles[p + 1 : -1]:
                if (
                    dist_sqr(particle.pos, p2.pos)
                    <= (particle.rad + p2.rad) ** 2
                ):
                    particle.vel, p2.vel = elastic_collision(particle, p2)

    ##########################
    #        Graphics        #
    ##########################

    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    frame_count_label = ax.annotate(
        f"Frame: 0/{num_steps}",
        xy=(w / 2, h - 10),
    )
    scat = ax.scatter(
        pos_matrix[0, :, X], pos_matrix[0, :, Y], s=rad_list, c="#0000ff"
    )

    anim = FuncAnimation(
        fig, animate, frames=num_steps, interval=20, blit=True
    )
    plt.show()
