import numpy as np

from .constants import AXES, BLB, URF, ZERO_VEC, npdarr
from .Container import Container
from .functions import normalize


class Particle:
    """An ellastic, perfectly spherical particle.

    Attributes:
        id: Particle's unique identification number.
        container: A reference to the container in which the particle exists.
        pos: Position of the particle in (x,y,z) format (numpt ndarr, double).
        vel: Velocity of the particle in (x,y,z) format (numpt ndarr, double).
        rad: Radius of the particle.
        mass: Mass of the particle.
        bbox: Bounding box of the particle. Represented as a 2x2 ndarray
              where the first row is the coordinates of the lower left corner
              of the bbox, and the second row is the coordinates of the upper
              right corner of the bbox.
    """

    def __init__(
        self,
        id: int = -1,
        container: Container = Container(),
        pos: npdarr = ZERO_VEC,
        vel: npdarr = ZERO_VEC,
        rad: float = 1.0,
        mass: float = 1.0,
        color: str = "#aaaaaa",
        opacity: float = 1.0,
    ) -> None:
        # Setting argument variables
        self.id: int = id
        self.container: Container = container
        self.pos: npdarr = pos
        self.vel: npdarr = vel
        self.rad: float = rad
        self.mass: float = mass
        self.color: str = color
        self.opacity: float = opacity

        # Setting non-argument variables
        self.bbox: npdarr = np.zeros((2, 3))
        self.set_bbox()

    def __repr__(self) -> str:
        return (
            f"id: {self.id}, position: {self.pos}, velocity: {self.vel}, "
            f"radius: {self.rad}, mass: {self.mass}, color: {self.color}, "
            f"bbox: {self.bbox}"
        )

    def set_bbox(self):
        self.bbox[BLB] = self.pos - self.rad
        self.bbox[URF] = self.pos + self.rad

    def bounce_wall(self, direction: int) -> None:
        self.vel[direction] *= -1.0

    def resolve_wall_collisions(self) -> None:
        for axis in AXES:
            if (self.bbox[BLB, axis] < 0.0) or (
                self.bbox[URF, axis] > self.container.dimensions[axis]
            ):
                self.bounce_wall(axis)

    def move(self, dt: float) -> None:
        """Advances the particle by its velocity in the given time step.

        If the particle hits a wall its velocity changes accordingly.
        Lastly, we reset its bbox so it moves with the particle.
        """
        self.pos += self.vel * dt
        self.set_bbox()


def elastic_collision(p1: Particle, p2: Particle) -> npdarr:
    m1: float = p1.mass
    m2: float = p2.mass

    n: npdarr = normalize(p1.pos - p2.pos)

    v1: npdarr = p1.vel
    v2: npdarr = p2.vel

    K: npdarr = 2 / (m1 + m2) * np.dot(v1 - v2, n) * n

    vels_after: npdarr = np.zeros((2, 3))
    vels_after[0] = v1 - K * m2
    vels_after[1] = v2 + K * m1

    return vels_after


if __name__ == "__main__":
    print("test")
