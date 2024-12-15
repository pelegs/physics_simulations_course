import numpy as np
from lib.constants import ZERO_VEC, npdarr
from lib.functions import normalize
from lib.Object import MovingObject


class Particle(MovingObject):
    """An ellastic, perfectly spherical particle.

    Attributes:
        rad: Radius of the particle.
        mass: Mass of the particle.
    """

    def __init__(
        self,
        pos: npdarr = np.copy(ZERO_VEC),
        vel: npdarr = np.copy(ZERO_VEC),
        rad: float = 1.0,
        mass: float = 1.0,
        color: str = "#FF0000",
        opacity: float = 1.0,
    ) -> None:
        bbox_pts: npdarr = np.array([pos - rad, pos + rad])
        super().__init__(pos, vel, bbox_pts, color, opacity)

        self.rad: float = rad
        self.mass: float = mass

    def __repr__(self) -> str:
        return (
            f"id: {self.id}, position: {self.pos}, velocity: {self.vel}, "
            f"radius: {self.rad}, mass: {self.mass}, color: {self.color}, "
            f"bbox: {self.bbox}"
        )


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
