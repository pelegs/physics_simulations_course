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
            f"position: {self.pos}, velocity: {self.vel}, "
            f"radius: {self.rad}, mass: {self.mass}, color: {self.color}, "
            f"bbox: {self.bbox}"
        )


def untangle_spheres(p1: Particle, p2: Particle) -> npdarr:
    DX: npdarr = p1.pos - p2.pos
    DV: npdarr = p1.vel - p2.vel
    R1: float = p1.rad
    R2: float = p2.rad

    A: float = np.dot(DV, DV)
    B: float = 2 * np.dot(DX, DV)
    C: float = np.dot(DX, DX) - (R1 + R2) ** 2

    DESC = B**2 - 4 * A * C
    if DESC >= 0.0:
        t0: float = (-B - np.sqrt(DESC)) / (2 * A)
        t1: float = (-B + np.sqrt(DESC)) / (2 * A)
        t_back: float = min(t0, t1)
        P1 = p1.pos + t_back * p1.vel
        P2 = p2.pos + t_back * p2.vel
        return np.array([P1, P2])
    else:
        return np.array([p1.pos, p2.pos])


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
