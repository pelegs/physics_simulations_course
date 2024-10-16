import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import XKCD_COLORS

# Time-related constants, variables and parameters
num_steps = 10
dt = 0.01
max_t = num_steps * dt
time_seris = np.arange(0, max_t, dt)
step = 0

# Other constants, variables and parameters
G = 1.0
ZERO_VEC = np.zeros(3)
X_, Y_, Z_ = np.identity(3)
INIT_MAT = np.zeros((num_steps, 3))

# Colors
Colors = {name[5:]: HTML_val for name, HTML_val in XKCD_COLORS.items()}


# Math and physics functions
def normalize(vec):
    if L := np.linalg.norm(vec) == 0:
        raise ValueError("Can't normalize the zero vector")
    return vec / L


def scale(vec, norm):
    return norm * normalize(vec)


def look_at(v1, v2):
    """
    Returns a unit vector in the direction
    from v1 to v2.
    """
    return normalize(v2 - v1)


def sqr_dist(p1, p2):
    return np.dot(p2 - p1, p2 - p1)


def gravity(p1, p2):
    r_vec = p2.pos - p1.pos
    r_dir = normalize(r_vec)
    r2 = np.dot(r_vec, r_vec)
    return (G * p1.mass * p2.mass / r2) * r_dir


# Objects with mass
class Object:
    def __init__(
        self,
        pos_init,
        vel_init=np.copy(ZERO_VEC),
        mass=1.0,
        rad=1.0,
        color=Colors["light red"],
        id=-1,
    ):
        self.pos = np.copy(INIT_MAT)
        self.pos[0] = pos_init

        self.vel = np.copy(INIT_MAT)
        self.vel[0] = vel_init

        self.acc = np.copy(INIT_MAT)
        self.force = np.copy(INIT_MAT)

        self.mass = mass
        self.mass_inv = 1.0 / mass
        self.rad = rad

        self.color = color
        self.id = id

        self = 0

    def add_force(self, force):
        self.force[step] = self.force[step] + force

    def verlet_1(self):
        self.pos[step] = (
            self.pos[step - 1]
            + self.vel[step - 1] * dt
            + 0.5 * self.acc[step - 1] * dt**2
        )

    def verlet_2(self):
        self.acc[step] = self.force[step] * self.mass_inv
        self.vel[step] = (
            self.vel[step - 1]
            + 0.5 * (self.acc[step - 1] + self.acc[step]) * dt
        )


if __name__ == "__main__":
    p = Object(pos_init=np.array([1, -2, 0.4]))
