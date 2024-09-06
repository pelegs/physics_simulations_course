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

    def __repr__(self) -> str:
        return (
            f"id: {self.id}, position: {self.pos}, velocity: {self.vel}, "
            f"radius: {self.rad}, mass: {self.mass}, bbox: {self.bbox}"
        )

    def set_bbox(self):
        self.bbox[X] = self.pos - self.rad
        self.bbox[Y] = self.pos + self.rad

    def bounce_wall(self, direction: int) -> None:
        self.vel[direction] *= -1.0

    def resolve_wall_collisions(self) -> None:
        if (self.bbox[LL, X] <= 0.0) or (
            self.bbox[UR, X] >= self.container.width
        ):
            self.bounce_wall(X)
        if (self.bbox[LL, Y] <= 0.0) or (
            self.bbox[UR, Y] >= self.container.height
        ):
            self.bounce_wall(Y)

    def move(self, dt: float) -> None:
        """
        Advances the particle by its velocity in the given time step.
        If the particle hits a wall its velocity changes accordingly.
        Lastly, we reset its bbox so it moves with the particle.
        """
        self.pos += self.vel * dt
        self.resolve_wall_collisions()
        self.set_bbox()


def animate(t: int):
    scat.set_offsets(pos_matrix[t])
    return [scat]


if __name__ == "__main__":
    container: Container = Container()
    num_particles: int = 10
    particles: list[Particle] = [
        Particle(
            pos=np.random.uniform(0, 1000, size=2),
            vel=np.random.uniform(-1000, 1000, size=2),
            rad=np.random.uniform(0.1, 10),
            container=container,
        )
        for _ in range(num_particles)
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

    ##########################
    #        Graphics        #
    ##########################

    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    scat = ax.scatter(
        pos_matrix[0, :, X], pos_matrix[0, :, Y], s=rad_list, c="#0000ff"
    )

    anim = FuncAnimation(
        fig, animate, frames=num_steps, interval=20, blit=True
    )
    plt.show()
