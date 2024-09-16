import json
from copy import deepcopy
from pathlib import Path
from sys import argv

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

# Types (for hints)
npdarr = npt.NDArray[np.float64]
npiarr = npt.NDArray[np.int8]

# Useful constants
X: int = 0
Y: int = 1
Z: int = 2
AXES: list[int] = [X, Y, Z]
ZERO_VEC: npdarr = np.zeros(3)
X_DIR: npdarr = np.array([1, 0, 0])
Y_DIR: npdarr = np.array([0, 1, 0])
Z_DIR: npdarr = np.array([0, 0, 1])
BLB: int = 0
URF: int = 1


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

# For pausing
pause = False


# Helper functions
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
    It has 3 dimensions (in the x-, y- and z-directions).
    It is assumed that one of its corners (Bottom-Left-Back, BLB) is at (0,0,0)
    and the other (Upper-Right-Front, URF) is at (Lx, Ly, Lz).

    Attributes:
        dimensions: the dimensions of the container in the x, y and z
        directions.
    """

    def __init__(
        self, dimensions: npdarr = np.array([100.0, 100.0, 100.0])
    ) -> None:
        self.dimensions: npdarr = dimensions

    def __repr__(self) -> str:
        return f"{self.dimensions}"


class Particle:
    """
    An ellastic, perfectly spherical particle.

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
        """
        Advances the particle by its velocity in the given time step.
        If the particle hits a wall its velocity changes accordingly.
        Lastly, we reset its bbox so it moves with the particle.
        """
        self.pos += self.vel * dt
        self.set_bbox()


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
            deepcopy(self.particle_list),
        ]
        self.reset_overlaps()

        # Data matrices
        self.pos_matrix: npdarr = np.zeros(
            (self.num_steps, self.num_particles, 3)
        )
        self.vel_matrix: npdarr = np.zeros(
            (self.num_steps, self.num_particles, 3)
        )
        self.collision_matrix: npiarr = np.zeros(
            (self.num_steps, self.num_particles), dtype=int
        )

    def __repr__(self) -> str:
        return (
            f"Container: {self.container}, particles: {self.particle_list}, "
            f"dt: {self.dt}, max time: {self.max_t}"
        )

    @staticmethod
    def order_x(particle: Particle):
        return particle.bbox[BLB, X]

    @staticmethod
    def order_y(particle: Particle):
        return particle.bbox[BLB, Y]

    @staticmethod
    def order_z(particle: Particle):
        return particle.bbox[BLB, Z]

    def sort_particles(self):
        for axis, order_func in zip(
            AXES, [self.order_x, self.order_y, self.order_z]
        ):
            self.sorted_by_bboxes[axis] = sorted(
                self.particle_list, key=order_func
            )

    def check_axis_overlaps(self, axis: int) -> None:
        for p1_idx, p1 in enumerate(self.sorted_by_bboxes[axis]):
            for p2 in self.sorted_by_bboxes[axis][p1_idx + 1 :]:
                if p2.bbox[BLB, axis] <= p1.bbox[URF, axis]:
                    self.axis_overlap_matrix[axis, p1.id, p2.id] = 1
                    self.axis_overlap_matrix[axis, p2.id, p1.id] = 1
                else:
                    break

    def set_full_overlaps(self):
        # reduce() is used because there are three overlap matrices
        self.full_overlap_matrix = np.triu(
            np.logical_and.reduce(self.axis_overlap_matrix)
        )
        self.overlap_ids = np.vstack(np.where(self.full_overlap_matrix)).T

    def reset_overlaps(self):
        self.axis_overlap_matrix: npiarr = np.zeros(
            (2, self.num_particles, self.num_particles), dtype=bool
        )
        self.full_overlap_matrix: npiarr = np.zeros(
            (self.num_particles, self.num_particles), dtype=bool
        )

    def resolve_elastic_collisions(self, time: int):
        for i, j in self.overlap_ids:
            p1, p2 = self.particle_list[i], self.particle_list[j]
            if distance(p1.pos, p2.pos) <= p1.rad + p2.rad:
                p1.vel, p2.vel = elastic_collision(p1, p2)
                self.collision_matrix[time, i] = 1
                self.collision_matrix[time, j] = 1

    def resolve_wall_collisions(self):
        for particle in self.particle_list:
            particle.resolve_wall_collisions()

    def advance_step(self) -> None:
        for particle in self.particle_list:
            particle.move(self.dt)

    def update_data_matrices(self, time: int) -> None:
        for pidx, particle in enumerate(self.particle_list):
            self.pos_matrix[time, pidx] = particle.pos
            self.vel_matrix[time, pidx] = particle.vel

    def run(self) -> None:
        for time, _ in enumerate(
            tqdm(self.time_series, desc="Running simulation")
        ):
            self.reset_overlaps()
            self.sort_particles()
            self.check_axis_overlaps(axis=X)
            self.check_axis_overlaps(axis=Y)
            self.set_full_overlaps()
            self.resolve_elastic_collisions(time)
            self.resolve_wall_collisions()
            self.advance_step()
            self.update_data_matrices(time)

    def save_to_file(self, filename: str) -> None:
        coordinates = self.pos_matrix.reshape(
            (self.num_steps, self.num_particles * 3)
        )
        other_data: npiarr = np.ones((self.num_steps, 2), dtype=int)
        other_data[:, 0] = np.arange(1, self.num_steps + 1)
        other_data[:, 1] = self.num_particles
        radii: npdarr = np.array(
            [
                [particle.rad for particle in self.particle_list]
                for _ in range(self.num_steps)
            ]
        )
        file_content = np.concatenate((other_data, radii, coordinates), axis=1)

        np.savetxt(
            filename,
            file_content,
            fmt=["%04d", "%04d"] + ["%0.5f"] * self.num_particles * 4,
        )

        with open(filename, "r") as original:
            data = original.read()
        sphere_names: str = " ".join(
            [
                f"sphere{i}_{x}"
                for i in range(1, self.num_particles + 1)
                for x in ["x", "y", "z"]
            ]
        )
        sphere_radii: str = " ".join(
            [f"sphere{i}_radius" for i in range(1, self.num_particles + 1)]
        )
        column_names: str = f"frame sphere_count {sphere_radii} {sphere_names}"
        with open(filename, "w") as modified:
            modified.write(f"{column_names}\n{data}")

    def save_np(self, filename: str) -> None:
        radii_data: npdarr = np.array(
            [particle.rad for particle in self.particle_list]
        )
        masses_data: npdarr = np.array(
            [particle.mass for particle in self.particle_list]
        )
        time_data: npdarr = np.array([self.dt, self.max_t])
        np.savez(
            filename,
            time_data=time_data,
            pos=self.pos_matrix,
            vel=self.vel_matrix,
            radii=radii_data,
            masses=masses_data,
            collisions=self.collision_matrix,
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


def load_scene(scene_path: str | Path) -> Simulation:
    with open(scene_path, "r") as json_file:
        scene = json.load(json_file)
    particles: list[Particle] = [
        Particle(
            id=particle["id"],
            pos=np.array(particle["pos"]),
            vel=np.array(particle["vel"]),
            rad=particle["rad"],
            mass=particle["mass"],
        )
        for particle in scene["particles"]
    ]
    container: Container = Container(
        np.array(scene["container"]["dimensions"])
    )
    return Simulation(
        container, particles, scene["time"]["dt"], scene["time"]["max_t"]
    )


if __name__ == "__main__":
    # Setup file names
    scene_file: Path = Path(argv[1])
    scene_filename: str = scene_file.stem
    output_path: str = f"outputs/{scene_filename}.npz"

    simulation = load_scene(scene_file)

    # Lx: float = 300.0
    # Ly: float = 300.0
    # Lz: float = 300.0
    # container: Container = Container(np.array([Lx, Ly, Lz]))
    #
    # radius: float = 10.0
    # dw: float = radius + 1.0
    #
    # Nx, Ny, Nz = 7, 7, 7
    # x_p = np.linspace(0 + dw, Lx - dw, Nx)
    # y_p = np.linspace(0 + dw, Ly - dw, Ny)
    # z_p = np.linspace(0 + dw, Lz - dw, Nz)
    # coordinates = np.vstack(np.meshgrid(x_p, y_p, z_p)).reshape(3, -1).T
    #
    # particles = [
    #     Particle(
    #         id=id,
    #         pos=np.array([x, y, z]),
    #         vel=np.random.uniform(-10, 10, size=3),
    #         rad=radius,
    #         container=container,
    #     )
    #     for id, (x, y, z) in enumerate(coordinates)
    # ]
    #
    # simulation = Simulation(container, particles, dt=0.1, max_t=500.0)

    # Main simulation run
    simulation.run()

    # Save positions to external file
    simulation.save_np(output_path)
