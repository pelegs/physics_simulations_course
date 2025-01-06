from enum import Enum

import numpy as np
from lib.AABB import SweepPruneSystem
from lib.constants import ZERO_VEC, Axes, npdarr, npiarr
from lib.functions import distance
from lib.Object import Object
from lib.Particle import Particle, elastic_collision, untangle_spheres
from tqdm import tqdm


class Boundary(Enum):
    EMPTY = 0
    WALL = 1
    PERIODIC = 2


class Simulation:
    """Docstring for Simulation."""

    objects_list: list[Object] = list()
    particle_list: list[Particle] = list()
    AABBs_overlaps: list[npiarr] = list()
    step: int = 0

    def __init__(
        self,
        dt: float = 0.01,
        max_t: float = 100.0,
        sides: list[float] = [100.0] * 3,
        boundaries: list[Boundary] = [Boundary.WALL] * 3,
    ) -> None:
        self.dt: float = dt
        self.max_t: float = max_t
        self.sides = np.array(sides)
        self.boundaries = boundaries
        self.center = self.sides / 2.0

    def __repr__(self) -> str:
        return (
            f"Objects: {self.objects_list}, "
            f"Particles: {self.particle_list}, "
            f"dt: {self.dt}, max time: {self.max_t}, "
            f"num steps: {self.num_steps}"
        )

    def add_object(self, object: Object) -> None:
        """
        Note: exception handling should be done better.
        """
        try:
            self.objects_list.append(object)
            if isinstance(object, Particle):
                self.particle_list.append(object)
        except Exception as e:
            print(e)

    def add_objects(self, objects: list[Object]) -> None:
        for object in objects:
            self.add_object(object)

    def remove_object(self, object: Object):
        """
        Note: exception handling should be implemented
        """
        if object in self.objects_list:
            self.objects_list.remove(object)
        if object in self.particle_list:
            self.particle_list.remove(object)

    def remove_objects(self, objects: list[Object]):
        for object in objects:
            self.remove_object(object)

    def put_particles_on_grid(self, Ns: npiarr, dL: npdarr = ZERO_VEC) -> None:
        lspaces = [
            np.linspace(dL[axis], self.sides[axis] - dL[axis], Ns[axis])
            for axis in Axes
        ]
        xv, yv, zv = np.meshgrid(*lspaces)
        coordinates = np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))
        for particle, coordinate in zip(self.particle_list, coordinates):
            particle.set_pos(coordinate)

    def setup_sweep_prune_system(self) -> None:
        self.sweep_prune_system: SweepPruneSystem = SweepPruneSystem(
            [object.bbox for object in self.objects_list]
        )

    def setup_simulation_parameters(self) -> None:
        self.time_series: np.ndarray = np.arange(0, self.max_t, self.dt)
        self.num_steps: int = self.time_series.shape[0]
        self.num_particles: int = len(self.particle_list)

    def setup_data_matrices(self) -> None:
        self.pos_matrix: npdarr = np.zeros(
            (self.num_steps, self.num_particles, 3)
        )
        self.vel_matrix: npdarr = np.zeros(
            (self.num_steps, self.num_particles, 3)
        )
        # self.collision_matrix: npiarr = np.zeros(
        #     (self.num_steps, self.num_particles), dtype=int
        # )

    def setup_system(self) -> None:
        self.setup_sweep_prune_system()
        self.setup_simulation_parameters()
        self.setup_data_matrices()

    def advance_particles(self) -> None:
        for particle in self.particle_list:
            particle.move(self.dt)

    def resolve_boundry_bounce(self, axis: Axes, particle: Particle) -> None:
        if (particle.pos[axis] - particle.rad <= 0) or (
            particle.pos[axis] + particle.rad >= self.sides[axis]
        ):
            particle.vel[axis] = -1 * particle.vel[axis]

    def resolve_periodic_condition(
        self, axis: Axes, particle: Particle
    ) -> None:
        if not (0 <= particle.pos[axis] <= self.sides[axis]):
            new_pos: npdarr = particle.pos
            new_pos[axis] = new_pos[axis] % self.sides[axis]
            particle.set_pos(new_pos)

    def resolve_boundries(self) -> None:
        for particle in self.particle_list:
            for axis in Axes:
                match self.boundaries[axis]:
                    case Boundary.WALL:
                        self.resolve_boundry_bounce(axis, particle)
                    case Boundary.PERIODIC:
                        self.resolve_periodic_condition(axis, particle)

    def resolve_collisions(self) -> None:
        for id_1, id_2 in self.AABBs_overlaps[-1]:
            obj_1 = self.sweep_prune_system.AABB_list[id_1].obj
            obj_2 = self.sweep_prune_system.AABB_list[id_2].obj
            if isinstance(obj_1, Particle) and isinstance(obj_2, Particle):
                d = distance(obj_1.pos, obj_2.pos)
                if d <= (obj_1.rad + obj_2.rad):
                    obj_1.pos, obj_2.pos = untangle_spheres(obj_1, obj_2)
                    obj_1.vel, obj_2.vel = elastic_collision(obj_1, obj_2)
                    # self.collision_matrix[time, i] = 1
                    # self.collision_matrix[time, j] = 1

    def update_data_matrices(self) -> None:
        for p_idx, particle in enumerate(self.particle_list):
            self.pos_matrix[self.step, p_idx] = particle.pos
            self.vel_matrix[self.step, p_idx] = particle.vel

    def run(self) -> None:
        for step, _ in enumerate(
            tqdm(self.time_series, desc="Running simulation")
        ):
            self.step = step
            self.advance_particles()
            self.resolve_boundries()
            self.AABBs_overlaps.append(self.sweep_prune_system.calc_overlaps())
            self.resolve_collisions()
            self.update_data_matrices()

    def save_np(self, filename: str) -> None:
        masses_data: npdarr = np.array(
            [particle.mass for particle in self.particle_list]
        )
        radii_data: npdarr = np.array(
            [particle.rad for particle in self.particle_list]
        )
        time_data: npdarr = np.array([self.dt, self.max_t])
        np.savez(
            filename,
            time_data=time_data,
            pos=self.pos_matrix,
            vel=self.vel_matrix,
            radii=radii_data,
            masses=masses_data,
            # collisions=self.collision_matrix,
        )


if __name__ == "__main__":
    pass
