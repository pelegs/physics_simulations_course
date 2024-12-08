from copy import deepcopy

import numpy as np
from tqdm import tqdm

from .constants import AXES, BLB, URF, X, Y, Z, npdarr, npiarr
from .Container import Container
from .functions import distance
from .Particle import Particle, elastic_collision


class Simulation:
    """Docstring for Simulation."""

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


if __name__ == "__main__":
    print("Testing Simulation class")
