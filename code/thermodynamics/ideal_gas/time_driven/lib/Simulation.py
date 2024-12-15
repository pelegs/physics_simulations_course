from copy import deepcopy

import numpy as np
from lib.AABB import AABB, SweepPruneSystem
from lib.constants import AXES, BLB, URF, X, Y, Z, npdarr, npiarr
from lib.Container import Container
from lib.functions import distance
from lib.Particle import Particle, elastic_collision
from tqdm import tqdm


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

        # !!!!!! TO BE IMPLEMENTED !!!!!!
        # Sweep and prune system
        # !!!!!! TO BE IMPLEMENTED !!!!!!
        self.sweep_prune_system: SweepPruneSystem

    def __repr__(self) -> str:
        return (
            f"Container: {self.container}, particles: {self.particle_list}, "
            f"dt: {self.dt}, max time: {self.max_t}"
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
