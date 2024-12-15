from copy import deepcopy

import numpy as np
from lib.AABB import SweepPruneSystem
from lib.constants import npdarr, npiarr
from lib.Object import Object
from lib.Particle import Particle

# from tqdm import tqdm


class Simulation:
    """Docstring for Simulation."""

    objects_list: list[Object] = list()
    particle_list: list[Particle] = list()

    def __init__(
        self,
        dt: float = 0.01,
        max_t: float = 100.0,
        sides: list[float] = [100, 100, 100],
    ) -> None:
        self.dt: float = dt
        self.max_t: float = max_t
        self.sides = np.array(sides)

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
        self.collision_matrix: npiarr = np.zeros(
            (self.num_steps, self.num_particles), dtype=int
        )

    def setup_system(self) -> None:
        self.setup_sweep_prune_system()
        self.setup_simulation_parameters()
        self.setup_data_matrices()

    def advance_step(self) -> None:
        for particle in self.particle_list:
            particle.move(self.dt)

    def update_data_matrices(self, time: int) -> None:
        for pidx, particle in enumerate(self.particle_list):
            self.pos_matrix[time, pidx] = particle.pos
            self.vel_matrix[time, pidx] = particle.vel

    # def run(self) -> None:
    #     for time, _ in enumerate(
    #         tqdm(self.time_series, desc="Running simulation")
    #     ):
    #         self.advance_step()
    #         self.update_data_matrices(time)
    #
    # def save_np(self, filename: str) -> None:
    #     radii_data: npdarr = np.array(
    #         [particle.rad for particle in self.particle_list]
    #     )
    #     masses_data: npdarr = np.array(
    #         [particle.mass for particle in self.particle_list]
    #     )
    #     time_data: npdarr = np.array([self.dt, self.max_t])
    #     np.savez(
    #         filename,
    #         time_data=time_data,
    #         pos=self.pos_matrix,
    #         vel=self.vel_matrix,
    #         radii=radii_data,
    #         masses=masses_data,
    #         collisions=self.collision_matrix,
    #     )


if __name__ == "__main__":
    pass
