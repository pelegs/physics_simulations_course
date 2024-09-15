import functools
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.stats import maxwell
from tqdm import tqdm

# Types (for hints)
npdarr = npt.NDArray[np.float64]
npiarr = npt.NDArray[np.int8]

# Useful constants
X: int = 0
Y: int = 1
ZERO_VEC: npdarr = np.zeros(2)
X_DIR: npdarr = np.array([1, 0])
Y_DIR: npdarr = np.array([0, 1])
LL: int = 0
UR: int = 1


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
        overlaps: List of all references to all particles which have overlapping bboxes to
            this particle.
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
        self.pos_save: npdarr = np.zeros(2)
        self.free_paths_list: list[np.float64] = list()

        # Setting non-argument variables
        self.bbox: npdarr = np.zeros((2, 2))
        self.set_bbox()

    def __repr__(self) -> str:
        return (
            f"id: {self.id}, position: {self.pos}, velocity: {self.vel}, "
            f"radius: {self.rad}, mass: {self.mass}, color: {self.color}, "
            f"bbox: {self.bbox}"
        )

    def set_bbox(self):
        self.bbox[LL] = self.pos - self.rad
        self.bbox[UR] = self.pos + self.rad

    def bounce_wall(self, direction: int) -> None:
        self.vel[direction] *= -1.0

    def resolve_wall_collisions(self) -> None:
        if (self.bbox[LL, X] < 0.0) or (
            self.bbox[UR, X] > self.container.width
        ):
            self.bounce_wall(X)
        if (self.bbox[LL, Y] < 0.0) or (
            self.bbox[UR, Y] > self.container.height
        ):
            self.bounce_wall(Y)

    def move(self, dt: float) -> None:
        """
        Advances the particle by its velocity in the given time step.
        If the particle hits a wall its velocity changes accordingly.
        Lastly, we reset its bbox so it moves with the particle.
        """
        self.pos += self.vel * dt
        self.set_bbox()

    def add_free_path(self) -> None:
        self.free_paths_list.append(distance(self.pos, self.pos_save))
        self.pos_save = np.zeros(2)


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
        ]
        self.reset_overlaps()

        # Data matrices
        self.pos_matrix: npdarr = np.zeros(
            (self.num_steps, self.num_particles, 2)
        )
        self.vel_matrix: npdarr = np.zeros(
            (self.num_steps, self.num_particles, 2)
        )

    @staticmethod
    def order_x(particle: Particle):
        return particle.bbox[LL, X]

    @staticmethod
    def order_y(particle: Particle):
        return particle.bbox[LL, Y]

    def sort_particles(self):
        self.sorted_by_bboxes[X] = sorted(self.particle_list, key=self.order_x)
        self.sorted_by_bboxes[Y] = sorted(self.particle_list, key=self.order_y)

    def check_axis_overlaps(self, axis: int) -> None:
        for p1_idx, p1 in enumerate(self.sorted_by_bboxes[axis]):
            for p2 in self.sorted_by_bboxes[axis][p1_idx + 1 :]:
                if p2.bbox[LL, axis] <= p1.bbox[UR, axis]:
                    self.axis_overlap_matrix[axis, p1.id, p2.id] = 1
                    self.axis_overlap_matrix[axis, p2.id, p1.id] = 1
                else:
                    break

    def set_full_overlaps(self):
        # overlaps.append(list())
        self.full_overlap_matrix = np.triu(
            np.logical_and(
                self.axis_overlap_matrix[X], self.axis_overlap_matrix[Y]
            )
        )
        self.overlap_ids = np.vstack(np.where(self.full_overlap_matrix)).T

    def reset_overlaps(self):
        self.axis_overlap_matrix: npiarr = np.zeros(
            (2, self.num_particles, self.num_particles), dtype=bool
        )
        self.full_overlap_matrix: npiarr = np.zeros(
            (self.num_particles, self.num_particles), dtype=bool
        )

    def resolve_elastic_collisions(self):
        for i, j in self.overlap_ids:
            p1, p2 = self.particle_list[i], self.particle_list[j]
            if distance(p1.pos, p2.pos) <= p1.rad + p2.rad:
                p1.vel, p2.vel = elastic_collision(p1, p2)
                p1.add_free_path()
                p2.add_free_path()

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

    def run(self):
        for time, _ in enumerate(
            tqdm(self.time_series, desc="Running simulation")
        ):
            self.reset_overlaps()
            self.sort_particles()
            self.check_axis_overlaps(axis=X)
            self.check_axis_overlaps(axis=Y)
            self.set_full_overlaps()
            self.resolve_elastic_collisions()
            self.resolve_wall_collisions()
            self.advance_step()
            self.update_data_matrices(time)


def elastic_collision(p1: Particle, p2: Particle) -> npdarr:
    m1: float = p1.mass
    m2: float = p2.mass

    n: npdarr = normalize(p1.pos - p2.pos)

    v1: npdarr = p1.vel
    v2: npdarr = p2.vel

    K: npdarr = 2 / (m1 + m2) * np.dot(v1 - v2, n) * n

    vels_after: npdarr = np.zeros((2, 2))
    vels_after[0] = v1 - K * m2
    vels_after[1] = v2 + K * m1

    return vels_after


# def init_spheres_animation():
#     for patch in patches:
#         ax.add_patch(patch)
#     return patches


# def animate_spheres(frame: int = 0):
#     frame_count_label.set_text(f"Frame: {frame:5d}")
#     # overlaps_text.set_text("\n".join([f"({x})" for x in overlaps[frame]]))
#     for i, patch in enumerate(patches):
#         patch.center = simulation.pos_matrix[frame, i]
#     # for i, (patch, label) in enumerate(zip(patches, labels)):
#     # patch.center = simulation.pos_matrix[frame, i]
#     # patch.set_xy(simulation.pos_matrix[frame, i] - particles[i].rad)
#     # label.set_position(patch.get_xy())
#     # return patches + labels + [frame_count_label]
#     return patches + [frame_count_label]


# def onClick(event):
#     global pause
#     pause = not pause
#     if pause:
#         anim.pause()
#     else:
#         anim.resume()


def animate_histograms(frame: int, bar_container):
    for count, rect in zip(histograms[frame], bar_container.patches):
        rect.set_height(count)
    frame_count_label.set_text(f"Frame: {frame:5d}/{simulation.num_steps:5d}")
    # mx = maxwell.pdf(hist_bins, *mb_params[frame])
    # mb_plot.set_ydata(mx)
    # print(mx)

    return bar_container.patches + [frame_count_label]  # + [mb_plot]


if __name__ == "__main__":
    w: float = 400.0
    h: float = 400.0
    container: Container = Container(width=w, height=h)
    N: int = 20
    num_particles: int = N**2
    # radius: float = w / N * 0.25
    radius: float = 5.0
    particles: list[Particle] = [
        Particle(
            id=i * N + j,
            pos=np.array([w / N * (i + 0.5), h / N * (j + 0.5)]),
            vel=normalize(np.random.uniform(-1, 1, size=2)) * 100.0,
            rad=radius,
            container=container,
        )
        for i in range(N)
        for j in range(N)
    ]

    simulation = Simulation(container, particles, dt=0.001, max_t=1.0)
    # overlaps = list()

    # Main simulation run
    simulation.run()

    ##########################
    #        Analysis        #
    ##########################

    # Histograms
    # num_bins: int = 100
    # speeds_matrix: npdarr = np.linalg.norm(simulation.vel_matrix, axis=2)
    # max_speed: float = float(np.max(speeds))
    # hist_bins = np.linspace(0, max_speed, num_bins)
    #
    # histograms: npdarr = np.zeros((simulation.num_steps, num_bins - 1))
    # mb_params: npdarr = np.zeros((simulation.num_steps, 2))
    # for frame, _ in enumerate(simulation.pos_matrix):
    #     histograms[frame], _ = np.histogram(
    #         speeds[frame], hist_bins, density=True
    #     )
    #     mb_params[frame] = maxwell.fit(speeds[frame], floc=0)
    # max_freq: float = float(np.max(histograms[900:]))

    # Mean free path
    mean_free_path_calculated: float = (
        simulation.container.height
        * simulation.container.width
        / (np.sqrt(8) * simulation.num_particles * 2 * radius)
    )
    print("Mean free path (calculated):", mean_free_path_calculated)

    free_paths_list: list[np.float64] = list()
    for particle in simulation.particle_list:
        free_paths_list += particle.free_paths_list
    free_paths_list_np = np.array(free_paths_list)
    print("Mean free path (simulated):", np.mean(free_paths_list_np))

    ##########################
    #        Graphics        #
    ##########################

    # Visual animation

    # patches = [
    #     Circle(particle.pos.tolist(), particle.rad, fc=particle.color)
    #     for particle in particles
    # ]
    #
    # fig, ax = plt.subplots()
    # ax.set_aspect("equal", "box")
    # ax.set_xlim(0, w)
    # ax.set_ylim(0, h)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # # labels = [ax.annotate(f"{p.id}", xy=p.pos.astype(int)) for p in particles]
    # # ax.grid()
    # frame_count_label = ax.annotate(
    #     f"Frame: {0:5d}",
    #     xy=(10, h - 20),
    # )
    # # overlaps_text = ax.annotate("", xy=(10, h - 80))
    #
    # fig.canvas.mpl_connect("button_press_event", onClick)
    # anim = FuncAnimation(
    #     fig,
    #     animate,
    #     init_func=init_plot,
    #     frames=simulation.num_steps,
    #     interval=0,
    #     blit=True,
    # )
    # plt.show()

    # Velocity histogram
    # fig, ax = plt.subplots()
    # ax.set_xlim(0, max_speed * 1.1)
    # ax.set_ylim(0, max_freq * 1.1)
    # ax.set_xlabel("Speeds")
    # ax.set_ylabel("Frequency")
    # ax.set_title("Speed histograms")
    # _, _, bar_container = ax.hist(
    #     speeds[0], hist_bins, lw=1, fc="#9ecae1", ec="#3182bd", alpha=0.5
    # )
    # frame_count_label = ax.annotate(
    #     f"Frame: {0:5d}/{simulation.num_steps:05d}",
    #     xy=(10.0, max_freq * 0.99),
    # )
    # # (mb_plot,) = ax.plot(
    # #     hist_bins, maxwell.pdf(hist_bins, *mb_params[0]), lw=3, c="red"
    # # )
    # anim = functools.partial(animate_histograms, bar_container=bar_container)
    # ani = FuncAnimation(
    #     fig,
    #     anim,
    #     simulation.num_steps,
    #     interval=0,
    #     repeat=False,
    #     blit=True,
    # )
    # plt.show()
