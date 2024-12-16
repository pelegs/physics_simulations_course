import matplotlib.pyplot as plt
from lib.constants import Axes
from lib.Simulation import Simulation
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle
from tqdm import tqdm


class Visualization2D:
    def __init__(self, simulation: Simulation) -> None:
        self.simulation: Simulation = simulation
        self.setup_axes()
        self.setup_labels()
        self.setup_particles()

    def setup_axes(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.simulation.sides[Axes.X])
        self.ax.set_ylim(0, self.simulation.sides[Axes.Y])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect("equal")

    def setup_labels(self):
        self.frames_label = self.ax.annotate(
            f"frame: 0/{self.simulation.num_steps:04d}",
            xy=(10, self.simulation.sides[Axes.Y] - 10),
        )

    def setup_particles(self):
        self.circles = [
            Circle(
                pos[:2],
                particle.rad,
                facecolor=particle.color,
                edgecolor="black",
                lw=1,
            )
            for particle, pos in zip(
                self.simulation.particle_list, self.simulation.pos_matrix[0]
            )
        ]
        for circle in self.circles:
            self.ax.add_patch(circle)

    def update_animation(self, frame):
        for pos, circle in zip(
            self.simulation.pos_matrix[frame],
            self.circles,
        ):
            circle.set_center(pos[:2])
        self.frames_label.set_text(
            f"frame: {frame:04d}/{self.simulation.num_steps:04d}"
        )
        return [self.frames_label]

    def create_animation(self):
        self.animation = FuncAnimation(
            fig=self.fig,
            func=self.update_animation,
            frames=self.simulation.num_steps,
            interval=1,
        )

    def save_animation(self, filename, dpi=150):
        save_pbar = lambda _i, _n: progress_bar.update(1)  # noqa
        writervideo = FFMpegWriter(fps=30)
        with tqdm(
            total=self.simulation.num_steps, desc="Saving video"
        ) as progress_bar:
            self.animation.save(
                filename,
                writer=writervideo,
                dpi=dpi,
                progress_callback=save_pbar,
            )
