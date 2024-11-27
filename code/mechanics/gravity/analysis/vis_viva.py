import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

parser = argparse.ArgumentParser(
    description="Plotting the Vis-Viva equation for a set of data points"
)
parser.add_argument(
    "-i", "--input-file", help="Data file (with folder)", required=True
)
parser.add_argument(
    "-f",
    "--num_frames",
    type=int,
    default=500,
    help="If > 0, show animation",
)
args = parser.parse_args()

data = np.load(args.input_file)
distances = np.linalg.norm(data["pos"], axis=1)
speeds = np.linalg.norm(data["vel"], axis=1)
M = 1.0e7  # should be taken from data, atm I'm saving only the planet's data
a = np.mean(
    M * distances / (2 * M - distances * speeds**2)
)  # semi-major axis as avg
vis_viva_speeds = np.sqrt(M * (2 / distances - 1 / a))


def setup_figure():
    fig, ax = plt.subplots()
    ax.set_title(f"Vis-Viva plot for file {args.input_file}", fontsize=15)
    ax.set_xlabel("r", fontsize=15)
    ax.set_ylabel("v", rotation=0, fontsize=15)

    min_x = min(np.min(distances), np.min(speeds)) - 50
    max_x = max(np.max(distances), np.max(speeds)) + 50
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_x, max_x)
    ax.set_aspect("equal", "box")

    frames_label = plt.text(
        0.05,
        0.95,
        f"frame {0:05d}/{num_steps:05d}",
        fontsize=10,
        transform=ax.transAxes,
    )

    r0 = distances[0]
    v0 = speeds[0]
    vvv0 = vis_viva_speeds[0]
    r0v0 = (r0, v0)

    circ_rad = (max_x - min_x) / 100

    vvv_line = ax.plot(r0, vvv0, linewidth=5, c="#6BBFFF")[0]
    speeds_line = ax.plot(r0, v0, linewidth=1, c="#000000")[0]
    state = Circle(xy=r0v0, radius=circ_rad, color="#FF0000")
    ax.add_patch(state)
    return fig, ax, frames_label, vvv_line, speeds_line, state


def anim_update(frame):
    step = frame * spf
    frames_label.set_text(f"frame {step:05d}/{num_steps:05d}")
    vvv_line.set_data(distances[:step], vis_viva_speeds[:step])
    speeds_line.set_data(distances[:step], speeds[:step])
    state.center = (distances[step], speeds[step])
    return [vvv_line, frames_label, speeds_line, state]


if __name__ == "__main__":
    num_steps = speeds.shape[0]
    num_frames = args.num_frames
    spf = num_steps // num_frames
    fig, ax, frames_label, vvv_line, speeds_line, state = setup_figure()

    if num_frames > 0:
        ani = animation.FuncAnimation(
            fig=fig,
            func=anim_update,
            frames=num_frames,
            interval=0,
            blit=True,
        )
        plt.show()
    # else:
    #     ax.plot(
    #         vis_viva_speeds,
    #         distances,
    #     )
    #     ax.plot(speeds, distances, linewidth=1, c="black")
    #     plt.show()
