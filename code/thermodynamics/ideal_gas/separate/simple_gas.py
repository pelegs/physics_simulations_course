import matplotlib.pyplot as plt
import numpy as np
from ideal_gas import Container, Particle, Simulation
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from tqdm import tqdm

# Container
L = 500
dL = 50
container = Container(np.array([L, L, L]))

# Particles
N = 30
x = np.linspace(dL, L - dL, N)
y = np.linspace(dL, L - dL, N)
xv, yv = np.meshgrid(x, y)
coordinates = np.column_stack((xv.ravel(), yv.ravel()))
hlid = N**2 // 2
particle_list: list[Particle] = [
    Particle(
        id=id,
        container=container,
        pos=np.append(xy, L / 2),
        rad=2,
        color="#00AAFF",
    )
    for id, xy in enumerate(coordinates)
]
# particle_list[hlid].color = "#FF0000"
particle_list[hlid].vel = np.array([500, 300, 0.0])
for particle in particle_list[::4]:
    particle.mass = 10.0
    particle.rad = 7.5
    particle.color = "#FF0000"

# Simulation
max_t = 50.0
dt = 0.025
simulation = Simulation(container, particle_list, dt, max_t)
simulation.run()

# Graphics
# fig, ax = plt.subplots()
# ax.set_xlim(0, L)
# ax.set_ylim(0, L)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_aspect("equal")
# frames_label = ax.annotate(
#     f"frame: 0/{simulation.num_steps:04d}", xy=(10, L - 10)
# )
# circles = [
#     Circle(
#         pos[:2],
#         particle.rad,
#         facecolor=particle.color,
#         edgecolor="black",
#         lw=1,
#     )
#     for particle, pos in zip(particle_list, simulation.pos_matrix[0])
# ]
# for circle in tqdm(circles, desc="Drawing first frame"):
#     ax.add_patch(circle)
#
#
# # Create animation
# def update_sphere_animation(frame):
#     for pos, circle in zip(simulation.pos_matrix[frame], circles):
#         circle.set_center(pos[:2])
#     frames_label.set_text(f"frame: {frame:04d}/{simulation.num_steps:04d}")
#     return circles + [frames_label]
#
#
# animation = FuncAnimation(
#     fig=fig,
#     func=update_sphere_animation,
#     frames=simulation.num_steps,
#     interval=1,
# )
#
# save_pbar = lambda _i, _n: progress_bar.update(1)
# plt.show()
# writervideo = FFMpegWriter(fps=30)
# with tqdm(total=simulation.num_steps, desc="Saving video") as progress_bar:
#     animation.save(
#         "figures/simple_gas_2.mp4",
#         writer=writervideo,
#         dpi=150,
#         progress_callback=save_pbar,
#     )

# Average kintetic energy over time
Ek_all = 0.5 * np.linalg.norm(simulation.vel_matrix, axis=2) ** 2
Ek_mean = np.mean(Ek_all, axis=1)
Ek_std = np.std(Ek_all, axis=1)
plt.rcParams["text.usetex"] = True

fig = plt.figure(figsize=(10, 9), layout="constrained")
gs = GridSpec(2, 1, figure=fig)
ax_mean = fig.add_subplot(gs[0])
ax_std = fig.add_subplot(gs[1])
fig.suptitle("System equilibration (ideal gas)", fontsize=35)

# Mean
ax_mean.set_xlim(0, max_t)
ax_mean.set_title("Standard error, kinetic energy over time", fontsize=25)
ax_mean.set_xlabel("Time", fontsize=20)
ax_mean.set_ylabel(r"$\langle E_{k}\rangle$", rotation=0, fontsize=20)
ax_mean.plot(simulation.time_series, Ek_mean, c="blue")

# STD
ax_std.set_xlim(0, max_t)
ax_std.set_title("Standard error, kinetic energy over time", fontsize=25)
ax_std.set_xlabel("Time", fontsize=20)
ax_std.set_ylabel(r"$\sigma_{E_{k}}$", rotation=0, fontsize=20)
ax_std.plot(simulation.time_series, Ek_std, c="red")

plt.show()

# # Speeds histogram over time
# num_bins = 25
# histograms = np.zeros((simulation.num_steps, num_bins))
# bin_edges = np.arange(0, num_bins)
# for step, time in enumerate(simulation.time_series):
#     histograms[step], bin_edges = np.histogram(
#         speeds[step], bins=num_bins, density=True
#     )
#
# fig, ax = plt.subplots()
# ax.set_xlim(0, bin_edges[-1])
# ax.set_ylim(0, 0.05)
# ax.set_title("Equilibration of particle speeds", fontsize=25)
# ax.set_xlabel("Speeds", fontsize=20)
# ax.set_ylabel("Frequency", fontsize=20)
# hist_bars = ax.bar(bin_edges[:-1], histograms[0], width=2, edgecolor="black")
#
#
# def update_histogram_animation(frame):
#     for i, rect in enumerate(hist_bars):
#         rect.set_height(histograms[frame][i])
#     return hist_bars
#
#
# animation = FuncAnimation(
#     fig=fig,
#     func=update_histogram_animation,
#     frames=simulation.num_steps,
#     interval=1,
# )
# save_pbar = lambda _i, _n: progress_bar.update(1)
# plt.show()
# writervideo = FFMpegWriter(fps=30)
# with tqdm(total=simulation.num_steps, desc="Saving video") as progress_bar:
#     animation.save(
#         "figures/equilibration_speeds_histogram.mp4",
#         writer=writervideo,
#         dpi=150,
#         progress_callback=save_pbar,
#     )
#
# plt.show()
