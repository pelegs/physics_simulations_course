from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

data = np.load(argv[1])
L1, L2 = data["Ls"]
m1, m2 = data["Ms"]
time_series = data["time_series"]
t_max = time_series[-1]
th1 = data["th1"]
th2 = data["th2"]
bob1x, bob1y = data["bob1_pos"]
bob2x, bob2y = data["bob2_pos"]

# Some colors
xred = "#bd4242"
xblue = "#4268bd"
xgreen = "#52B256"
xpurple = "#7F52b2"
xorange = "#fd9337"
xgrey = "550000"
xred_soft = "#bd7171"
xblue_soft = "#798ebd"

# General
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
fig = plt.figure(figsize=(10, 9), layout="constrained")
gs = GridSpec(2, 2, figure=fig)
ax_vis = fig.add_subplot(gs[0, 0])
ax_th1_vs_th2 = fig.add_subplot(gs[0, 1])
ax_time = fig.add_subplot(gs[1, 0])
ax_heatmap = fig.add_subplot(gs[1, 1])
fig.suptitle("Double pendulum", fontsize=25)

# Visual
ax_vis.set_title("Visual view", fontsize=20)
ax_vis.get_xaxis().set_ticks([])
ax_vis.get_yaxis().set_ticks([])
ax_vis.set_xlim(-1.25 * (L1 + L2), 1.25 * (L1 + L2))
ax_vis.set_ylim(-1.25 * (L1 + L2), 1.25 * (L1 + L2))
ax_vis.plot(
    bob1x,
    bob1y,
    "-o",
    color=xred_soft,
)
ax_vis.plot(
    bob2x,
    bob2y,
    "-o",
    color=xblue_soft,
)

# th1 vs th2
ax_th1_vs_th2.set_title(r"$\theta_{1}$ vs. $\theta_{2}$", fontsize=20)
ax_th1_vs_th2.set_xlabel(r"$\theta_{1}$\ [rad]", fontsize=15)
ax_th1_vs_th2.set_ylabel(r"$\theta_{2}$\ [rad]", fontsize=15)
ax_th1_vs_th2.set_xlim(np.min(th1) - 0.5, np.max(th1) + 0.5)
ax_th1_vs_th2.set_ylim(np.min(th2) - 0.5, np.max(th2) + 0.5)
ax_th1_vs_th2.plot(th1, th2, xpurple)

# Time plot
ax_time.set_title("Time plot", fontsize=20)
ax_time.set_xlabel(r"$t$\ [s]", fontsize=15)
ax_time.set_ylabel(r"$\theta_{1,2}$", fontsize=15)
ax_time.set_xlim(0, t_max)
ax_time.set_ylim(np.min(th1 + th2) - 0.5, np.max(th1 + th2) + 0.5)
ax_time.plot(time_series, th1, xred)
ax_time.plot(time_series, th2, xblue)

# Heat map
ax_heatmap.set_title(r"Heat map of $\vec{r}_{2}$", fontsize=20)
ax_heatmap.set_xlabel(r"$x$", fontsize=15)
ax_heatmap.set_ylabel(r"$y$", fontsize=15)
H, _, _ = np.histogram2d(bob2x, bob2y, bins=100)
ax_heatmap.pcolor(H, cmap=plt.cm.Spectral)

# Save
plt.savefig("double_bobs.png")
