import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

num_frames = 900
xs = np.linspace(0, 10, num_frames)
ys = xs * np.cos(xs)

fig, ax = plt.subplots()
ax.set_title("Some basic animation", size=25)
ax.set_xlabel("x", size=20)
ax.set_ylabel("y", size=20)
line = ax.plot(xs, ys)[0]


def animate(frame):
    line.set_xdata(xs[:frame])
    line.set_ydata(ys[:frame])


ani = FuncAnimation(fig=fig, func=animate, frames=num_frames, interval=0)
plt.show()
