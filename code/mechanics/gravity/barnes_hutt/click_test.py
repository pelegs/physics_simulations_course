import matplotlib.pyplot as plt
import numpy as np

pos = np.zeros((1, 2))


def onclick(event):
    global pos
    if event.button == 1:
        x, y = event.xdata, event.ydata
        pt = np.array([x, y])
        pos = np.vstack((pos, pt))
    # clear frame
    # plt.clf()
    scatter.set_offsets(pos)
    # inform matplotlib of the new data
    plt.draw()  # redraw


fig, ax = plt.subplots()
scatter = ax.scatter(pos[:, 0], pos[:, 1])
fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()
plt.draw()
