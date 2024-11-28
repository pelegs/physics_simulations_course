from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import quads_local
from matplotlib.patches import Rectangle


def get_bbs(tree):
    bbs = list()
    walk_node(tree._root, bbs)
    return bbs


def walk_node(node, bbs):
    bb = node.bounding_box
    bbs.append(np.array([[bb.min_x, bb.min_y], [bb.max_x, bb.max_y]]))
    for child in [node.ur, node.ul, node.ll, node.lr]:
        if child:
            walk_node(child, bbs)


def setup_fig():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", "box")
    scatter = ax.scatter(pts[:, 0], pts[:, 1], s=10, c="r")
    fig.canvas.mpl_connect("button_press_event", onclick)
    return fig, ax, scatter


def redraw_boxes():
    for patch in ax.patches:
        patch.remove()
    for bb in get_bbs(tree):
        sides = bb[1] - bb[0]
        rect = Rectangle(
            bb[0],
            sides[0],
            sides[1],
            linewidth=0.5,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(rect)


def onclick(event):
    global pts, boxes
    if event.button == 1:
        x, y = event.xdata, event.ydata
        pt = np.array([x, y])
        tree.insert(tuple(pt))
        pts = np.vstack((pts, pt))
    scatter.set_offsets(pts)
    redraw_boxes()
    # inform matplotlib of the new data
    plt.draw()


if __name__ == "__main__":
    capacity = int(argv[1])

    tree = quads_local.QuadTree(
        (5, 5),
        10,
        10,
        capacity=1,
    )

    pts = np.empty((1, 2))

    fig, ax, scatter = setup_fig()
    plt.show()
    plt.draw()
