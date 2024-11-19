from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import quads_local
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from tqdm import tqdm


def insert_pts(tree, pts):
    tree.reset()
    for pt in pts:
        tree.insert(tuple(pt))
    return tree


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
    boxes = list()
    return fig, ax, boxes


def draw(step):
    scatter = ax.scatter(
        pts[step, :, 0], pts[step, :, 1], s=6, c=np.arange(num_pts)
    )

    for patch in list(ax.patches):
        patch.set_visible(False)
        patch.remove()

    boxes = list()
    for bb in bounding_boxes[step]:
        sides = bb[1] - bb[0]
        rect = Rectangle(
            bb[0],
            sides[0],
            sides[1],
            linewidth=0.5,
            edgecolor="black",
            facecolor="none",
        )
        boxes.append(rect)
        ax.add_patch(rect)
    return [scatter] + boxes


if __name__ == "__main__":
    num_pts = int(argv[1])
    num_steps = int(argv[2])
    capacity = int(argv[3])

    tree = quads_local.QuadTree(
        (5, 5),
        10,
        10,
        capacity=1,
    )

    bounding_boxes = list()
    bounding_boxes.append(get_bbs(tree))

    dt = 0.1

    pts = np.zeros((num_steps, num_pts, 2))
    pts[0] = np.random.uniform(0, 10, (num_pts, 2))
    dxs = np.random.normal(size=(num_steps - 1, num_pts, 2))

    for i, dx in enumerate(tqdm(dxs)):
        pts[i + 1] = pts[i] + dx * dt
        pts[i + 1] = np.mod(pts[i + 1], 10)
        tree = insert_pts(tree, pts[i + 1])
        bounding_boxes.append(get_bbs(tree))

    fig, ax, boxes = setup_fig()
    ani = FuncAnimation(
        fig=fig, func=draw, frames=num_steps, interval=1, blit=True
    )
    plt.show()
    # ani.save("quads_test.mp4")
