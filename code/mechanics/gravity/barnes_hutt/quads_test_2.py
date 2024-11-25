import matplotlib.pyplot as plt
import numpy as np
import quads_local
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle


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


def on_move(event):
    if event.inaxes:
        print(
            f"data coords {event.xdata} {event.ydata},",
            f"pixel coords {event.x} {event.y}",
        )


def on_click(event):
    if event.button is MouseButton.LEFT:
        print("disconnecting callback")
        plt.disconnect(binding_id)


if __name__ == "__main__":
    tree = quads_local.QuadTree(
        (5, 5),
        10,
        10,
        capacity=1,
    )

    pts = list()

    # binding_id = plt.connect("motion_notify_event", on_move)
    # plt.connect("button_press_event", on_click)

    setup_fig()
    plt.show()
