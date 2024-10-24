import argparse
from subprocess import run

import manim as mn
import numpy as np


class Matrices:
    def __init__(self, matrices_str):
        self.parse(matrices_str)

    def scale_x(self, s):
        return np.array([[s, 0], [0, 1]])

    def scale_y(self, s):
        return np.array([[1, 0], [0, s]])

    def scale_xy(self, sx, sy):
        return np.array([[sx, 0], [0, sy]])

    def rotate(self, t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s], [s, c]])

    def shear_x(self, k):
        return np.array([[1, k], [0, 1]])

    def shear_y(self, k):
        return np.array([[1, 0], [k, 1]])

    def shear_xy(self, kx, ky):
        return np.array([[1, kx], [ky, 1]])

    def reflect(self, lx, ly):
        return (lx**2 + ly**2) ** -2 * np.array(
            [[lx**2 - ly**2, 2 * lx * ly], [2 * lx * ly, ly**2 - lx**2]]
        )

    def project(self, t):
        lx, ly = np.dot(self.rotate(t), np.array([1, 0]))
        return np.array([[lx, lx], [ly, ly]])

    def matrix(self, ax, ay, bx, by):
        return np.array([[ax, bx], [ay, by]])

    def parse_single(self, mat_params):
        mat_type = getattr(self, mat_params[0])
        mat_args = [float(x) for x in mat_params[1:]]
        return mat_type(*mat_args)

    def parse(self, mlist):
        self.matrix_list = [
            self.parse_single(a.split()) for a in mlist.split(",")
        ]


class MScence(mn.Scene):
    def construct(self):
        grid_original = mn.NumberPlane(
            background_line_style={
                "stroke_color": mn.BLUE,
                "stroke_width": 3,
                "stroke_opacity": 0.4,
            }
        )
        grid_transformed = mn.NumberPlane(
            background_line_style={
                "stroke_color": mn.RED,
                "stroke_width": 4,
                "stroke_opacity": 0.9,
            }
        )
        tapir = mn.SVGMobject(args.picture)
        xhat = mn.Vector(direction=[1, 0, 0], color=mn.RED)
        # xhat_label = mn.Tex(r"$\hat{x}$", color=mn.RED)
        # xhat_label.shift([1.2, 0, 0])
        yhat = mn.Vector(direction=[0, 1, 0], color=mn.GREEN)
        # det_square = mn.Square(
        #     side_length=1.0,
        #     color=mn.ORANGE,
        #     fill_color=mn.ORANGE,
        #     fill_opacity=0.5,
        # ).shift([0.5, 0.5, 0])
        self.add(grid_original)
        for matrix in matrix_list:
            self.play(
                mn.ApplyMatrix(matrix, grid_transformed),
                mn.ApplyMatrix(matrix, tapir),
                mn.ApplyMatrix(matrix, xhat),
                mn.ApplyMatrix(matrix, yhat),
                # mn.ApplyMatrix(matrix, det_square),
                run_time=1.5,
            )


def get_args():
    parser = argparse.ArgumentParser(
        prog="LTs_demo.py",
        description="Visualizes linear transformations",
        epilog="End of help string.",
    )
    parser.add_argument(
        "-m",
        "--matrices",
        type=str,
        default="rotate 1.5708",
        help="List of matrices with arguments, separated by commas",
    )
    parser.add_argument(
        "-p",
        "--picture",
        type=str,
        default="svg/tapir.svg",
        help="path to SVG picture to show on grid",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    matrix_list = Matrices(args.matrices).matrix_list

    with mn.tempconfig(
        {
            "quality": "medium_quality",
            "preview": False,
        }
    ):
        scene = MScence()
        scene.render()
        output_file = mn.config["output_file"]

    run([f"mplayer {output_file} -idle -fixed-vo"], shell=True)
