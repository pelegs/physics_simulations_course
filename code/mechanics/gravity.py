#!/usr/bin/env python

import numpy as np

# --- Constants --- #

G = 1.0
dt = 0.001

# --- Vector functions --- #

def normalize(vec):
    norm = np.linalg.norm(vec)
    assert norm != 0, "Can't normalize zero vectors"
    return vec / norm


def scale(vec, s):
    return normalize(vec) * s


def look_at(v1, v2):
    return normalize(v2-v1)


# --- Physics functions --- #

def F_gravity(m1, m2, p1, p2):
    dr = p2 - p1
    r = np.linanlg.norm(dr)
    F = G * m1 * m2 / r**2
    return scale(dr, F)

# --- Integrators --- #

# Simple Forward-Euler
def forward_euler(pos_arr, vel_arr, mass_arr):
    # itertools? Numpy meshgrid?..
    pass


# --- Main --- #

if __name__ == "__main__":
    xs = np.array([[0,0,0], [1,0,0], [-3,2,0]])
    vs = np.array([[-1,1,0], [0,-1,0], [0,2,0]])
    ms = np.array([1, 4, 0.5]).T
    forward_euler(xs, vs, ms)
