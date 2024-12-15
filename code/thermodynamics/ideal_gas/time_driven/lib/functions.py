import numpy as np
from lib.constants import npdarr


def normalize(v: npdarr) -> npdarr:
    if not np.any(v):
        raise ValueError("Can't normalize zero vector")
    return v / np.linalg.norm(v)


def dist_sqr(vec1: npdarr, vec2: npdarr) -> np.float64:
    return np.dot(vec1 - vec2, vec1 - vec2)


def distance(vec1: npdarr, vec2: npdarr) -> np.float64:
    return np.linalg.norm(vec1 - vec2)


if __name__ == "__main__":
    v: npdarr = np.array([3, -4])
    print(f"{v} -> {normalize(v)}")
