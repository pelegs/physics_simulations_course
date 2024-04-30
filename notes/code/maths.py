import numpy as np
import numpy.typing as npt


# Axes
X_, Y_, Z_ = np.ones(3)

# NumPy types
NDArrayFloat = npt.NDArray[np.float_]


# Functions
def normalize(vec: NDArrayFloat) -> NDArrayFloat:
    L: float = np.linalg.norm(vec)
    assert L != 0., "Can't normalize the zero vector."
    return vec / L


def scale(vec: NDArrayFloat, a: float) -> NDArrayFloat:
    return normalize(vec) * a
