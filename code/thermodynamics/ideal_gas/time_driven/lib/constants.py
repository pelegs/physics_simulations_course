import numpy as np
import numpy.typing as npt

# Types (for hints)
npdarr = npt.NDArray[np.float64]
npiarr = npt.NDArray[np.int8]

# Useful constants
X: int = 0
Y: int = 1
Z: int = 2
UNION: int = 3
AXES: list[int] = [X, Y, Z]
ZERO_VEC: npdarr = np.zeros(3)
X_DIR, Y_DIR, Z_DIR = np.identity(3)
BLB: int = 0
URF: int = 1

if __name__ == "__main__":
    print("test:", AXES, X_DIR, Y_DIR, Z_DIR)
