from enum import IntEnum

import numpy as np
import numpy.typing as npt

# Types (for hints)
npdarr = npt.NDArray[np.float64]
npiarr = npt.NDArray[np.int8]


class Axes(IntEnum):
    X = 0
    Y = 1
    Z = 2


UNION = 3


class Pts(IntEnum):
    LLF = 0
    RHB = 1


ZERO_VEC: npdarr = np.zeros(3)
X_DIR, Y_DIR, Z_DIR = np.identity(3)

if __name__ == "__main__":
    print("test:", Axes, X_DIR, Y_DIR, Z_DIR)
