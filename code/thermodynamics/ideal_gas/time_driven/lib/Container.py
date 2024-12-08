import numpy as np

from .constants import npdarr


class Container:
    """A container holding all the particles in the simulation. It has 3
    dimensions (in the x-, y- and z-directions). It is assumed that one of its
    corners (Bottom-Left-Back, BLB) is at (0,0,0) and the other (Upper-Right-
    Front, URF) is at (Lx, Ly, Lz).

    Attributes:
        dimensions: the dimensions of the container in the x, y and z
        directions.
    """

    def __init__(self, dimensions: npdarr = 100.0 * np.ones(3)) -> None:
        self.dimensions: npdarr = dimensions

    def __repr__(self) -> str:
        return f"{self.dimensions}"
