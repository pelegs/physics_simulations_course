import numpy as np
from lib.Particle import Particle

if __name__ == "__main__":
    P1 = Particle(pos=np.array([1, 2, -3]), vel=np.array([0, 0, 1]))
    P2 = Particle(vel=np.array([-1, 0, -1]))
    print(P1)
    print(P2)
