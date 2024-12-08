import numpy as np
from lib.functions import distance

if __name__ == "__main__":
    v1 = np.array([1, 2, -3])
    v2 = np.array([-2, 0, 5])
    print(distance(v1, v2))
