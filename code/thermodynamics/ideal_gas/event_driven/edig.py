import numpy as np

###########################
#        Constants        #
###########################

X_, Y_, Z_ = np.identity(3)
ZERO_VEC = np.zeros(3)

#########################
#        Classes        #
#########################


class Event:
    """Docstring for Event."""

    def __init__(self, type, p1):
        self.type = type
        self.p1 = p1
