from pySDC.implementations.datatype_classes.mesh import mesh
import numpy as np
import time
import copy as cp


class mydata():

    def __init__(self, init):
        # self.values = init.values
        if isinstance(init, type(self)):
            self.values = init.values
        elif isinstance(init, tuple) or isinstance(init, int):
            self.values = np.zeros(init)
        else:
            raise NotImplementedError()

    # def __init__(self, init):
    #     # print(init)
    #     self.values = np.zeros(init)
    #
    # @classmethod
    # def fill(cls, values):
    #     c = cls(len(values))
    #     c.values = values

    def __add__(self, other):
        # s = mydata(len(self.values))
        s = mydata(self)
        # s += other.values
        s.values = self.values + other.values
        return s



nruns = int(1E6)

z = mydata(160)
z.values[:] = 1
y = mydata(160)
y.values[:] = 1
t0 = time.time()
for i in range(nruns):
    z += y
t1 = time.time()
print(z.values[-1])
print(t1-t0)
# print(y.values)
