from pySDC.implementations.datatype_classes.mesh import mesh
import numpy as np
import time


class wrapper(np.ndarray):

    def __init__(self, *args):
        super(wrapper, self).__init__(*args)
        self[:] = val



n = 128*128

t0 = time.time()
a = mesh(n, val=0)

for i in range(10000):
    b = mesh(n, val=1)
    a += b
t1 = time.time()
print(t1-t0)

t0 = time.time()
a = np.zeros(n)
b = np.zeros(n)
b[:] = 1
for i in range(10000):
    a += b
t1 = time.time()
print(t1-t0)

t0 = time.time()
a = wrapper(shape=n, val=0)
for i in range(10000):
    b = wrapper(n, val=1)
    a += b
t1 = time.time()
print(t1-t0)