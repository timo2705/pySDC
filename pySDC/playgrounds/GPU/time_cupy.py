import cupy as cp
import numpy as np
import cupyx.scipy.sparse as csp
import matplotlib.pyplot as plt


class cupy_time:
    def __init__(self):
        self.start = cp.cuda.Event()
        self.ende = cp.cuda.Event()
        self.Ns = np.logspace(1, 7, 7, dtype=int)

    def measure(self, N, dtype_gpu=cp.dtype('float32')):
        A = (5 * csp.eye(N, format='csr')).astype(dtype_gpu)
        self.start.record()
        A.dot(A)
        self.ende.record()
        self.ende.synchronize()
        return cp.cuda.get_elapsed_time(self.start, self.ende)

    def measure_all(self):
        times_32 = np.empty_like(self.Ns, dtype=float)
        times_64 = np.empty_like(self.Ns, dtype=float)
        for i, N in enumerate(self.Ns):
            times_32[i] = self.measure(N, dtype_gpu=cp.dtype('float32'))
            times_64[i] = self.measure(N, dtype_gpu=cp.dtype('float64'))
        return times_32, times_64


timing = cupy_time()
print(timing.measure(10))
t32, t64 = timing.measure_all()
plt.plot(timing.Ns, t32, label="float32 GPU")
plt.plot(timing.Ns, t64, label="float64 GPU")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Freiheitsgrade')
plt.ylabel('Zeit in s')
plt.legend()
plt.show()

print(t32)
print(t64)
