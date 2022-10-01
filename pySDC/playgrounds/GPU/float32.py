import time
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg as cg_cpu
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import cg as cg_gpu
name = 'masterwork_timo/pickle/float.pickle'
# Ns = np.logspace(1, 14, 14, dtype=int)
Ns = np.logspace(1, 7, 7, dtype=int)
times_cpu_32 = np.zeros_like(Ns, dtype=float)
times_gpu_32 = np.zeros_like(Ns, dtype=float)
times_cpu_64 = np.zeros_like(Ns, dtype=float)
times_gpu_64 = np.zeros_like(Ns, dtype=float)
dtype = 'float32'
dtype_cpu = np.dtype(dtype)
dtype_gpu = cp.dtype(dtype)
for i, N in enumerate(Ns):
    A = 5 * sp.eye(N, format='csr', dtype=dtype_cpu)
    b = np.asarray(np.ones(N), dtype=dtype_cpu)
    start = time.perf_counter()
    res = cg_cpu(A, b, maxiter=99)[0]
    ende = time.perf_counter()
    times_cpu_32[i] = ende - start
    A = 5 * csp.eye(N, format='csr', dtype=dtype_gpu)
    b = cp.asarray(cp.ones(N), dtype=dtype_gpu)
    start = time.perf_counter()
    res = cg_gpu(A, b, maxiter=99)[0]
    ende = time.perf_counter()
    times_gpu_32[i] = ende - start

dtype = 'float64'
dtype_cpu = np.dtype(dtype)
dtype_gpu = cp.dtype(dtype)
for i, N in enumerate(Ns):
    A = 5 * sp.eye(N, format='csr', dtype=dtype_cpu)
    b = np.asarray(np.ones(N), dtype=dtype_cpu)
    start = time.perf_counter()
    res = cg_cpu(A, b, maxiter=99)[0]
    ende = time.perf_counter()
    times_cpu_64[i] = ende - start
    A = 5 * sp.eye(N, format='csr', dtype=dtype_cpu)
    b = np.asarray(np.ones(N), dtype=dtype_cpu)
    start = time.perf_counter()
    res = cg_cpu(A, b, maxiter=99)[0]
    ende = time.perf_counter()
    times_cpu_64[i] = ende - start

# write down stats to .pickle file
data = {
    'Ns': Ns,
    'times-cpu-32': times_cpu_32,
    'times-gpu-32': times_gpu_32,
    'times-cpu-64': times_cpu_64,
    'times-gpu-64': times_gpu_64
}
print(data)
# with open(name, 'wb') as f:
#     pickle.dump(data, f)
