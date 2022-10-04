import time
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg as cg_cpu
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import cg as cg_gpu
import matplotlib.pyplot as plt

name = 'masterwork_timo/pickle/Dot.pickle'
# Ns = np.logspace(1, 9, 9, dtype=int)
# Ns = np.logspace(1, 8, 8, dtype=int)
Ns = np.logspace(1, 7, 7, dtype=int)
times_cpu_16 = np.zeros_like(Ns, dtype=float)
times_gpu_16 = np.zeros_like(Ns, dtype=float)
times_cpu_32 = np.zeros_like(Ns, dtype=float)
times_gpu_32 = np.zeros_like(Ns, dtype=float)
times_cpu_64 = np.zeros_like(Ns, dtype=float)
times_gpu_64 = np.zeros_like(Ns, dtype=float)
dtype = 'float32'
dtype_cpu = np.dtype(dtype)
dtype_gpu = cp.dtype(dtype)


def get_sets(N, order=2):
    if order == 2:
        stencil = [1, -2, 1]
        zero_pos = 2
    elif order == 4:
        stencil = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]
        zero_pos = 3
    elif order == 6:
        stencil = [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]
        zero_pos = 4
    elif order == 8:
        stencil = [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560]
        zero_pos = 5
    dstencil = np.concatenate((stencil, np.delete(stencil, zero_pos - 1)))
    offsets = np.concatenate(([N - i - 1 for i in reversed(range(zero_pos - 1))],
                              [i - zero_pos + 1 for i in range(zero_pos - 1, len(stencil))]))
    doffsets = np.concatenate((offsets, np.delete(offsets, zero_pos - 1) - N))
    return dstencil, doffsets


def __get_A_CPU(N, nu=1., order=2):
    """
    Helper function to assemble FD matrix A in sparse format

    Args:
        N (float): number of dofs
        nu (float): diffusion coefficient
    Returns:
        scipy.sparse.csc_matrix: matrix A in CSC format
    """
    dx = 1. / N
    dstencil, doffsets = get_sets(N, order)

    A = sp.diags(dstencil, doffsets, shape=(N, N), format='csc', dtype=dtype_cpu)
    A *= nu / (dx ** 2)

    return A


def __get_A_GPU(N, nu=1., order=2):
    """
    Helper function to assemble FD matrix A in sparse format

    Args:
        N (float): number of dofs
        nu (float): diffusion coefficient

    Returns:
        scipy.sparse.csc_matrix: matrix A in CSC format
    """
    dx = 1. / N
    dstencil, doffsets = get_sets(N, order)
    A = csp.diags(dstencil, doffsets, shape=(N, N), format='csc', dtype=dtype_gpu)
    A *= nu / (dx ** 2)

    return A

dtype = 'float16'
dtype_cpu = np.dtype(dtype)
dtype_gpu = cp.dtype(dtype)
for i, N in enumerate(Ns):
    # print(N)
    # A = 5 * sp.eye(N, format='csr', dtype=dtype_cpu)
    A = __get_A_CPU(N)
    # b = np.asarray(np.ones(N), dtype=dtype_cpu)
    start = time.perf_counter()
    # res = cg_cpu(A, b, maxiter=99)[0]
    res = A.dot(A)
    ende = time.perf_counter()
    times_cpu_32[i] = ende - start
    # A = 5 * csp.eye(N, format='csr', dtype=dtype_gpu)
    A = __get_A_GPU(N)
    # b = cp.asarray(cp.ones(N), dtype=dtype_gpu)
    start = time.perf_counter()
    # res = cg_gpu(A, b, maxiter=99)[0]
    res = A.dot(A)
    ende = time.perf_counter()
    times_gpu_32[i] = ende - start

dtype = 'float32'
dtype_cpu = np.dtype(dtype)
dtype_gpu = cp.dtype(dtype)
for i, N in enumerate(Ns):
    # print(N)
    # A = 5 * sp.eye(N, format='csr', dtype=dtype_cpu)
    A = __get_A_CPU(N)
    # b = np.asarray(np.ones(N), dtype=dtype_cpu)
    start = time.perf_counter()
    # res = cg_cpu(A, b, maxiter=99)[0]
    res = A.dot(A)
    ende = time.perf_counter()
    times_cpu_32[i] = ende - start
    # A = 5 * csp.eye(N, format='csr', dtype=dtype_gpu)
    A = __get_A_GPU(N)
    # b = cp.asarray(cp.ones(N), dtype=dtype_gpu)
    start = time.perf_counter()
    # res = cg_gpu(A, b, maxiter=99)[0]
    res = A.dot(A)
    ende = time.perf_counter()
    times_gpu_32[i] = ende - start

dtype = 'float64'
dtype_cpu = np.dtype(dtype)
dtype_gpu = cp.dtype(dtype)
for i, N in enumerate(Ns):
    print(N)
    # A = 5 * sp.eye(N, format='csr', dtype=dtype_cpu)
    A = __get_A_CPU(N)
    # b = np.asarray(np.ones(N), dtype=dtype_cpu)
    start = time.perf_counter()
    # res = cg_cpu(A, b, maxiter=99)[0]
    res = A.dot(A)
    ende = time.perf_counter()
    times_cpu_64[i] = ende - start
    # A = 5 * csp.eye(N, format='csr', dtype=dtype_gpu)
    A = __get_A_GPU(N)
    # b = cp.asarray(cp.ones(N), dtype=dtype_gpu)
    start = time.perf_counter()
    # res = cg_gpu(A, b, maxiter=99)[0]
    res = A.dot(A)
    ende = time.perf_counter()
    times_gpu_64[i] = ende - start

# write down stats to .pickle file
data = {
    'Ns': Ns,
    'times-cpu-16': times_cpu_16,
    'times-gpu-16': times_gpu_16,
    'times-cpu-32': times_cpu_32,
    'times-gpu-32': times_gpu_32,
    'times-cpu-64': times_cpu_64,
    'times-gpu-64': times_gpu_64
}
"""
plt.plot(Ns, times_cpu_32, label="float32 CPU")
plt.plot(Ns, times_gpu_32, label="float32 GPU")
plt.plot(Ns, times_cpu_64, label="float64 CPU")
plt.plot(Ns, times_gpu_64, label="float64 GPU")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Freiheitsgrade')
plt.ylabel('Zeit in s')
plt.legend()
plt.show()
print(times_cpu_32)
print(times_gpu_32)
print(times_cpu_64)
print(times_gpu_64)
"""
with open(name, 'wb') as f:
    pickle.dump(data, f)
