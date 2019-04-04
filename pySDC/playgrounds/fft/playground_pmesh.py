from mpi4py import MPI
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pfft
import time
import matplotlib.pyplot as plt
from numpy.fft import rfft2, irfft2

from pmesh.pm import ParticleMesh, RealField, ComplexField

def doublesine(i, v):
    r = [ii * (Li / ni) for ii, ni, Li in zip(i, v.Nmesh, v.BoxSize)]
    # xx, yy = np.meshgrid(r[0], r[1])
    return np.sin(2*np.pi*r[0]) * np.sin(2*np.pi*r[1])

def Lap_doublesine(i, v):
    r = [ii * (Li / ni) for ii, ni, Li in zip(i, v.Nmesh, v.BoxSize)]
    # xx, yy = np.meshgrid(r[0], r[1])
    return -2.0 * (2.0 * np.pi) ** 2 * np.sin(2*np.pi*r[0]) * np.sin(2*np.pi*r[1])

def Laplacian(k, v):
    k2 = sum(ki ** 2 for ki in k)
    # print([type(ki[0][0]) for ki in k])
    # k2[k2 == 0] = 1.0
    return -k2 * v


nvars = 128
nruns = 1000

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

t0 = time.time()
pm = ParticleMesh(BoxSize=1.0, Nmesh=[nvars] * 2, dtype='f8', plan_method='measure', comm=comm)
t1 = time.time()

print(f'PMESH setup time: {t1 - t0:6.4f} sec.')

dt = 0.121233
res = 0.0
t0 = time.time()
for n in range(nruns):

    # set initial condition
    u = pm.create(type='real')
    u = u.apply(doublesine, kind='index', out=Ellipsis)

    # save initial condition
    u_old = pm.create(type='real', value=u)

    def linear_solve(k, v):
        global dt
        k2 = sum(ki ** 2 for ki in k)
        factor = 1 + dt * k2
        return 1.0 / factor * v

    # solve (I-dt*A)u = u_old
    u = u.r2c().apply(linear_solve, out=Ellipsis).c2r(out=Ellipsis)

    # compute Laplacian
    uxx = u.r2c().apply(Laplacian, out=Ellipsis).c2r(out=Ellipsis)

    # compute residual of (I-dt*A)u = u_old
    res = max(np.amax(abs(u.preview() - dt*uxx.preview() - u_old.preview())), res)
t1 = time.time()

print(f'PMESH residual: {res:6.4e}')  # Should be approx. 5.9E-11
print(f'PMESH runtime: {t1 - t0:6.4f} sec.')  # Approx. 0.9 seconds on my machine

exit()

pmf = ParticleMesh(BoxSize=1.0, Nmesh=[nvars] * 2, comm=comm)
pmc = ParticleMesh(BoxSize=1.0, Nmesh=[nvars//2] * 2, comm=comm)

uexf = pmf.create(type='real')
uexf = uexf.apply(doublesine, kind='index')

uexc = pmc.create(type='real')
uexc = uexc.apply(doublesine, kind='index')

uc = pmc.upsample(uexf, keep_mean=True)
# uc = pmc.create(type='real')
# uexf.resample(uc)
print(uc.preview().shape, np.amax(abs(uc-uexc)))

uf = pmf.create(type='real')
uexc.resample(uf)
print(uf.preview().shape, np.amax(abs(uf-uexf)))
print()

t1 = time.time()
print(f'Time: {t1-t0}')






#
# print(type(u.preview()))

# plt.figure(1)
# #
# plt.contourf((uf-uexf).preview(), 50)
# plt.colorbar()
# plt.show()
#










exit()

procmesh = pfft.ProcMesh([size], comm=None)
partition = pfft.Partition(pfft.Type.PFFT_R2C, [8, 8], procmesh, flags=pfft.Flags.PFFT_DESTROY_INPUT | pfft.Flags.PFFT_TRANSPOSED_OUT)
# for irank  in range(size):
#     MPI.COMM_WORLD.barrier()
#     if irank != procmesh.rank:
#         continue
#     print('My rank is', procmesh.this)
#     print('local_i_start', partition.local_i_start)
#     print('local_o_start', partition.local_o_start)
#     print('i_edges', partition.i_edges)
#     print('o_edges', partition.o_edges)

buffer = pfft.LocalBuffer(partition)

plan = pfft.Plan(partition, pfft.Direction.PFFT_FORWARD, buffer)
iplan = pfft.Plan(partition, pfft.Direction.PFFT_BACKWARD, buffer, flags=pfft.Flags.PFFT_DESTROY_INPUT | pfft.Flags.PFFT_TRANSPOSED_OUT)

rinit = partition.local_i_shape
print(rank, rinit)
dx = 1.0 / partition.n[0]
dy = 1.0 / partition.n[1]
xvalues = np.array([i * dx - 1.0 / 2.0 for i in range(rank * rinit[0], (rank + 1) * rinit[0])])
yvalues = np.array([i * dy - 1.0 / 2.0 for i in range(rinit[1])])

xv, yv = np.meshgrid(xvalues, yvalues, indexing='ij')



input = buffer.view_input()
# print(rank, input.shape)

input[...] = np.sin(2*np.pi*xv) * np.sin(2*np.pi*yv)

exact = -2.0 * (2.0 * np.pi) ** 2 * np.sin(2*np.pi*xv) * np.sin(2*np.pi*yv)
print(rank, input.shape, exact.shape)
# print(rank, input)

cinit = partition.local_o_shape
print('cinit', rank, cinit)

recvdata = comm.allgather(cinit[1])

cinit_prev = sum(recvdata[:rank])

kx = np.zeros(cinit[0])
ky = np.zeros(cinit[1])

kx[:int(cinit[0] / 2) + 1] = 2 * np.pi / 1 * np.arange(0, int(cinit[0] / 2) + 1)
kx[int(cinit[0] / 2) + 1:] = 2 * np.pi / 1 * np.arange(int(cinit[0] / 2) + 1 - cinit[0], 0)
ky[:] = 2 * np.pi / 1 * np.arange(cinit_prev, cinit_prev + cinit[1])

xv, yv = np.meshgrid(kx, ky, indexing='ij')
lap = -xv ** 2 - yv ** 2

plan.execute(buffer)

output = buffer.view_output()

# denormalize the forward transform
output *= lap / np.product(partition.n)
print(rank, output.shape)

iplan.execute(buffer)
print(rank, buffer.view_output().shape)

result = buffer.view_output()
print(np.amax(abs(result.T + exact)))



