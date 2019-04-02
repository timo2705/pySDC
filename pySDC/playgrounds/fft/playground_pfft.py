from mpi4py import MPI
import numpy as np
import pfft
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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



