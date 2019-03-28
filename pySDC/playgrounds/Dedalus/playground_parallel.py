from mpi4py import MPI
import matplotlib
matplotlib.use("TkAgg")

from dedalus import public as de
import numpy as np
import matplotlib.pyplot as plt


comm = MPI.COMM_WORLD

world_rank = comm.Get_rank()
world_size = comm.Get_size()
# split world communicator to create space-communicators
color = int(world_rank / world_size)
space_comm = comm.Split(color=color)
space_size = space_comm.Get_size()
space_rank = space_comm.Get_rank()

de.logging_setup.rootlogger.setLevel('INFO')

xbasis = de.Fourier('x', 4, interval=(0, 1), dealias=1)
ybasis = de.Fourier('y', 4, interval=(0, 1), dealias=1)

domain = de.Domain([xbasis, ybasis], grid_dtype=np.float64, comm=space_comm)

print(domain.global_grid_shape(), domain.local_grid_shape())

f = domain.new_field()
g = domain.new_field()

x = domain.grid(0, scales=1)
y = domain.grid(1, scales=1)


# g['g'] = np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
#
# u = domain.new_field()
#
# xbasis_c = de.Fourier('x', 16, interval=(0, 1), dealias=1)
# ybasis_c = de.Fourier('y', 16, interval=(0, 1), dealias=1)
# domain_c = de.Domain([xbasis_c, ybasis_c], grid_dtype=np.float64, comm=space_comm)
#
# f.set_scales(0.5)
# fex = domain_c.new_field()
# fex['g'] = np.copy(f['g'])
#
# f.set_scales(1)
# ff = domain.new_field()
# ff['g'] = np.copy(f['g'])
# ff.set_scales(scales=0.5)
#
# fc = domain_c.new_field()
# fc['g'] = ff['g']
#
# print(fc['g'].shape, fex['g'].shape)
# #
#
# local_norm = np.linalg.norm((fc-fex).evaluate()['g'])
#
# if space_size == 1:
#     global_norm = local_norm
# else:
#     global_norm = space_comm.allreduce(sendobj=local_norm, op=MPI.MAX)
#
# print(global_norm)
# print(domain.distributor.comm, space_comm)
# # exit()
#
# h = (f + g).evaluate()
#
#
# f_x = de.operators.differentiate(h, x=2).evaluate()

dt = 0.1 / 32
Tend = 1.0
nsteps = int(Tend/dt)


uex = domain.new_field()
uex['g'] = np.sin(2*np.pi*x) * np.sin(2*np.pi*y) * np.cos(Tend)

problem = de.IVP(domain=domain, variables=['u'])
problem.add_equation("dt(u) - dx(dx(u)) - dy(dy(u)) = 0")

ts = de.timesteppers.SBDF1
solver = problem.build_solver(ts)
u = solver.state['u']


t = 0.0
for n in range(nsteps):
    u['g'] = u['g'] - dt * np.sin(np.pi * 2 * x) * np.sin(2*np.pi*y) * (np.sin(t) - 2.0 * (np.pi * 2) ** 2 * np.cos(t))
    solver.step(dt)
    t += dt
# print(t, nsteps)

local_norm = np.amax(abs(u['g']-uex['g']))
if space_size == 1:
    global_norm = local_norm
else:
    global_norm = space_comm.allreduce(sendobj=local_norm, op=MPI.MAX)

print(local_norm, global_norm)

# print(np.linalg.norm(u['g']-uex['g'], np.inf))

# xx, yy = np.meshgrid(x, y)
#
# plt.figure(1)
# #
# plt.contourf(xx, yy, u['g'].T, 50)
# plt.colorbar()
#
# plt.figure(2)
# plt.contourf(xx, yy, uex['g'].T,50)
# plt.colorbar()
# # plt.figure(3)
# # plt.plot(u['g'][8,:])
# # plt.plot(uex['g'][8,:])
# # #
# plt.show(1)
