
from dedalus import public as de
import numpy as np
import matplotlib.pyplot as plt

de.logging_setup.rootlogger.setLevel('INFO')

xbasis = de.Fourier('x', 32, interval=(0,1), dealias=1)


domain = de.Domain([xbasis],grid_dtype=np.float64,mesh=[1])

f = domain.new_field()

g = domain.new_field()

x = domain.grid(0, scales=1)

f['g'] = np.sin(2*np.pi*x)

g['g'] = np.cos(2*np.pi*x)

u = domain.new_field()

fc = domain.new_field()
fc['g'] = np.copy(f['g'])

ff = domain.new_field()
ff['g'] = np.copy(f['g'])
ff.set_scales(scales=2)


# print(ff['g'].shape, fc['g'].shape)
#
# print((ff-fc).evaluate()['g'])


h = (f + g).evaluate()


f_x = de.operators.differentiate(h, x=2).evaluate()

forcing = domain.new_field()
forcing['g'] = -np.sin(np.pi * 2 * x) * (np.sin(0) - (np.pi * 2) ** 2 * np.cos(0))

dt = 0.1 / 16

u_old = domain.new_field()
u_old['g'] = np.copy(f['g'])

problem = de.LinearBoundaryValueProblem(domain=domain, variables=['u'])
problem.meta[:]['x']['dirichlet'] = True
problem.parameters['dt'] = dt
problem.parameters['u_old'] = u_old + dt*forcing
problem.add_equation("u - dt * dx(dx(u)) = u_old")


solver = problem.build_solver()
u = solver.state['u']

Tend = 1.0
nsteps = int(Tend/dt)

t = 0.0
for n in range(nsteps):
    problem.parameters['u_old'] = u_old + dt*forcing
    solver.solve()
    t += dt
    forcing['g'] = -np.sin(np.pi * 2 * x) * (np.sin(t) - (np.pi * 2) ** 2 * np.cos(t))
    u_old['g'] = np.copy(u['g'])
    # print(n)


uex = domain.new_field()
# uex['g'] = np.sin(2*np.pi*x) * np.exp(-(2*np.pi)**2 * Tend)
uex['g'] = np.sin(2*np.pi*x) * np.cos(Tend)

print(np.linalg.norm(u['g']-uex['g'], np.inf))

# plt.figure(1)
# plt.plot(x,u['g'])
# plt.plot(x,uex['g'])
#
# plt.pause(1)
#
# exit()

forcing = domain.new_field()
forcing['g'] = -np.sin(np.pi * 2 * x) * (np.sin(0) - (np.pi * 2) ** 2 * np.cos(0))

u_old['g'] = np.zeros(32)
problem = de.IVP(domain=domain, variables=['u'])
# problem.parameters['RHS'] = u_old + forcing
problem.parameters['RHS'] = 0
problem.add_equation("dt(u) - dx(dx(u)) = RHS")

ts = de.timesteppers.SBDF1
solver = problem.build_solver(ts)
u = solver.state['u']
u_old['g'] = np.copy(f['g'])

t = 0.0
for n in range(nsteps):
    u['g'] = u['g'] - dt * np.sin(np.pi * 2 * x) * (np.sin(t) - (np.pi * 2) ** 2 * np.cos(t))
    # problem.parameters['RHS'] = u_old + forcing
    solver.step(dt)
    t += dt
    # forcing['g'] = -np.sin(np.pi * 2 * x) * (np.sin(t) - (np.pi * 2) ** 2 * np.cos(t))
    # u_old['g'] = 1.0 / dt * u['g']
    # u['g'] -= de.operators.differentiate(u_old, x=2)*dt

print(np.linalg.norm(u['g']-uex['g'], np.inf))

#
plt.figure(1)
plt.plot(x,u['g'])
plt.plot(x,uex['g'])
#
plt.pause(1)
