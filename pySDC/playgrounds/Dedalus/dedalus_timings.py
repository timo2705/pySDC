
from dedalus import public as de
import pyfftw

import numpy as np
import time


nvars = 128
nruns = 1000

##################################
# This is the Dedalus setup block

t0 = time.time()

de.logging_setup.rootlogger.setLevel('INFO')

xbasis = de.Fourier('x', nvars, interval=(0, 1), dealias=1)
ybasis = de.Fourier('y', nvars, interval=(0, 1), dealias=1)

domain = de.Domain([xbasis, ybasis], grid_dtype=np.float64, comm=None)

x = domain.grid(0, scales=1)
y = domain.grid(1, scales=1)

# pre-define Laplacian incl. buffer
ubuffer = domain.new_field()
lap_u = xbasis.Differentiate(xbasis.Differentiate(ubuffer)) + ybasis.Differentiate(ybasis.Differentiate(ubuffer))

# setup SBDF1 for the heat equation with zero RHS, will replace initial condition in each step
problem = de.IVP(domain=domain, variables=['u'])
problem.add_equation("dt(u) - dx(dx(u)) - dy(dy(u)) = 0")
solver = problem.build_solver(de.timesteppers.SBDF1)
u = solver.state['u']

t1 = time.time()

print(f'Dedalus setup time: {t1 - t0:6.4f} sec.')  # Approx. 3.5 seconds on my machine

##################################
# This is the Dedalus "simulation" block

dt = 1.0
res = 0.0
t0 = time.time()
for n in range(nruns):

    # set initial condition
    u['g'] = np.tanh((0.25 - np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)) / (np.sqrt(2) * 0.04))

    # save initial condition (note: in the original code, new fields are created all over the place..)
    u_old = domain.new_field()
    u_old['g'] = u['g']
    u_old['c'] = u['c']  # I need to set this, since otherwise the residual will be really high...

    # solve (I-dt*A)u = u_old
    solver.step(dt)

    # fill buffer to Laplacian and evaluate
    ubuffer['g'] = u['g']
    uxx = lap_u.evaluate()

    # compute residual of (I-dt*A)u = u_old
    res = max(np.amax(abs(u['g'] - dt * uxx['g'] - u_old['g'])), res)
t1 = time.time()

print(f'Dedalus residual: {res:6.4e}')  # Should be approx. 3.4E-11
print(f'Dedalus runtime: {t1 - t0:6.4f} sec.')  # Approx. 11.7 seconds on my machine

print()

##################################
# This is the Numpy/Scipy setup block

t0 = time.time()

xvalues = np.array([i * 1.0 / nvars - 1.0 / 2.0 for i in range(nvars)])

# Setup Laplacian
kx = np.zeros(nvars)
ky = np.zeros(nvars // 2 + 1)

kx[:nvars // 2 + 1] = 2 * np.pi * np.arange(0, nvars // 2 + 1)
kx[nvars // 2 + 1:] = 2 * np.pi * np.arange(nvars // 2 + 1 - nvars, 0)
ky[:] = 2 * np.pi * np.arange(0, nvars // 2 + 1)

xv, yv = np.meshgrid(kx, ky, indexing='ij')
lap_u = -xv ** 2 - yv ** 2

# Setup FFTW plans (R2C)
rfft_in = pyfftw.empty_aligned((nvars, nvars), dtype='float64')
fft_out = pyfftw.empty_aligned((nvars, nvars // 2 + 1), dtype='complex128')
ifft_in = pyfftw.empty_aligned((nvars, nvars // 2 + 1), dtype='complex128')
irfft_out = pyfftw.empty_aligned((nvars, nvars), dtype='float64')
rfft_object = pyfftw.FFTW(rfft_in, fft_out, direction='FFTW_FORWARD', axes=(0, 1))
irfft_object = pyfftw.FFTW(ifft_in, irfft_out, direction='FFTW_BACKWARD', axes=(0, 1))

t1 = time.time()

print(f'Numpy/Scipy setup time: {t1 - t0:6.4f} sec.')  # Approx. 0.2 seconds on my machine

##################################
# This is the Numpy/Scipy "simulation" block

dt = 1.0
res = 0.0
t0 = time.time()
for n in range(nruns):

    # set initial condition
    xv, yv = np.meshgrid(xvalues, xvalues, indexing='ij')
    u_old = np.tanh((0.25 - np.sqrt((xv - 0.5) ** 2 + (yv - 0.5) ** 2)) / (np.sqrt(2) * 0.04))

    # solve (I-dt*A)u = u_old
    tmp = rfft_object(u_old) / (1.0 - dt * lap_u)
    u = np.empty((nvars, nvars))
    u[:] = irfft_object(tmp)

    # compute Laplacian
    tmp = lap_u * rfft_object(u)
    uxx = np.empty((nvars, nvars))
    uxx[:] = irfft_object(tmp)

    # compute residual of (I-dt*A)u = u_old
    res = max(np.amax(abs(u - dt * uxx - u_old)), res)
t1 = time.time()

print(f'Numpy/Scipy residual: {res:6.4e}')  # Should be approx. 5.9E-11
print(f'Numpy/Scipy runtime: {t1 - t0:6.4f} sec.')  # Approx. 0.9 seconds on my machine
