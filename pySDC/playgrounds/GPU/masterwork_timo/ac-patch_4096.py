import matplotlib.pyplot as plt
import cupy as cp
import time

from pySDC.core.Collocation import CollBase as Collocation
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.problem_classes.AllenCahn_FFT_gpu import allencahn_imex as ac_gpu
from pySDC.helpers.gpu_hook import hook_gpu

start = time.perf_counter()
# initialize level parameters
level_params = dict()
level_params['restol'] = 1E-08
level_params['dt'] = 1E-03
level_params['nsweeps'] = 1

# initialize sweeper parameters
sweeper_params = dict()
sweeper_params['collocation_class'] = Collocation
sweeper_params['node_type'] = 'LEGENDRE'
sweeper_params['quad_type'] = 'RADAU-RIGHT'
sweeper_params['num_nodes'] = 3
sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
sweeper_params['initial_guess'] = 'zero'

# initialize problem parameters
problem_params = dict()
# problem_params['nvars'] = (2048, 2048)  # für 128x128
# problem_params['nvars'] = (1024, 1024)  # für 64x64
# problem_params['nvars'] = (512, 512)  # für 32x32
# problem_params['L'] = 16.0

problem_params['nvars'] = (4096, 4096)
problem_params['L'] = 128.0
problem_params['eps'] = 0.04
problem_params['dw'] = -10  # -23.6
problem_params['radius'] = 0.08
problem_params['name'] = "name"
problem_params['init_type'] = 'circle_rand'
problem_params['spectral'] = False

# initialize step parameters
step_params = dict()
step_params['maxiter'] = 50

# initialize controller parameters
controller_params = dict()
controller_params['logger_level'] = 30
controller_params['hook_class'] = hook_gpu

# fill description dictionary for easy step instantiation
description = dict()
description['problem_class'] = ac_gpu
description['problem_params'] = problem_params  # pass problem parameters
description['sweeper_class'] = imex_1st_order
description['sweeper_params'] = sweeper_params  # pass sweeper parameters
description['level_params'] = level_params  # pass level parameters
description['step_params'] = step_params  # pass step parameters

# instantiate controller
controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

# set time parameters
t0 = 0.0
schritte = 2000
Tend = t0 + 1 * level_params['dt']

# get initial values on finest level
P = controller.MS[0].levels[0].prob
uinit = P.u_exact(t0)
plt.imshow(cp.asnumpy(uinit), extent=[-0.5, 0.5, -0.5, 0.5])
plt.title("Time = {time:.3f}".format(time=t0))
plt.colorbar()
plt.savefig("pngs/uend_4096_32x32_0000.png")
plt.clf()
for i in range(1, schritte+1):
    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    if i % 1 == 0:
        print("saved a png at Tend = {time:.3f}".format(time=Tend))
        plt.imshow(cp.asnumpy(uend), extent=[-0.5, 0.5, -0.5, 0.5])
        plt.title("Time = {time:.3f}".format(time=Tend))
        plt.colorbar()
        plt.savefig("pngs/uend_4096_32x32_{index:04d}.png".format(index=i))
        plt.clf()
    t0 = Tend
    Tend = t0 + 1 * level_params['dt']
    uinit = uend
plt.imshow(cp.asnumpy(uend), extent=[-0.5, 0.5, -0.5, 0.5])
plt.title("Time = {time:.3f}".format(time=Tend))
plt.colorbar()
plt.savefig("pngs/uend_4096_32x32_{index:04d}.png".format(index=i))
plt.clf()
end = time.perf_counter()
print('done in {time:5.3f} seconds'.format(time=(end-start)))
