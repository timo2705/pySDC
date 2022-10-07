from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.problem_classes.AllenCahn_FFT_gpu import allencahn_imex
from pySDC.helpers.gpu_hook import hook_gpu
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import filter_stats, sort_stats

# initialize level parameters
level_params = dict()
level_params['restol'] = 1E-08
level_params['dt'] = 1E-03
level_params['nsweeps'] = 1

# initialize sweeper parameters
sweeper_params = dict()
sweeper_params['collocation_class'] = CollGaussRadau_Right
sweeper_params['num_nodes'] = 3
sweeper_params['QI'] = 'LU' # For the IMEX sweeper, the LU-trick can be activated for the implicit part
sweeper_params['initial_guess'] = 'zero'

# initialize problem parameters
problem_params = dict()
problem_params['L'] = 64.0
problem_params['nvars'] = (1024, 1024)
problem_params['eps'] = 0.04
problem_params['dw'] = -23.6
problem_params['radius'] = 0.25
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
description['problem_class'] = allencahn_imex
description['problem_params'] = problem_params  # pass problem parameters
description['sweeper_class'] = imex_1st_order
description['sweeper_params'] = sweeper_params  # pass sweeper parameters
description['level_params'] = level_params  # pass level parameters
description['step_params'] = step_params  # pass step parameters

# set time parameters
t0 = 0.0
Tend = 32 * level_params['dt']

# instantiate controller
controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

# get initial values on finest level
P = controller.MS[0].levels[0].prob
uinit = P.u_exact(t0)
plt.imshow(uinit.get())
plt.show()
# call main function to get things done...
uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

plt.imshow(uend.get())
plt.show()

timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
print('Laufzeit:', timing[0][1])
