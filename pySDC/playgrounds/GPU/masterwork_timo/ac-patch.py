import numpy as np
import pickle

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.problem_classes.AllenCahn_FFT_gpu import allencahn_imex as ac_gpu
from pySDC.implementations.problem_classes.AllenCahn_FFT import allencahn_imex as ac_cpu
from pySDC.helpers.gpu_hook import hook_gpu
import matplotlib.pyplot as plt
from pySDC.helpers.stats_helper import filter_stats, sort_stats

name = 'pickle/ac-fft-pySDC-patch.pickle'
Ns = np.asarray([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
Ls = np.asarray([1, 1, 2, 4, 8, 16, 32, 64, 128])
# Ns = np.asarray([16, 32, 64, 128, 256, 512, 1024])
# Ls = np.asarray([1, 2, 4, 8, 16, 32, 64])
# initialize time array for parameters
times_cpu = np.zeros_like(Ns, dtype=float)
setup_cpu = np.zeros_like(Ns, dtype=float)
cg_cpu = np.zeros_like(Ns, dtype=float)
cg_Count_cpu = np.zeros_like(Ns)
f_im_cpu = np.zeros_like(Ns, dtype=float)
f_ex_cpu = np.zeros_like(Ns, dtype=float)
times_gpu = np.zeros_like(Ns, dtype=float)
setup_gpu = np.zeros_like(Ns, dtype=float)
cg_gpu = np.zeros_like(Ns, dtype=float)
cg_Count_gpu = np.zeros_like(Ns)
f_im_gpu = np.zeros_like(Ns, dtype=float)
f_ex_gpu = np.zeros_like(Ns, dtype=float)

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
# problem_params['L'] = 16.0
# problem_params['nvars'] = (256, 256)
problem_params['eps'] = 0.04
problem_params['dw'] = -23.6
problem_params['radius'] = 0.25
problem_params['name'] = "name"
problem_params['init_type'] = 'circle_rand'
problem_params['spectral'] = False

# initialize step parameters
step_params = dict()
step_params['maxiter'] = 50

# set time parameters
t0 = 0.0
schritte = 1
Tend = schritte * level_params['dt']
for i in range(len(Ns)):
    problem_params['L'] = Ls[i]
    problem_params['nvars'] = (Ns[i], Ns[i])

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = ac_cpu
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # get the stats
    timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
    times_cpu[i] = timing[0][1]
    timing_setup = sort_stats(filter_stats(stats, type='timing_setup'), sortby='time')
    setup_cpu[i] = timing_setup[0][1]
    timing_step = sort_stats(filter_stats(stats, type='timing_step'), sortby='time')
    timing_step = [ts[1] for ts in timing_step]
    cg_cpu[i] = np.asarray(sum(timing_step))
    # cg_Count[i] = P.lin_ncalls
    f_im_cpu[i] = P.f_im
    f_ex_cpu[i] = P.f_ex

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_gpu

    description['problem_class'] = ac_gpu

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # get the stats
    timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
    times_gpu[i] = timing[0][1]
    timing_setup = sort_stats(filter_stats(stats, type='timing_setup'), sortby='time')
    setup_gpu[i] = timing_setup[0][1]
    timing_step = sort_stats(filter_stats(stats, type='timing_step'), sortby='time')
    timing_step = [ts[1] for ts in timing_step]
    cg_gpu[i] = np.asarray(sum(timing_step))
    # cg_Count[i] = P.lin_ncalls
    f_im_gpu[i] = P.f_im
    f_ex_gpu[i] = P.f_ex

# write down stats to .pickle file
data = {
    'Ns': Ns[1:],
    'D': 2,
    'dt': level_params['dt'],
    'schritte': schritte,
    'iteration': step_params['maxiter'],
    'times-cpu': times_cpu[1:],
    'setup-cpu': setup_cpu[1:],
    'cg-time-cpu': cg_cpu[1:],
    # 'cg-count': cg_Count_cpu[1:],
    'f-time-imp-cpu': f_im_cpu[1:],
    'f-time-exp-cpu': f_ex_cpu[1:],
    'times-gpu': times_gpu[1:],
    'setup-gpu': setup_gpu[1:],
    'cg-time-gpu': cg_gpu[1:],
    # 'cg-count': cg_Count_gpu[1:],
    'f-time-imp-gpu': f_im_gpu[1:],
    'f-time-exp-gpu': f_ex_gpu[1:]
}
with open(name, 'wb') as f:
    pickle.dump(data, f)
# print(data)
print('done')
