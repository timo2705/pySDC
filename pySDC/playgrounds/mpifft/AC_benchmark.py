import sys
import numpy as np
from mpi4py import MPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex
from pySDC.projects.AllenCahn_Bayreuth.AllenCahn_dump import dump
from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft
import matplotlib.pyplot as plt

def run_simulation(name=''):
    """
    A simple test program to do PFASST runs for the AC equation
    """

    # set MPI communicator
    comm = MPI.COMM_WORLD

    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    # split world communicator to create space-communicators
    if len(sys.argv) >= 2:
        color = int(world_rank / int(sys.argv[1]))
    else:
        color = int(world_rank / 1)
    space_comm = comm.Split(color=color)
    # space_size = space_comm.Get_size()
    space_rank = space_comm.Get_rank()

    # split world communicator to create time-communicators
    if len(sys.argv) >= 2:
        color = int(world_rank % int(sys.argv[1]))
    else:
        color = int(world_rank / world_size)
    time_comm = comm.Split(color=color)
    # time_size = time_comm.Get_size()
    time_rank = time_comm.Get_rank()

    # print("IDs (world, space, time):  %i / %i -- %i / %i -- %i / %i" % (world_rank, world_size, space_rank,
    #                                                                     space_size, time_rank, time_size))

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = 1E-03
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['L'] = 64.0
    problem_params['nvars'] = (1024, 1024)
    problem_params['eps'] = 0.04
    problem_params['dw'] = -23.6
    problem_params['radius'] = 0.25
    problem_params['comm'] = space_comm
    problem_params['name'] = name
    problem_params['init_type'] = 'circle_rand'
    problem_params['spectral'] = False

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30 if space_rank == 0 else 99  # set level depending on rank
    # controller_params['hook_class'] = dump

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = allencahn_imex
    # description['problem_class'] = allencahn_imex_timeforcing
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    # set time parameters
    t0 = 0.0
    Tend = 32 * level_params['dt']

    # instantiate controller
    controller = controller_MPI(controller_params=controller_params, description=description, comm=time_comm)

    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    plt.imshow(uinit)
    plt.show()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    plt.imshow(uend)
    plt.show()
    timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
    print('Laufzeit:', timing[0][1])


if __name__ == "__main__":
    name = 'AC-test-constforce'
    run_simulation(name=name)
