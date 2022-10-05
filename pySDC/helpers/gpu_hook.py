import logging
from collections import namedtuple
import cupy as cp
from pySDC.core.Hooks import hooks


class hook_gpu(hooks):
    """
        Hook class to contain the functions called during the controller runs (e.g. for calling user-routines)

        Attributes:
            __t0_setup (float): private variable to get starting time of setup
            __t0_run (float): private variable to get starting time of the run
            __t0_predict (float): private variable to get starting time of the predictor
            __t0_step (float): private variable to get starting time of the step
            __t0_iteration (float): private variable to get starting time of the iteration
            __t0_sweep (float): private variable to get starting time of the sweep
            __t0_comm (list): private variable to get starting time of the communication
            __t1_run (float): private variable to get end time of the run
            __t1_predict (float): private variable to get end time of the predictor
            __t1_step (float): private variable to get end time of the step
            __t1_iteration (float): private variable to get end time of the iteration
            __t1_sweep (float): private variable to get end time of the sweep
            __t1_setup (float): private variable to get end time of setup
            __t1_comm (list): private variable to hold timing of the communication (!)
            logger: logger instance for output
            __stats (dict): dictionary for gathering the statistics of a run
            __entry (namedtuple): statistics entry containign all information to identify the value
            __start_gpu (cupy.cuda.Event): start event to measure time for the gpu
            __end_gpu (cupy.cuda.Event): end event to measure time for the gpu
        """

    def __init__(self):
        """
        Initialization routine
        """
        self.__t0_setup = cp.cuda.Event()
        self.__t0_run = cp.cuda.Event()
        self.__t0_predict = cp.cuda.Event()
        self.__t0_step = cp.cuda.Event()
        self.__t0_iteration = cp.cuda.Event()
        self.__t0_sweep = cp.cuda.Event()
        self.__t0_comm = []
        self.__t1_run = cp.cuda.Event()
        self.__t1_predict = cp.cuda.Event()
        self.__t1_step = cp.cuda.Event()
        self.__t1_iteration = cp.cuda.Event()
        self.__t1_sweep = cp.cuda.Event()
        self.__t1_setup = cp.cuda.Event()
        self.__t1_comm = []

        self.logger = logging.getLogger('hook_gpu')

        # create statistics and entry elements
        self.__stats = {}
        self.__entry = namedtuple('Entry', ['process', 'time', 'level', 'iter', 'sweep', 'type'])

        self.__start_gpu = cp.cuda.Event()
        self.__end_gpu = cp.cuda.Event()

    def add_to_stats(self, process, time, level, iter, sweep, type, value):
        """
        Routine to add data to the statistics dict

        Args:
            process: the current process recording this data
            time (float): the current simulation time
            level (int): the current level index
            iter (int): the current iteration count
            sweep (int): the current sweep count
            type (str): string to describe the type of value
            value: the actual data
        """
        # create named tuple for the key and add to dict
        self.__stats[self.__entry(process=process, time=time, level=level, iter=iter, sweep=sweep, type=type)] = value

    def increment_stats(self, process, time, level, iter, sweep, type, value, initialize=None):
        """
        Routine to increment data to the statistics dict. If the data is not yet created, it will be initialized to
        initialize if applicable and to value otherwise

        Args:
            process: the current process recording this data
            time (float): the current simulation time
            level (int): the current level index
            iter (int): the current iteration count
            sweep (int): the current sweep count
            type (str): string to describe the type of value
            value: the actual data
            initialize: if supplied and data does not exist already, this will be used over value
        """
        key = self.__entry(process=process, time=time, level=level, iter=iter, sweep=sweep, type=type)
        if key in self.__stats.keys():
            self.__stats[key] += value
        elif initialize is not None:
            self.__stats[key] = initialize
        else:
            self.__stats[key] = value

    def return_stats(self):
        """
        Getter for the stats

        Returns:
            dict: stats
        """
        return self.__stats

    def reset_stats(self):
        """
        Function to reset the stats for multiple runs
        """
        self.__stats = {}

    def pre_setup(self, step, level_number):
        """
        Default routine called before setup starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__t0_setup.record()

    def pre_run(self, step, level_number):
        """
        Default routine called before time-loop starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__t0_run.record()

    def pre_predict(self, step, level_number):
        """
        Default routine called before predictor starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__t0_predict.record()

    def pre_step(self, step, level_number):
        """
        Hook called before each step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__t0_step.record()

    def pre_iteration(self, step, level_number):
        """
        Default routine called before iteration starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__t0_iteration.record()

    def pre_sweep(self, step, level_number):
        """
        Default routine called before sweep starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__t0_sweep.record()

    def pre_comm(self, step, level_number):
        """
        Default routine called before communication starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """


    def post_comm(self, step, level_number, add_to_stats=False):
        """
        Default routine called after each communication

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
            add_to_stats (bool): set if result should go to stats object
        """


    def post_sweep(self, step, level_number):
        """
        Default routine called after each sweep

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__t1_sweep.record()
        self.__t1_sweep.synchronize()

        L = step.levels[level_number]

        self.logger.info('Process %2i on time %8.6f at stage %15s: Level: %s -- Iteration: %2i -- Sweep: %2i -- '
                         'residual: %12.8e',
                         step.status.slot, L.time, step.status.stage, L.level_index, step.status.iter, L.status.sweep,
                         L.status.residual)

        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='residual_post_sweep', value=L.status.residual)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='timing_sweep',
                          value=cp.cuda.get_elapsed_time(self.__t0_sweep, self.__t1_sweep)/1000)

    def post_iteration(self, step, level_number):
        """
        Default routine called after each iteration

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """

        self.__t1_iteration.record()
        self.__t1_iteration.synchronize()

        L = step.levels[level_number]

        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='residual_post_iteration', value=L.status.residual)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='timing_iteration',
                          value=cp.cuda.get_elapsed_time(self.__t0_iteration, self.__t1_iteration)/1000)

    def post_step(self, step, level_number):
        """
        Default routine called after each step or block

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """

        self.__t1_step.record()
        self.__t1_step.synchronize()

        L = step.levels[level_number]

        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='timing_step',
                          value=cp.cuda.get_elapsed_time(self.__t0_step, self.__t1_step)/1000)
        self.add_to_stats(process=step.status.slot, time=L.time, level=-1, iter=step.status.iter,
                          sweep=L.status.sweep, type='niter', value=step.status.iter)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=-1,
                          sweep=L.status.sweep, type='residual_post_step', value=L.status.residual)

        # record the recomputed quantities at weird positions to make sure there is only one value for each step
        self.add_to_stats(process=-1, time=L.time + L.dt, level=-1, iter=-1,
                          sweep=-1, type='recomputed', value=step.status.restart)
        self.add_to_stats(process=-1, time=L.time, level=-1, iter=-1,
                          sweep=-1, type='recomputed', value=step.status.restart)

    def post_predict(self, step, level_number):
        """
        Default routine called after each predictor

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__t1_predict.record()
        self.__t1_predict.synchronize()

        L = step.levels[level_number]

        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='timing_predictor',
                          value=cp.cuda.get_elapsed_time(self.__t0_predict, self.__t1_predict)/1000)

    def post_run(self, step, level_number):
        """
        Default routine called after each run

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__t1_run.record()
        self.__t1_run.synchronize()

        L = step.levels[level_number]

        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='timing_run',
                          value=cp.cuda.get_elapsed_time(self.__t0_run, self.__t1_run)/1000)

    def post_setup(self, step, level_number):
        """
        Default routine called after setup

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        self.__t1_setup.record()
        self.__t1_setup.synchronize()

        self.add_to_stats(process=-1, time=-1, level=-1, iter=-1, sweep=-1, type='timing_setup',
                          value=cp.cuda.get_elapsed_time(self.__t0_setup, self.__t1_setup)/1000)

