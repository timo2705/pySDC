import numpy as np

from dedalus import public as de

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError

from pySDC.playgrounds.Dedalus.dedalus_mesh import dedalus_mesh, rhs_imex_dedalus_mesh


class heat1d_dedalus_forced(ptype):
    """
    Example implementing the forced 1D heat equation with periodic BC in [0,1], discretized using Dedalus
    """
    def __init__(self, problem_params, dtype_u=dedalus_mesh, dtype_f=rhs_imex_dedalus_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'nu', 'freq', 'scale']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if problem_params['freq'] % 2 != 0:
            raise ProblemError('setup requires freq to be an equal number')

        xbasis = de.Fourier('x', problem_params['nvars'], interval=(0, 1), dealias=1)
        domain = de.Domain([xbasis], grid_dtype=np.float64, mesh=[1])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heat1d_dedalus_forced, self).__init__(init=domain, dtype_u=dtype_u, dtype_f=dtype_f,
                                                    params=problem_params)

        self.x = self.init.grid(0, scales=self.params.scale)
        self.rhs = self.dtype_u(self.init, val=0.0)
        self.problem = de.IVP(domain=self.init, variables=['u'])
        self.problem.parameters['nu'] = self.params.nu
        self.problem.add_equation("dt(u) - nu * dx(dx(u)) = 0")
        ts = de.timesteppers.SBDF1
        self.solver = self.problem.build_solver(ts)

    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS with two parts
        """

        f = self.dtype_f(self.init)
        f.impl.values = (self.params.nu * de.operators.differentiate(u.values, x=2)).evaluate()
        f.expl.values['g'] = -np.sin(np.pi * self.params.freq * self.x) * (np.sin(t) - (np.pi * self.params.freq) ** 2 * np.cos(t))
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)

        if factor != 0.0:

            u = self.solver.state['u']

            u['g'] = rhs.values['g']

            self.solver.step(factor)

            me.values = u

        else:

            me.values = rhs.values

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init)
        me.values['g'] = np.sin(np.pi * self.params.freq * self.x) * np.cos(t)
        return me
