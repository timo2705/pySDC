import numpy as np
import cupy as cp
import cupyx.scipy.sparse as csp
from cupyx.scipy.sparse.linalg import spsolve, cg  # , gmres, minres

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh
# from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh11 as cupy_mesh
# from pySDC.implementations.datatype_classes.cupy_mesh import imex_cupy_mesh11 as imex_cupy_mesh


# noinspection PyUnusedLocal
class heatNd_periodic(ptype):
    """
    Example implementing the unforced ND heat equation with periodic BCs in [0,1]^N,
    discretized using central finite differences

    Attributes:
        A: second-order FD discretization of the ND laplace operator
        dx: distance between two spatial nodes (here: being the same in all dimensions)
    """

    def __init__(self, problem_params, dtype_u=cupy_mesh, dtype_f=imex_cupy_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence

        if 'order' not in problem_params:
            problem_params['order'] = 2
        if 'lintol' not in problem_params:
            problem_params['lintol'] = 1E-12
        if 'liniter' not in problem_params:
            problem_params['liniter'] = 10000
        if 'direct_solver' not in problem_params:
            problem_params['direct_solver'] = False

        essential_keys = ['nvars', 'nu', 'freq', 'order', 'ndim', 'lintol', 'liniter', 'direct_solver']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # make sure parameters have the correct form
        if problem_params['ndim'] > 3:
            raise ProblemError(f'can work with up to three dimensions, got {problem_params["ndim"]}')
        if type(problem_params['freq']) is not tuple or len(problem_params['freq']) != problem_params['ndim']:
            raise ProblemError(f'need {problem_params["ndim"]} frequencies, got {problem_params["freq"]}')
        for freq in problem_params['freq']:
            if freq % 2 != 0:
                raise ProblemError('need even number of frequencies due to periodic BCs')
        if type(problem_params['nvars']) is not tuple or len(problem_params['nvars']) != problem_params['ndim']:
            raise ProblemError(f'need {problem_params["ndim"]} nvars, got {problem_params["nvars"]}')
        for nvars in problem_params['nvars']:
            if nvars % 2 != 0:
                raise ProblemError('the setup requires nvars = 2^p per dimension')
        if problem_params['nvars'][1:] != problem_params['nvars'][:-1]:
            raise ProblemError('need a square domain, got %s' % problem_params['nvars'])

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(heatNd_periodic, self).__init__(init=(problem_params['nvars'], None, cp.dtype('float64')),
                                              dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        # compute dx (equal in both dimensions) and get discretization matrix A
        self.dx = 1.0 / self.params.nvars[0]
        self.A = self.__get_A(self.params.nvars, self.params.nu, self.dx, self.params.ndim, self.params.order)
        xvalues = cp.array([i * self.dx for i in range(self.params.nvars[0])])
        self.xv = cp.meshgrid(*[xvalues for _ in range(self.params.ndim)])
        self.Id = csp.eye(self.params.nvars[0] ** self.params.ndim, format='csr')

        self.start_gpu = cp.cuda.Event()
        self.end_gpu = cp.cuda.Event()
        self.f_im = 0
        self.f_ex = 0
        self.f_im_count = 0
        self.f_ex_count = 0

    @staticmethod
    def __get_A(N, nu, dx, ndim, order):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (list): number of dofs
            nu (float): diffusion coefficient
            dx (float): distance between two spatial nodes
            ndim (int): number of dimensions

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        if order == 2:
            stencil = [1, -2, 1]
            zero_pos = 2
        elif order == 4:
            stencil = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]
            zero_pos = 3
        elif order == 6:
            stencil = [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]
            zero_pos = 4
        elif order == 8:
            stencil = [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560]
            zero_pos = 5
        else:
            raise ProblemError(f'wrong order given, has to be 2, 4, 6, or 8, got {order}')
        dstencil = np.concatenate((stencil, np.delete(stencil, zero_pos - 1)))
        offsets = np.concatenate(([N[0] - i - 1 for i in reversed(range(zero_pos - 1))],
                                  [i - zero_pos + 1 for i in range(zero_pos - 1, len(stencil))]))
        doffsets = np.concatenate((offsets, np.delete(offsets, zero_pos - 1) - N[0]))

        A = csp.diags(dstencil, doffsets, shape=(N[0], N[0]), format='csc')
        # stencil = [1, -2, 1]
        # A = sp.diags(stencil, [-1, 0, 1], shape=(N[0], N[0]), format='csc')
        if ndim == 2:
            A = csp.kron(A, csp.eye(N[0])) + csp.kron(csp.eye(N[1]), A)
        elif ndim == 3:
            A = csp.kron(A, csp.eye(N[1] * N[0])) + csp.kron(csp.eye(N[2] * N[1]), A) + \
                csp.kron(csp.kron(csp.eye(N[2]), A), csp.eye(N[0]))
        A *= nu / (dx ** 2)

        return A

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init)
        self.start_gpu.record()
        v = u.flatten()
        f.impl[:] = self.A.dot(v).reshape(self.params.nvars)
        self.end_gpu.record()
        self.end_gpu.synchronize()
        self.f_im += cp.cuda.get_elapsed_time(self.start_gpu, self.end_gpu)/1000
        self.f_im_count += 1
        self.start_gpu.record()
        if self.params.ndim == 1:
            f.expl[:] = cp.sin(np.pi * self.params.freq[0] * self.xv[0]) * (self.params.nu * np.pi ** 2 *
                                                                            sum([freq ** 2 for freq in
                                                                                 self.params.freq]) *
                                                                            cp.cos(t) - cp.sin(t))
        elif self.params.ndim == 2:
            f.expl[:] = cp.sin(np.pi * self.params.freq[0] * self.xv[0]) * cp.sin(np.pi * self.params.freq[1] *
                                                                                  self.xv[1]) * (self.params.nu *
                                                                                                 np.pi ** 2 *
                                                                                                 sum([freq ** 2 for freq
                                                                                                      in
                                                                                                      self.params.freq]
                                                                                                     ) * cp.cos(t) -
                                                                                                 cp.sin(t)
                                                                                                 )
        elif self.params.ndim == 3:
            f.expl[:] = cp.sin(np.pi * self.params.freq[0] * self.xv[0]) * cp.sin(np.pi * self.params.freq[1] *
                                                                                  self.xv[1]) * cp.sin(
                np.pi * self.params.freq[2] * self.xv[2]) * (self.params.nu * np.pi ** 2 * sum([freq ** 2 for freq in
                                                                                                self.params.freq]) *
                                                             cp.cos(t) - cp.sin(t))
        self.end_gpu.record()
        self.end_gpu.synchronize()
        self.f_ex += cp.cuda.get_elapsed_time(self.start_gpu, self.end_gpu)/1000
        self.f_ex_count += 1
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

        if self.params.direct_solver:
            me[:] = spsolve(self.Id - factor * self.A, rhs.flatten()).reshape(self.params.nvars)
        else:
            me[:] = cg(self.Id - factor * self.A, rhs.flatten(), x0=u0.flatten(),
                       tol=self.params.lintol, maxiter=self.params.liniter)[0].reshape(self.params.nvars)
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
        if self.params.ndim == 1:
            me[:] = cp.sin(np.pi * self.params.freq[0] * self.xv[0]) * cp.cos(t)
        elif self.params.ndim == 2:
            me[:] = cp.sin(np.pi * self.params.freq[0] * self.xv[0]) * \
                cp.sin(np.pi * self.params.freq[1] * self.xv[1]) * cp.cos(t)
        elif self.params.ndim == 3:
            me[:] = cp.sin(np.pi * self.params.freq[0] * self.xv[0]) * \
                cp.sin(np.pi * self.params.freq[1] * self.xv[1]) * \
                cp.sin(np.pi * self.params.freq[2] * self.xv[2]) * cp.cos(t)
        return me
