import numpy as np
import time
from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class allencahn_imex(ptype):
    """
    Example implementing Allen-Cahn equation in 2-3D using cupy-fft for solving linear parts, IMEX time-stepping

    Attributes:
        X: grid coordinates in real space
        K2: Laplace operator in spectral space
        dx: mesh width in x direction
        dy: mesh width in y direction
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: fft data type (will be passed to parent class)
            dtype_f: fft data type wuth implicit and explicit parts (will be passed to parent class)
        """

        if 'L' not in problem_params:
            problem_params['L'] = 1.0
        if 'init_type' not in problem_params:
            problem_params['init_type'] = 'circle'
        if 'comm' not in problem_params:
            problem_params['comm'] = None
        if 'dw' not in problem_params:
            problem_params['dw'] = 0.0

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'eps', 'L', 'radius', 'dw', 'spectral']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        if not (isinstance(problem_params['nvars'], tuple) and len(problem_params['nvars']) > 1):
            raise ProblemError('Need at least two dimensions')

        # Creating FFT structure
        ndim = len(problem_params['nvars'])

        # invoke super init, passing the communicator and the local dimensions as init
        super(allencahn_imex, self).__init__(init=(problem_params['nvars'], None, np.dtype('float64')),
                                             dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        L = np.array([self.params.L] * ndim, dtype=float)

        # get local mesh
        X = np.ogrid[0:self.init[0][0], 0:self.init[0][1]]
        N = self.init[0]
        for i in range(len(N)):
            X[i] = (X[i] * L[i] / N[i])
        self.X = [np.broadcast_to(x, N) for x in X]

        # get local wavenumbers and Laplace operator
        s = (slice(0, N[0]), slice(0, int(N[0]/2+1)))
        N = self.init[0]
        k = [np.fft.fftfreq(n, 1. / n).astype(int) for n in N[:-1]]
        k.append(np.fft.rfftfreq(N[-1], 1. / N[-1]).astype(int))
        K = [ki[si] for ki, si in zip(k, s)]
        Ks = np.meshgrid(*K, indexing='ij', sparse=True)
        Lp = 2 * np.pi / L
        for i in range(ndim):
            Ks[i] = (Ks[i] * Lp[i]).astype(float)
        K = [np.broadcast_to(k, (N[0], int(N[0]/2+1))) for k in Ks]
        K = np.array(K).astype(float)
        self.K2 = np.sum(K * K, 0, dtype=float)

        # Need this for diagnostics
        self.dx = self.params.L / problem_params['nvars'][0]
        self.dy = self.params.L / problem_params['nvars'][1]

        self.f_im = 0
        self.f_ex = 0

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

        if self.params.spectral:

            f.impl = -self.K2 * u

            if self.params.eps > 0:
                tmp = np.fft.irfft2(u)
                tmpf = - 2.0 / self.params.eps ** 2 * tmp * (1.0 - tmp) * (1.0 - 2.0 * tmp) - \
                    6.0 * self.params.dw * tmp * (1.0 - tmp)
                f.expl[:] = np.fft.rfft2(tmpf)

        else:
            start = time.perf_counter()
            u_hat = np.fft.rfft2(u)
            lap_u_hat = -self.K2 * u_hat
            f.impl[:] = np.fft.irfft2(lap_u_hat)
            end = time.perf_counter()
            self.f_im += end - start
            start = time.perf_counter()
            if self.params.eps > 0:
                f.expl = - 2.0 / self.params.eps ** 2 * u * (1.0 - u) * (1.0 - 2.0 * u) - \
                    6.0 * self.params.dw * u * (1.0 - u)
            end = time.perf_counter()
            self.f_ex += end - start
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        if self.params.spectral:

            me = rhs / (1.0 + factor * self.K2)

        else:

            me = self.dtype_u(self.init)
            rhs_hat = np.fft.rfft2(rhs)
            rhs_hat /= (1.0 + factor * self.K2)
            me[:] = np.fft.irfft2(rhs_hat)

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init, val=0.0)
        if self.params.init_type == 'circle':
            r2 = (self.X[0] - 0.5) ** 2 + (self.X[1] - 0.5) ** 2
            if self.params.spectral:
                tmp = 0.5 * (1.0 + np.tanh((self.params.radius - np.sqrt(r2)) / (np.sqrt(2) * self.params.eps)))
                me[:] = np.fft.rfft2(tmp, norm="forward")
            else:
                me[:] = 0.5 * (1.0 + np.tanh((self.params.radius - np.sqrt(r2)) / (np.sqrt(2) * self.params.eps)))
        elif self.params.init_type == 'circle_rand':
            ndim = len(me.shape)
            L = int(self.params.L)
            # get random radii for circles/spheres
            np.random.seed(1)
            lbound = 3.0 * self.params.eps
            ubound = 0.5 - self.params.eps
            rand_radii = (ubound - lbound) * np.random.random_sample(size=tuple([L] * ndim)) + lbound
            # distribute circles/spheres
            # tmp = newDistArray(self.fft, False)
            tmp = np.zeros_like(me)
            if ndim == 2:
                for i in range(0, L):
                    for j in range(0, L):
                        # build radius
                        r2 = (self.X[0] + i - L + 0.5) ** 2 + (self.X[1] + j - L + 0.5) ** 2
                        # add this blob, shifted by 1 to avoid issues with adding up negative contributions
                        tmp += np.tanh((rand_radii[i, j] - np.sqrt(r2)) / (np.sqrt(2) * self.params.eps)) + 1
            # normalize to [0,1]
            tmp *= 0.5
            assert np.all(tmp <= 1.0)
            if self.params.spectral:
                me[:] = np.fft.rfft2(tmp)
            else:
                me[:] = tmp[:]
        else:
            raise NotImplementedError('type of initial value not implemented, got %s' % self.params.init_type)

        return me


class allencahn_imex_timeforcing(allencahn_imex):
    """
    Example implementing Allen-Cahn equation in 2-3D using mpi4py-fft for solving linear parts, IMEX time-stepping,
    time-dependent forcing
    """

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

        if self.params.spectral:

            f.impl = -self.K2 * u

            tmp = np.zeros_like(u)
            tmp[:] = np.fft.rfft2(u, norm="backward")

            if self.params.eps > 0:
                tmpf = -2.0 / self.params.eps ** 2 * tmp * (1.0 - tmp) * (1.0 - 2.0 * tmp)
            else:
                tmpf = self.dtype_f(self.init, val=0.0)

            # build sum over RHS without driving force
            Rt_global = float(np.sum(np.fft.rfft2(f.impl, norm="backward") + tmpf))

            # build sum over driving force term
            Ht_global = float(np.sum(6.0 * tmp * (1.0 - tmp)))

            # add/substract time-dependent driving force
            if Ht_global != 0.0:
                dw = Rt_global / Ht_global
            else:
                dw = 0.0

            tmpf -= 6.0 * dw * tmp * (1.0 - tmp)
            f.expl[:] = np.fft.rfft2(tmpf, norm="forward")

        else:

            u_hat = np.fft.rfft2(u, norm="forward")
            lap_u_hat = -self.K2 * u_hat
            f.impl[:] = np.fft.rfft2(lap_u_hat, norm="backward")

            if self.params.eps > 0:
                f.expl = -2.0 / self.params.eps ** 2 * u * (1.0 - u) * (1.0 - 2.0 * u)

            # build sum over RHS without driving force
            Rt_global = float(np.sum(f.impl + f.expl))

            # build sum over driving force term
            Ht_global = float(np.sum(6.0 * u * (1.0 - u)))

            # add/substract time-dependent driving force
            if Ht_global != 0.0:
                dw = Rt_global / Ht_global
            else:
                dw = 0.0

            f.expl -= 6.0 * dw * u * (1.0 - u)

        return f
