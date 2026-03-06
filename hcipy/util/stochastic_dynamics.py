import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

class StateSpaceDynamics:
    r"""A continuous-time state space stochastic process.

    This class generates trajectories of a continuous-time state space model
    of the form

    .. math::

        \frac{\mathrm{d} x}{\mathrm{d}t} = Ax + Bw,

    where x is the internal state, :math:`A` is the transition matrix, :math:`B`
    is the noise matrix, and :math:`w` is white noise with unit variance.

    The observation equation

    .. math::

        y = Cx

    maps the internal state to output coefficients.

    Parameters
    ----------
    transition_matrix : array_like
        Continuous-time state transition matrix :math:`A` (num_states x num_states).
    noise_matrix : array_like
        Continuous-time noise/input matrix :math:`B` (num_states x num_inputs).
    observation_matrix : array_like, optional
        Observation matrix :math:`C` (num_outputs x num_states). If None, uses identity
        matrix (state directly maps to output).
    initial_state : array_like, optional
        Initial state vector. If None, samples from the stationary distribution.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    transition_matrix : array_like
        Continuous-time state transition matrix A.
    noise_matrix : array_like
        Continuous-time noise/input matrix B.
    observation_matrix : array_like
        Observation matrix C.
    """
    def __init__(self, transition_matrix, noise_matrix, observation_matrix=None, initial_state=None, seed=None):
        self.transition_matrix = transition_matrix
        self.noise_matrix = noise_matrix

        n_states = self.transition_matrix.shape[0]

        if observation_matrix is None:
            self.observation_matrix = np.eye(n_states)
        else:
            self.observation_matrix = observation_matrix

        self._t = 0.0
        self._rng = np.random.default_rng(seed)

        noise_covariance = self.noise_matrix @ self.noise_matrix.T
        self._P = solve_continuous_lyapunov(self.transition_matrix, -noise_covariance)

        if initial_state is None:
            eigenvalues, eigenvectors = np.linalg.eigh(self._P)
            eigenvalues = np.maximum(eigenvalues, 0)
            L = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            self._state = L @ self._rng.standard_normal(n_states)
        else:
            self._state = initial_state

    @property
    def state(self):
        """Current internal state vector.
        """
        return self._state

    @property
    def coefficients(self):
        """Current output coefficients (y = C @ x).
        """
        return self.observation_matrix @ self._state

    @property
    def t(self):
        """Current simulation time.
        """
        return self._t

    @t.setter
    def t(self, t):
        self.evolve_until(t)

    @property
    def num_states(self):
        """Number of internal states.
        """
        return self.transition_matrix.shape[0]

    @property
    def num_outputs(self):
        """Number of output coefficients.
        """
        return self.observation_matrix.shape[0]

    @property
    def stationary_covariance(self):
        """Stationary covariance matrix P of the state.
        """
        return self._P

    def evolve_until(self, t):
        """Evolve the state from current time to time t.

        Uses exact matrix exponential for the deterministic part and
        exact noise covariance for the stochastic part.

        Parameters
        ----------
        t : float
            Target time to evolve to. Must be greater than or equal to
            the current time.

        Raises
        ------
        ValueError
            If t is less than the current time (backwards evolution not allowed).
        """
        if t < self._t:
            raise ValueError(f"Backwards evolution not allowed: target time {t} is less than current time {self._t}")

        if t == self._t:
            return

        dt = t - self._t

        A_disc = expm(self.transition_matrix * dt)
        Q_disc = self._P - A_disc @ self._P @ A_disc.T

        eigenvalues, eigenvectors = np.linalg.eigh(Q_disc)
        eigenvalues = np.maximum(eigenvalues, 0)
        L = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        noise = L @ self._rng.standard_normal(self.n_states)

        self._state = A_disc @ self._state + noise
        self._t = t

def make_continuous_time_companion(poles):
