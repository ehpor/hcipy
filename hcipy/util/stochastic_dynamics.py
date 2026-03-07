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

        noise = L @ self._rng.standard_normal(self.num_states)

        self._state = A_disc @ self._state + noise
        self._t = t

def make_continuous_time_companion_matrix(poles):
    """Create a continuous-time companion matrix from poles.

    For a system with poles p1, p2, ..., pn, creates the controllable
    companion form state matrix that has these poles as eigenvalues.

    Parameters
    ----------
    poles : array_like
        Poles (eigenvalues) of the system. For stable systems, poles
        should have negative real parts.

    Returns
    -------
    transition_matrix : ndarray
        Companion form state matrix (n x n).
    observation_matrix : ndarray
        Observation matrix (1 x n), selects the first state.
    """
    n = len(poles)

    coeffs = np.poly(poles)

    transition_matrix = np.zeros((n, n))
    transition_matrix[0, :] = -coeffs[1:n + 1]
    transition_matrix[1:, :-1] = np.eye(n - 1)

    observation_matrix = np.zeros((1, n))
    observation_matrix[0, 0] = 1.0

    return transition_matrix, observation_matrix

def ar_to_state_space(ar_coefficients, noise_variance, dt):
    """Convert a discrete-time AR(p) model to continuous-time state space.

    Converts a discrete-time autoregressive model:

    .. math::

        x_k = a_1 x_{k-1} + a_2 x_{k-2} + \cdots + a_p x_{k-p} + w_k

    to an equivalent continuous-time state space model.

    Parameters
    ----------
    ar_coefficients : array_like
        AR coefficients :math:`[a_1, a_2, ..., a_p]` for the discrete-time model.
    noise_variance : float
        Variance of the discrete-time driving noise.
    dt : float
        Sampling time of the discrete-time AR model.

    Returns
    -------
    transition_matrix : ndarray
        Continuous-time state transition matrix (:math:`p` x :math:`p`).
    noise_matrix : ndarray
        Continuous-time noise matrix (:math:`p` x 1).
    observation_matrix : ndarray
        Observation matrix (1 x :math:`p`), selects the first state.
    """
    ar_coefficients = np.asarray(ar_coefficients)
    p = len(ar_coefficients)

    # Discrete-time companion matrix (state = [x[k-1], x[k-2], ..., x[k-p]])
    A_disc = np.zeros((p, p))
    A_disc[0, :] = ar_coefficients
    A_disc[1:, :-1] = np.eye(p - 1)

    # Get discrete-time poles (eigenvalues of companion matrix)
    disc_poles = np.linalg.eigvals(A_disc)

    # Convert discrete-time poles to continuous-time poles
    # For stable discrete-time system: |z| < 1, so ln(z) has negative real part
    # Continuous-time pole: s = ln(z) / dt
    cont_poles = np.log(disc_poles) / dt

    # Create continuous-time companion matrix from continuous-time poles
    transition_matrix, observation_matrix = make_continuous_time_companion_matrix(cont_poles)

    # Noise matrix: discrete noise acts on first state
    # For continuous-time: need to scale appropriately
    # The continuous noise intensity is noise_variance / dt (approximate for small dt)
    noise_matrix = np.zeros((p, 1))
    noise_matrix[0, 0] = np.sqrt(noise_variance / dt)

    return transition_matrix, noise_matrix, observation_matrix

def make_random_state_space(n_states, bandwidth=1.0, seed=None):
    """Generate a random stable state space transition matrix.

    Creates a random state transition matrix with stable eigenvalues
    (all have negative real parts). Useful for testing or generating
    synthetic dynamics models.

    Parameters
    ----------
    n_states : int
        Number of internal states.
    bandwidth : float, optional
        Approximate bandwidth of the dynamics. Larger values give faster
        dynamics. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    transition_matrix : ndarray
        Random stable state transition matrix (n_states x n_states).
    """
    rng = np.random.default_rng(seed)

    eigenvalues = []
    while len(eigenvalues) < n_states:
        eig_real = -rng.uniform(0.1, 1.0) * bandwidth
        eig_imag = rng.uniform(-bandwidth, bandwidth)

        eigenvalues.append(complex(eig_real, eig_imag))

        if len(eigenvalues) < n_states and eig_imag != 0:
            eigenvalues.append(complex(eig_real, -eig_imag))

    eigenvalues = np.array(eigenvalues[:n_states])

    q, r = np.linalg.qr(rng.standard_normal((n_states, n_states)))
    transition_matrix = q @ np.diag(eigenvalues) @ q.conj().T
    transition_matrix = np.real(transition_matrix)

    return transition_matrix
