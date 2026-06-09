import numpy as np

class Estimator:
    '''The base class for all state estimators.

    Subclasses should implement :meth:`estimate` to produce a state estimate
    from measurements.
    '''
    def estimate(self, *args, **kwargs):
        '''Estimate the state from measurements.

        Parameters
        ----------
        *args
            Positional arguments for the estimator.
        **kwargs
            Keyword arguments for the estimator.

        Returns
        -------
        ndarray
            The state estimate.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        '''
        raise NotImplementedError

class KalmanFilter(Estimator):
    '''A discrete-time Kalman filter.

    This filter estimates the state of a linear dynamical system from
    noisy measurements. The system is modeled as::

        x_{k+1} = A x_k + B u_k + w_k,     w_k ~ N(0, Q)
        z_k     = C x_k + v_k,             v_k ~ N(0, R)

    The filter supports separate :meth:`predict` and :meth:`update` steps
    for multi-rate or asynchronous operation, as well as a combined
    :meth:`estimate` convenience method.

    Parameters
    ----------
    A : ndarray
        The state transition matrix.
    B : ndarray or None
        The control input matrix. Can be `None` if there is no known input.
    C : ndarray
        The measurement matrix.
    Q : ndarray
        The process noise covariance matrix.
    R : ndarray
        The measurement noise covariance matrix.
    x0 : ndarray or None
        The initial state estimate. If `None`, zeros are used.
    P0 : ndarray or None
        The initial error covariance matrix. If `None`, the identity matrix
        is used.
    '''
    def __init__(self, A, B, C, Q, R, x0=None, P0=None):
        self.A = np.asarray(A)
        self.B = np.asarray(B) if B is not None else None
        self.C = np.asarray(C)
        self.Q = np.asarray(Q)
        self.R = np.asarray(R)

        n = self.A.shape[0]

        if x0 is not None:
            self.x = np.asarray(x0, dtype=float)
        else:
            self.x = np.zeros(n)

        if P0 is not None:
            self.P = np.asarray(P0, dtype=float)
        else:
            self.P = np.eye(n)

    def _predict(self, dt=1, u=None):
        '''Predict the state and covariance forward in time.

        The state is propagated as ``x = A @ x + B @ u`` and the error
        covariance as ``P = A @ P @ A.T + Q * dt``.

        Parameters
        ----------
        dt : scalar
            The time step for the prediction. The process noise covariance
            is scaled linearly with `dt`.
        u : ndarray or None
            The control input at the current timestep. Only used if `B` was
            provided at construction and `u` is not `None`.
        '''
        self.x = self.A @ self.x
        if u is not None and self.B is not None:
            self.x += self.B @ np.asarray(u)
        self.P = self.A @ self.P @ self.A.T + self.Q * dt

    def _update(self, measurement):
        '''Update the state estimate from a measurement.

        The standard Kalman filter measurement update is performed::

            innovation = z - C @ x
            S = C @ P @ C.T + R
            K = P @ C.T @ inv(S)
            x = x + K @ innovation
            P = (I - K @ C) @ P

        Parameters
        ----------
        measurement : ndarray
            The measurement vector at the current timestep.
        '''
        measurement = np.asarray(measurement)

        innovation = measurement - self.C @ self.x
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        self.P = (np.eye(self.P.shape[0]) - K @ self.C) @ self.P

    def estimate(self, measurement, dt=1, u=None):
        '''Perform a complete predict-update cycle.

        This is a convenience method that calls :meth:`predict` followed
        by :meth:`update` and returns the updated state estimate.

        Parameters
        ----------
        measurement : ndarray
            The measurement vector at the current timestep.
        dt : scalar
            The time step for the prediction.
        u : ndarray or None
            The control input at the current timestep.

        Returns
        -------
        ndarray
            The updated state estimate.
        '''
        self._predict(dt, u)
        self._update(measurement)

        return self.x
