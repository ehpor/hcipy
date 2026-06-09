import numpy as np

class Controller:
    '''The base class for all controllers.

    Subclasses should implement :meth:`compute` to produce a control signal
    from a measurement or state estimate.
    '''
    def compute(self, *args, **kwargs):
        '''Go from a measurement or state estimate to a control signal.

        Parameters
        ----------
        *args
            Positional arguments for the controller.
        **kwargs
            Keyword arguments for the controller.

        Returns
        -------
        ndarray
            The control signal.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        '''
        raise NotImplementedError

class IntegralController(Controller):
    '''A discrete-time integral controller with optional leak.

    This controller computes a control signal as the accumulated sum of
    gain-scaled measurements. A multiplicative leak factor can be applied
    at each timestep to prevent windup.

    Parameters
    ----------
    gain_matrix : scalar or ndarray
        The gain applied to each measurement before integration. Can be a
        scalar (uniform gain), a 1D array (per-element gain), or a 2D array
        (full matrix gain).
    leak : scalar
        The multiplicative leak factor applied to the integrator state at
        each timestep. A value of 1 corresponds to no leak.
    leak_tau : scalar or None
        If provided, overrides `leak` by computing a per-step leak factor
        as ``exp(-dt / leak_tau)``. This allows the leak to be specified as
        a time constant.
    '''
    def __init__(self, gain_matrix, leak=1, leak_tau=None):
        self.gain_matrix = np.asarray(gain_matrix)
        self.leak = leak
        self.leak_tau = leak_tau
        self._accumulator = None

    def compute(self, measurement, dt=1):
        '''Compute the control signal from a measurement.

        Parameters
        ----------
        measurement : ndarray
            The current measurement or error signal.
        dt : scalar
            The time step since the last call.

        Returns
        -------
        ndarray
            The control signal.
        '''
        measurement = np.asarray(measurement)

        if self._accumulator is None:
            output_dim = self._output_dim(measurement)
            self._accumulator = np.zeros(output_dim)

        leak_factor = self.leak
        if self.leak_tau is not None:
            leak_factor = np.exp(-dt / self.leak_tau)

        self._accumulator *= leak_factor
        self._accumulator += self._apply_gain(self.gain_matrix, measurement) * dt

        return self._accumulator

    def _apply_gain(self, gain, x):
        if gain.ndim <= 1:
            return gain * x
        else:
            return gain @ x

    def _output_dim(self, x):
        if self.gain_matrix.ndim <= 1:
            return np.shape(x)
        else:
            return (self.gain_matrix.shape[0],)

class PIDController(Controller):
    '''A discrete-time PID controller with optional derivative filtering.

    The control signal is computed as the sum of three terms: proportional,
    integral, and derivative. The derivative term can be low-pass filtered
    using a time constant `tau_d` to reduce noise amplification.

    Parameters
    ----------
    Kp : scalar or ndarray
        The proportional gain.
    Ki : scalar or ndarray
        The integral gain.
    Kd : scalar or ndarray
        The derivative gain.
    tau_d : scalar or None
        The time constant for the derivative low-pass filter. If `None`,
        no filtering is applied.
    '''
    def __init__(self, Kp, Ki, Kd, tau_d=None):
        self.Kp = np.asarray(Kp)
        self.Ki = np.asarray(Ki)
        self.Kd = np.asarray(Kd)
        self.tau_d = tau_d
        self._integral = None
        self._prev_error = None
        self._prev_derivative = None

    def compute(self, error, dt=1):
        '''Compute the control signal from an error signal.

        Parameters
        ----------
        error : ndarray
            The current error signal.
        dt : scalar
            The time step since the last call.

        Returns
        -------
        ndarray
            The control signal.
        '''
        error = np.asarray(error)

        if self._integral is None:
            output_dim = self._output_dim(self.Ki, error)
            self._integral = np.zeros(output_dim)

        if self._prev_error is None:
            self._prev_error = np.zeros_like(error)

        output_dim_control = self._output_dim(self.Kp, error)

        p_term = self._apply_gain(self.Kp, error)

        i_increment = self._apply_gain(self.Ki, error) * dt
        self._integral += i_increment
        i_term = self._integral

        d_error = (error - self._prev_error) / dt
        if self.tau_d is not None:
            if self._prev_derivative is None:
                self._prev_derivative = np.zeros(output_dim_control)
            alpha = dt / (self.tau_d + dt)
            self._prev_derivative = (1 - alpha) * self._prev_derivative + alpha * d_error
            d_term = self._apply_gain(self.Kd, self._prev_derivative)
        else:
            d_term = self._apply_gain(self.Kd, d_error)

        self._prev_error = error.copy()

        return p_term + i_term + d_term

    def _apply_gain(self, gain, x):
        if gain.ndim <= 1:
            return gain * x
        else:
            return gain @ x

    def _output_dim(self, gain, x):
        if gain.ndim <= 1:
            return np.shape(x)
        else:
            return (gain.shape[0],)

class LQRController(Controller):
    '''A linear-quadratic regulator controller.

    This controller computes the control signal as ``-K @ measurement``
    where ``K`` is the optimal state-feedback gain matrix obtained from
    solving the discrete-time algebraic Riccati equation.

    Parameters
    ----------
    K : ndarray
        The optimal state-feedback gain matrix.
    '''
    def __init__(self, K):
        self.K = np.asarray(K)

    def compute(self, measurement):
        '''Compute the control signal from a state measurement.

        Parameters
        ----------
        measurement : ndarray
            The current state measurement or estimate.

        Returns
        -------
        ndarray
            The control signal.
        '''
        measurement = np.asarray(measurement)

        return -self.K @ measurement
