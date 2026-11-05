import heapq
import math

class DynamicOpticalSystem(object):
    '''A dynamically varying optical system.

    This can be used as a starting point for an implementation of an adaptive optics system. The optics
    are changed as specific moments in time, which can be scheduled by submitting a callback. The time
    points are bridged using the integrate function, which is written by the derived class. This
    function should integrate on each detector without reading out the detector. Detector readouts or
    DM changes should be implemented as callbacks.
    '''
    def __init__(self):
        self.callbacks = []
        self.t = 0
        self.callback_counter = 0

    def evolve_until(self, t):
        '''Evolve the optical system until time `t`.

        Parameters
        ----------
        t : scalar
            The point in time to which to simulate this optical system. Callbacks and integrations
            are handled along the way, including callbacks at time `t`.

        Raises
        ------
        ValueError
            If `t` is nonfinite or earlier than the current time.

        Notes
        -----
        Every positive interval is integrated, including intervals below one microsecond.
        Callbacks at the same time run in insertion order without intervening integration.
        If integration raises, the pending callback and scheduler time are retained.
        Side effects of a failed integration must be handled by the derived class.
        A callback that raises has already been removed and is not retried automatically.
        '''
        if not math.isfinite(t):
            raise ValueError('Evolution time must be finite.')
        if t < self.t:
            raise ValueError('Backwards evolution is not allowed.')

        end = False
        while not end:
            if self.callbacks:
                t_next, _, callback = self.callbacks[0]
            else:
                # There are no callbacks, so evolve till the end.
                t_next = float('inf')
                callback = None

            if t_next <= t:
                integration_time = t_next - self.t
            else:
                integration_time = t - self.t
                end = True

            # Avoid multiple expensive integrations if we have multiple callbacks
            # at the same time.
            if integration_time > 0:
                self.integrate(integration_time)
            self.t = t if end else t_next

            if not end:
                heapq.heappop(self.callbacks)
                callback()

    def integrate(self, integration_time):
        '''Integrate the current optical system for a certain integration time.

        This function should be implemented by a user in a derived class.

        Parameters
        ----------
        integration_time : scalar
            The integration time with which to integrate the optical system.
            The current time is the start of this interval; the scheduler advances it
            after this method returns successfully. Schedule events in callbacks,
            rather than in this method.
        '''
        pass

    def add_callback(self, t, callback):
        '''Add a callback to the list of callbacks.

        This function can even be called during handling of a callback. This is
        especially useful when implementing periodic callbacks: a callback can
        reinsert itself at a later time at the end of handling of the callback.

        Parameters
        ----------
        t : scalar
            The finite time at which to call the callback, at or after the current time.
        callback : function
            The function to call at time `t`. This function should have no arguments.

        Raises
        ------
        ValueError
            If `t` is nonfinite or earlier than the current time.
        TypeError
            If `callback` is not callable.
        '''
        if not math.isfinite(t):
            raise ValueError('Callback time must be finite.')
        if t < self.t:
            raise ValueError('Callbacks cannot be scheduled in the past.')
        if not callable(callback):
            raise TypeError('The callback must be callable.')

        # Callback counter is to avoid comparison of callback functions
        # if the times are equal, as comparison of functions is not allowed.
        heapq.heappush(self.callbacks, (t, self.callback_counter, callback))
        self.callback_counter += 1
