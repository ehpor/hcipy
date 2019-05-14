import heapq
import numpy as np

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
			are handled along the way.
		'''
		if t < self.t:
			raise ValueError('Backwards evolution is not allowed.')
		
		end = False
		while not end:
			t_next, _, callback = self.callbacks[0]
			if t_next < t:
				integration_time = t_next - self.t
				heapq.heappop(self.callbacks)
			else:
				integration_time = t - self.t
				end = True
			
			# Avoid multiple expensive integrations if we have multiple callbacks
			# at the same time.
			if integration_time > 1e-6:
				self.integrate(integration_time)
				self.t += integration_time
			
			if not end:
				callback()
	
	def integrate(self, integration_time):
		'''Integrate the current optical system for a certain integration time.

		This function should be implemented by a user in a derived class.

		Parameters
		----------
		integration_time : scalar
			The integration time with which to integrate the optical system.
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
			The time at which to call the callback.
		callback : function
			The function to call at time `t`. This function should have no arguments.
		'''
		# Callback counter is to avoid comparison of callback functions
		# if the times are equal, as comparison of functions is not allowed.
		heapq.heappush(self.callbacks, (t, self.callback_counter, callback))
		self.callback_counter += 1