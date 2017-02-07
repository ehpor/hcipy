import heapq
import numpy as np

class DynamicOpticalSystem(object):
	def __init__(self):
		self.callbacks = []
		self.t = 0
		self.callback_counter = 0
	
	def evolve_until(self, t):
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
		pass
	
	def add_callback(self, t, callback):
		# Callback counter is to avoid comparison of callback functions
		# (which you can't), if the times are equal.
		heapq.heappush(self.callbacks, (t, self.callback_counter, callback))
		self.callback_counter += 1