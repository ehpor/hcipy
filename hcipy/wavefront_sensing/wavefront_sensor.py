class WavefrontSensor(object):
	def __init__(self, detector):
		self.detector = detector
		self.optical_system = self.make_optical_system()

	# Implement this in derived classes
	def make_optical_system(self):
		raise NotImplementedError()

	def measurement(self):
		raise NotImplementedError()

	def calibrate(self):
		raise NotImplementedError()

	def integrate(self, wavefront, dt=1, weight=1):
		wavefront_out = self.optical_system.forward( wavefront )
		self.detector.integrate( wavefront_out, dt, weight )

	def read_out(self):
		return self.detector.read_out()

	def __call__(self, wavefront, dt=1, weight=1):
		self.integrate(wavefront, dt, weight)
		return self.measurement( self.read_out() )