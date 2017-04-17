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

from ..optics import Detector, OpticalSystem

class WavefrontSensorNew(Detector):
	pass

class PerfectWavefrontSensor(WavefrontSensorNew):
	def __init__(self):
		self.phase = 0
		self.integration_time = 0
	
	def integrate(self, wavefront, dt, weight=1):
		self.phase += wavefront.phase * dt * weight
		self.integration_time = dt * weight
	
	def read_out(self):
		return self.phase / self.integration_time

class WavefrontSensorEstimator(object):
	def reconstruct(self, images):
		raise NotImplementedError()

class PerfectPhaseSensor(OpticalSystem):
	def forward(self, wavefront):
		wf = wavefront.copy()
		return wf.phase
	
	def backward(self, wavefront):
		wf = wavefront.copy()
		return -wf.phase

class PerfectWavefrontSensorNew(OpticalSystem):
	def forward(self, wavefront):
		wf = wavefront.copy()
		return wf.phase / wf.wavenumber
	
	def backward(self, wavefront):
		wf = wavefront.copy()
		return -wf.phase / wf.wavenumber