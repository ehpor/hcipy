from ..optics import OpticalSystem

class WavefrontSensorOptics(OpticalSystem):
	pass

class WavefrontSensorEstimator(object):
	def estimate(self, images):
		raise NotImplementedError()