from ..optics import OpticalSystem

class WavefrontSensorOptics(OpticalSystem):
	'''The optics for a wavefront sensor.

	This object serves as the base class for all optics in a wavefront sensor.
	'''
	pass

class WavefrontSensorEstimator(object):
	'''The estimator for a wavefront sensor.

	This object serves as the base class for all estimators in a wavefront sensor.
	It calculates from a list of images what the wavefront is. This wavefront can be
	anything that responds to wavefront changes, such as slopes, pupil amplitude, etc...
	'''
	def estimate(self, images):
		'''Estimate the wavefront from `images`.

		Parameters
		----------
		images : list
			The list of images on which to base the estimate.
		
		Returns
		-------
		Field or ndarray
			The estimate of the wavefront.
		'''
		raise NotImplementedError()

class WavefrontSensor(object):
	'''A complete wavefront sensor.

	This convenience class consolidates the optics, detector, frame correction 
	and estimation parts of a wavefront sensor.

	.. caution::
		This class only works with estimators that require only one image.
		For estimators with more than one required image, the wavefront
		sensor must be handcoded.

	Parameters
	----------
	wfs_optics : WavefrontSensorOptics object
		The optics of the wavefront sensor. If it is not supplied, no optics 
		will be used.
	detector : Detector object
		The detector of the wavefront sensor. If it is not supplied, a NoiselessDetector
		will be used.
	frame_correction : FrameCorrector object
		The frame correction of the detector. If it is not supplied, no frame
		correction will be used.
	wfs_estimator : WavefrontEstimator
		The wavefront estimator. If it is not supplied, the corrected image will
		be returned instead.
	'''
	def __init__(self, wfs_optics=None, detector=None, frame_corrector=None, wfs_estimator=None):
		self.wfs_optics = wfs_optics
		self.detector = detector
		self.frame_corrector = frame_corrector
		self.wfs_estimator = wfs_estimator

	@property
	def detector(self):
		'''The used detector.
		'''
		return self._detector
	
	@detector.setter
	def detector(self, detector):
		if detector is None:
			self._detector = NoiselessDetector()
		else:
			self._detector = detector
	
	@property
	def wfs_optics(self):
		'''The used wavefront sensor optics.
		'''
		return self._wfs_optics
	
	@wfs_optics.setter
	def wfs_optics(self, wfs_optics):
		if wfs_optics is None:
			self._wfs_optics = EmptyOpticalElement()
		else:
			self._wfs_optics = wfs_optics
		
	@property
	def frame_corrector(self):
		'''The used frame corrector.
		'''
		return self._frame_corrector
	
	@frame_corrector.setter
	def frame_corrector(self, frame_corrector):
		if frame_corrector is None:
			self._frame_corrector = FrameCorrector()
		else:
			self._frame_corrector = frame_corrector

	def integrate(self, wavefront, dt, weight=1):
		'''Integrate the wavefront sensor.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront will be propagated through the wavefront sensor optics 
			and fall onto the detector.
		dt : scalar
			The integration time in units of time.
		weight : scalar
			Weight of every unit of integration time.
		'''
		wf = self.wfs_optics.forward(wavefront)
		self.detector.integrate(wf, dt, weight)
	
	def read_out(self):
		'''Estimate the wavefront from the read-out detector image.

		If there is no wavefront estimator avaiable, the corrected detector
		image will be returned instead.

		Returns
		-------
		Field or ndarray
			The estimate of the wavefront sensor. If the wavefront sensor estimator
			is not give, the corrected detector image will be given instead.
		'''
		img = self.detector.read_out()
		corrected_img = self.frame_corrector.correct(img)

		if self.wfs_estimator is None:
			return corrected_img

		wfs_response = self.wfs_estimator.estimate([corrected_img])
		return wfs_response

	def __call__(self, wavefront, dt=1, weight=1):
		'''Convenience function to integrate and read out the wavefront sensor.

		Parameters
		----------
		wavefront : Wavefront
			The wavefront will be propagated through the wavefront sensor optics 
			and fall onto the detector.
		dt : scalar
			The integration time in units of time.
		weight : scalar
			Weight of every unit of integration time.
		
		Returns
		-------
		Field or ndarray
			The estimate of the wavefront sensor. If the wavefront sensor estimator
			is not give, the corrected detector image will be given instead.
		'''
		self.integrate(wavefront, dt, weight)
		return self.read_out()
