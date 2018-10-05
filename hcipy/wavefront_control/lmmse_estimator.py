from .observer import Observer
from scipy.linalg import solve_discrete_are
from numpy.linalg import inv

class LMMSE(Observer):
	'''A linear minimum mean square estimator

	This class implements a linear minimum mean square estimator with three different options: a batch process, recursive process,
	and a forgetting process. See https://arxiv.org/pdf/1707.00570.pdf for more details on setup of input data. 

	Parameters
	----------
	t: String
		The type of LMMSE. Can be either 'batch', 'recursive' or 'forgetting'
	Forgetting: scalar
		Default is 1 which corresponds to a pure recusive solution. Set to forget previous data to make robust again slowly varying statistics

	Attributes
	----------
	F : ndarray
		The computed observer matrix -- the filter
	'''
	def __init__(self, Observer, t='batch',forgetting=1):
		if t=='recursive': #recursive mode
			self.type=1
		elif t=='forgetting': #forgetting mode -- need to then also set forgetting factor. 
			#Best when between 0.95 and 0.999
			self.type=2
		else: #batch mode -- default
			self.type=0
		self.forgetting=forgetting
		self.cov_oo_inv=[]
		self.cov_or=[]
		self.F=[]
		self.estimate=[]
	
	@property
	def F(self,m,n,training_data=None, output_data=None):
		'''Updates the estimator for the recursive LMMSE only. If you have a batch implemented - this will not update. 

		Parameters
		----------
		training_data: 1xnm ndarray
			A ndarray containing the input data for the filter to estimate the new data
		output_data: 1x1 ndarray
			A ndarray containing the desired output data allowing the data to train using it. 		
		'''

		if self.type==0:
			try:
				self.F=numpy.linalg.inv(training_data)*output_data
			except:
				print('Need to give training data: No training data provided for batch LMMSE')
		else:
	
			self.cov_oo_inv=numpy.zeros((n,n*m))
			self.cov_or=numpy.ones((n,m))
		 
	def F_update(self,training_data,output_data):
		'''Updates the estimator for the recursive LMMSE only. If you have a batch implemented - this will not update. 

		Parameters
		----------
		training_data: 1xnm ndarray
			A ndarray containing the input data for the filter to estimate the new data
		output_data: 1x1 ndarray
			A ndarray containing the desired output data allowing the data to train using it. 
		
		'''
		try:
			self.type!=0
		except:
			print('Batch LMMSE - you cannot update')
			return None
		top=(self.cov_oo_inv*training_data.T*self.cov_oo_inv)
		bottom=training_data.T*self.cov_oo_inv*training_data
		bottom=1+self.forgetting**(-1)*bottom
		self.cov_oo_inv=self.forgetting**(-1)*self.cov_oo_inv - (top/bottom)      
		self.cov_or=self.forgetting*self.cov_or +output_data*training_data.T             
		self.F=cov_oo_inv.dot(cov_or.T)
	
	def estimater(self,data):
		'''Estimates the new data.

		Parameters
		----------
		data: nxm ndarray
			A ndarray containing the input data for the filter to estimate the new data
		
		'''
		t=self.F*data
		self.estimate=t
		return t 
		   
