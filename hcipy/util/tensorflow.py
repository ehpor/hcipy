try:
	from contextlib import nullcontext
except ImportError:
	from contextlib import contextmanager

	@contextmanager
	def nullcontext(enter_result=None):
		yield enter_result

def tf_name_scope(field, name):
	if field.backend == 'tensorflow':
		import tensorflow as tf
		context = tf.name_scope('VortexCoronagraph.forward')
	else:
		context = nullcontext()

	return context
