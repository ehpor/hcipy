import multiprocessing
import numpy as np

def fun(f, q_in, res, reduce, lock, reseed):
	if reseed:
		np.random.seed()
	
	while True:
		i, x = q_in.get()
		if i is None:
			break
		
		y = f(x)

		lock.acquire()
		res[0] = reduce(res[0], y)
		lock.release()

def reduce_add(a, b):
	if a is None:
		return b
	return a + b

def reduce_multiply(a, b):
	if a is None:
		return b
	return a * b

def par_map_reduce(f_map, X, f_reduce=reduce_add, res_start=None, nprocs=multiprocessing.cpu_count(), reseed=True, use_progressbar=True):
	q_in = multiprocessing.Queue(1)

	# Make lock to atomize access to the result variable.
	lock = multiprocessing.Lock()

	# Make a shared result variable.
	manager = multiprocessing.Manager()
	res = manager.list([res_start])

	# Start processes.
	proc = [multiprocessing.Process(target=fun, args=(f_map, q_in, res, f_reduce, lock, reseed)) for _ in range(nprocs)]
	for p in proc:
		p.daemon = True
		p.start()
	
	# Send data.
	if use_progressbar:
		import progressbar
		widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
		pbar = progressbar.ProgressBar(widgets=widgets)
		sent = [q_in.put((i, x)) for i, x in enumerate(pbar(X))]
	else:
		sent = [q_in.put((i, x)) for i, x in enumerate(X)]
	
	# Stop processes.
	[q_in.put((None, None)) for _ in range(nprocs)]
	[p.join() for p in proc]

	return res[0]