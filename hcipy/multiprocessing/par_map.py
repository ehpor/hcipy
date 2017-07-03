import multiprocessing
import os
import numpy as np

def fun(f, q_in, q_out, reseed):
	if reseed:
		np.random.seed()
	
	while True:
		i, x = q_in.get()
		if i is None:
			break
		q_out.put((i, f(x)))

def par_map(f, X, nprocs=multiprocessing.cpu_count(), reseed=True, use_progressbar=True):
	q_in = multiprocessing.Queue(1)
	q_out = multiprocessing.Queue()

	proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out, reseed)) for _ in range(nprocs)]
	for p in proc:
		p.daemon = True
		p.start()

	if use_progressbar:
		import progressbar
		widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
		pbar = progressbar.ProgressBar(widgets=widgets)
		sent = [q_in.put((i, x)) for i, x in enumerate(pbar(X))]
	else:
		sent = [q_in.put((i, x)) for i, x in enumerate(X)]
	
	[q_in.put((None, None)) for _ in range(nprocs)]
	res = [q_out.get() for _ in range(len(sent))]

	[p.join() for p in proc]

	return [x for i, x in sorted(res)]