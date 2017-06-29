#!/bin/env python
"""Provides a map() interface to mpi4py.
License: MIT
Copywrite (c) 2012 Thomas Wiecki

Usage
*****

Create a python file (e.g. mpi_square.py):

from mpi4py_map import map
print map(lambda x, y: x**y, [1,2,3,4], 2)

Then call the function with mpirun, e.g.:
mpirun -n 4 mpi_square.py
"""

import sys

def mpi_map(function, sequence, *args, **kwargs):
    """Return a list of the results of applying the function in
    parallel (using mpi4py) to each element in sequence.

    :Arguments:
        function : python function
            Function to be called that takes as first argument an element of sequence.
        sequence : list
            Sequence of elements supplied to function.

    :Optional:
        args : tuple
            Additional constant arguments supplied to function.
        debug : bool=False
            Be very verbose (for debugging purposes).

    """

    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        # Controller
        result = _mpi_controller(sequence, *args, **kwargs)
    else:
        # Worker
        _mpi_worker(function, sequence, *args, **kwargs)
        result = None
    
    MPI.COMM_WORLD.Barrier()
    return result

def _mpi_controller(sequence, *args, **kwargs):
    """Controller function that sends each element in sequence to
    different workers. Handles queueing and job termination.

    :Arguments:
        sequence : list
            Sequence of elements supplied to the workers.

    :Optional:
        args : tuple
            Additional constant arguments supplied to function.
        debug : bool=False
            Be very verbose (for debugging purposes).

    """

    from mpi4py import MPI

    debug = 'debug' in kwargs
    if debug:
        del kwargs['debug']

    rank = MPI.COMM_WORLD.Get_rank()
    assert rank == 0, "rank has to be 0."
    proc_name = MPI.Get_processor_name()
    status = MPI.Status()

    process_list = list(range(1, MPI.COMM_WORLD.Get_size()))
    workers_done = []
    results = {}
    if debug: print("Data:", sequence)

    # Instead of distributing the actual elements, we just distribute
    # the index as the workers already have the sequence. This allows
    # objects to be used as well.
    queue = iter(range(len(sequence)))

    if debug: print("Controller %i on %s: ready!" % (rank, proc_name))

    # Feed all queued jobs to the childs
    while(True):
        status = MPI.Status()
        # Receive input from workers.
        recv = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        if debug: print("Controller: received tag %i from %s" % (status.tag, status.source))

        if status.tag == 1 or status.tag == 10:
            # tag 1 codes for initialization.
            # tag 10 codes for requesting more data.
            if status.tag == 10: # data received
                if debug: print("Controller: Job %i completed by %i" % (recv[0], status.source))
                results[recv[0]] = recv[1] # save back

            # Get next item and send to worker
            try:
                task = next(queue)
                # Send job to worker
                if debug: print("Controller: Sending task to %i" % status.source)
                MPI.COMM_WORLD.send(task, dest=status.source, tag=10)

            except StopIteration:
                # Send kill signal
                if debug:
                    print("Controller: Task queue is empty")
                workers_done.append(status.source)
                MPI.COMM_WORLD.send([], dest=status.source, tag=2)

                # Task queue is empty
                if len(process_list) == 0:
                    break

        # Tag 2 codes for a worker exiting.
        elif status.tag == 2:
            if recv != []:
                # Worker seems to have crashed but was nice enough to
                # send us the item number he has been working on
                results[recv[0]] = recv[1] # save back

            if debug: print('Process %i exited, removing.' % status.source)
            process_list.remove(status.source)
            if debug: print('Processes left over: ' + str(process_list))
            # Task queue is empty
            if len(process_list) == 0:
                break

        else:
            print('Unknown tag %i with msg %s' % (status.tag, str(recv)))

    if len(process_list) == 0:
        if debug: print("All workers done.")
        sorted_results = [results[i] for i in range(len(sequence))]
        if debug: print(sorted_results)
        return sorted_results
    else:
        raise IOError("Something went wrong, workers still active")
        print(process_list)
        return False

def _mpi_worker(function, sequence, *args, **kwargs):
    """Worker that applies function to each element it receives from
    the controller.

    :Arguments:
        function : python function
            Function to be called that takes as first argument an
            element received from the controller.

    :Optional:
        args : tuple
            Additional constant arguments supplied to function.
        debug : bool=False
            Be very verbose (for debugging purposes).

    """

    from mpi4py import MPI

    debug = 'debug' in kwargs
    if debug:
        del kwargs['debug']

    rank = MPI.COMM_WORLD.Get_rank()
    assert rank != 0, "rank is 0 which is reserved for the controller."
    proc_name = MPI.Get_processor_name()
    status = MPI.Status()
    if debug: print("Worker %i on %s: ready!" % (rank, proc_name))

    # Send ready signal
    MPI.COMM_WORLD.send([{'rank': rank, 'name':proc_name}], dest=0, tag=1)

    # Start main data loop
    while True:
        # Wait for element
        if debug: print("Worker %i on %s: waiting for data" % (rank, proc_name))
        recv = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if debug: print("Worker %i on %s: received data, tag: %i" % (rank, proc_name, status.tag))

        if status.tag == 2:
            # Kill signal received
            if debug: print("Worker %i on %s: recieved kill signal" % (rank, proc_name))
            MPI.COMM_WORLD.send([], dest=0, tag=2)
            break

        if status.tag == 10:
            # Call function on received element
            if debug: print("Worker %i on %s: Calling function %s with %s" % (rank, proc_name, function.__name__, recv))

            try:
                result = function(sequence[recv], *args, **kwargs)
            except:
                # Send to master that we are quitting
                MPI.COMM_WORLD.send((recv, None), dest=0, tag=2)
                # Reraise exception
                raise

            if debug: print("Worker %i on %s: finished job %i" % (rank, proc_name, recv))
            # Return sequence number and result to controller
            MPI.COMM_WORLD.send((recv, result), dest=0, tag=10)
    
    if debug: print('Worker %i on %s: exited.' % (rank, proc_name))