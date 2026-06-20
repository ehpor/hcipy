import os
import platform
import multiprocessing
import threadpoolctl

def get_num_available_cores():
    '''Get the number of cores available to the current process.

    Checks, in order of priority:
    1. Actual BLAS thread count via threadpoolctl (if available and loaded)
    2. BLAS/threading environment variables (OMP_NUM_THREADS, etc.)
    3. HPC scheduler allocations (SLURM, SGE)
    4. Process CPU affinity (Linux via sched_getaffinity, Windows via Win32 API)
    5. Total logical CPU count as a last resort

    Returns
    -------
    int
        The number of available cores.

    '''
    # 1. Ground truth — what BLAS is actually using right now
    pools = threadpoolctl.threadpool_info()
    if pools:
        return min(p["num_threads"] for p in pools)

    # 2. Explicit BLAS/threading env var overrides
    env_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]

    for env_var in env_vars:
        val = os.environ.get(env_var)
        if val is not None:
            return int(val)

    # 3. HPC scheduler allocations
    for env_var in ("SLURM_CPUS_PER_TASK", "NSLOTS"):
        val = os.environ.get(env_var)
        if val is not None:
            return int(val)

    # 4. Process affinity — respects taskset, cgroups, containers, etc.
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    elif platform.system() == 'Windows':
        import ctypes
        import ctypes.wintypes

        kernel32 = ctypes.WinDLL('kernel32')

        DWORD_PTR = ctypes.wintypes.WPARAM
        PDWORD_PTR = ctypes.POINTER(DWORD_PTR)

        GetCurrentProcess = kernel32.GetCurrentProcess
        GetCurrentProcess.restype = ctypes.wintypes.HANDLE

        GetProcessAffinityMask = kernel32.GetProcessAffinityMask
        GetProcessAffinityMask.argtypes = (ctypes.wintypes.HANDLE, PDWORD_PTR, PDWORD_PTR)

        mask = DWORD_PTR()

        if GetProcessAffinityMask(GetCurrentProcess(), ctypes.byref(mask), ctypes.byref(DWORD_PTR())):
            return bin(mask.value).count('1')

    # 5. Last resort
    return multiprocessing.cpu_count()
