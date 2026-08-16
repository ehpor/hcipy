import math
import os
import platform
import multiprocessing
import threadpoolctl


def _cgroup_v2_quota():
    """cgroup v2 unified quota (cpu.max), fractional CPUs or None if unlimited/unavailable."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota == "max":
            return None
        return int(quota) / int(period)
    except (FileNotFoundError, ValueError):
        return None


def _cgroup_v1_quota():
    """cgroup v1 quota (cfs_quota_us / cfs_period_us), fractional CPUs or None."""
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        if quota <= 0:  # -1 means unlimited
            return None
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read().strip())
        return quota / period
    except (FileNotFoundError, ValueError):
        return None


def _cgroup_quota_cpus():
    return _cgroup_v2_quota() or _cgroup_v1_quota()


def _blas_threadpool_count():
    """Actual thread count BLAS is using right now, or None if no BLAS info is available.
    """
    pools = threadpoolctl.threadpool_info()
    if not pools:
        return None
    return min(p["num_threads"] for p in pools)


def _blas_env_override_count():
    """BLAS/threading environment variable override, or None if none is set.
    """
    BLAS_THREAD_ENV_VARS = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]

    for env_var in BLAS_THREAD_ENV_VARS:
        val = os.environ.get(env_var)
        if val is not None:
            # Deal with comma-separated values.
            return int(val.split(",")[0])
    return None


def _scheduler_env_override_count():
    """HPC scheduler allocation from the environment, or None if none is set.
    """
    _SCHEDULER_ENV_VARS = [
        "SLURM_CPUS_PER_TASK",
        "NSLOTS",
    ]

    for env_var in _SCHEDULER_ENV_VARS:
        val = os.environ.get(env_var)
        if val is not None:
            # Deal with comma-separated values.
            return int(val.split(",")[0])
    return None


def _linux_affinity_count():
    """Process CPU affinity, intersected with cgroup CPU quota, or None if unavailable.
    """
    if not hasattr(os, "sched_getaffinity"):
        return None

    affinity_count = len(os.sched_getaffinity(0))
    quota = _cgroup_quota_cpus()

    if quota is not None:
        n = math.ceil(quota)
        return max(1, min(affinity_count, n))

    return affinity_count


def _windows_affinity_count():
    """Process CPU affinity count via Win32 GetProcessAffinityMask, or None on failure.
    """
    if platform.system() != "Windows":
        return None

    import ctypes
    import ctypes.wintypes

    kernel32 = ctypes.WinDLL("kernel32")

    DWORD_PTR = ctypes.wintypes.WPARAM
    PDWORD_PTR = ctypes.POINTER(DWORD_PTR)

    GetCurrentProcess = kernel32.GetCurrentProcess
    GetCurrentProcess.restype = ctypes.wintypes.HANDLE

    GetProcessAffinityMask = kernel32.GetProcessAffinityMask
    GetProcessAffinityMask.argtypes = (ctypes.wintypes.HANDLE, PDWORD_PTR, PDWORD_PTR)

    process_mask = DWORD_PTR()
    system_mask = DWORD_PTR()  # named, not a throwaway temporary

    ok = GetProcessAffinityMask(
        GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)
    )
    if not ok or process_mask.value == 0:
        return None
    return bin(process_mask.value).count("1")


def get_num_available_cores():
    '''Get the number of cores available to the current process.

    Checks, in order of priority:
    1. Actual BLAS thread count via threadpoolctl (if available and loaded)
    2. BLAS/threading environment variables (OMP_NUM_THREADS, etc.)
    3. HPC scheduler allocations (SLURM, SGE)
    4. Process CPU affinity, intersected with cgroup CPU-bandwidth quota
       where applicable (Linux); Win32 affinity mask (Windows)
    5. Total logical CPU count as a last resort

    Returns
    -------
    int
        The number of available cores, always >= 1.
    '''
    funcs = [
        _blas_threadpool_count,
        _blas_env_override_count,
        _scheduler_env_override_count,
        _linux_affinity_count,
        _windows_affinity_count,
    ]

    for func in funcs:
        count = func()

        if count is not None:
            return count

    # Fallback.
    return multiprocessing.cpu_count()
