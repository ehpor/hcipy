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


def _windows_affinity_count():
    """Process CPU affinity count via Win32 GetProcessAffinityMask, or None on failure."""
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
            # Deal with comma-separated values.
            return int(val.split(",")[0])

    # 3. HPC scheduler allocations
    for env_var in ("SLURM_CPUS_PER_TASK", "NSLOTS"):
        val = os.environ.get(env_var)
        if val is not None:
            # Deal with comma-separated values.
            return int(val.split(",")[0])

    # 4a. Linux: process affinity, intersected with cgroup CPU quota
    #     (a container can show a full affinity mask while throttled
    #     to a fraction of a CPU by cpu.max / cfs_quota_us)
    if hasattr(os, "sched_getaffinity"):
        affinity_count = len(os.sched_getaffinity(0))
        quota = _cgroup_quota_cpus()
        if quota is not None:
            n = math.ceil(quota)
            return max(1, min(affinity_count, n))
        return affinity_count

    # 4b. Windows: Win32 affinity mask
    if platform.system() == "Windows":
        count = _windows_affinity_count()
        if count is not None:
            return count

    # 5. Last resort
    return multiprocessing.cpu_count()
