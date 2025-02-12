import os
import platform
import multiprocessing

def get_num_available_cores():
    '''Get the number of cores available on the system.

    The attempt at retrieving the number of cores that our process
    is assigned may not be available on all operating systems. On
    unsupported operating systems, the total number of cores is
    returned instead.

    Returns
    -------
    int
        The number of available cores.

    Raises
    ------
    RuntimeError
        If we are unable to
    '''
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
            # Call successful. Return result.
            return bin(mask.value).count('1')

    # Operating system is unsupported or the call to retrieve the
    # core count has failed. Fall back to all cores.
    return multiprocessing.cpu_count()
