import ctypes
import ctypes.wintypes
import os
import platform

def get_num_available_cores():
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    elif platform.system() == 'Windows':
        kernel32 = ctypes.WinDLL('kernel32')

        DWORD_PTR = ctypes.wintypes.WPARAM
        PDWORD_PTR = ctypes.POINTER(DWORD_PTR)

        GetCurrentProcess = kernel32.GetCurrentProcess
        GetCurrentProcess.restype = ctypes.wintypes.HANDLE

        GetProcessAffinityMask = kernel32.GetProcessAffinityMask
        GetProcessAffinityMask.argtypes = (ctypes.wintypes.HANDLE, PDWORD_PTR, PDWORD_PTR)

        mask = DWORD_PTR()

        if not GetProcessAffinityMask(GetCurrentProcess(), ctypes.byref(mask), ctypes.byref(DWORD_PTR())):
            raise RuntimeError("Call to 'GetProcessAffinityMask' failed")

        return bin(mask.value).count('1')
    else:
        raise RuntimeError('Cannot determine the number of available cores')
