import ctypes as ct
import numpy as np

libtest2d = ct.cdll.LoadLibrary("./libtest2d.so")
libtest2d.print_2d_list.argtypes = (ct.c_ulong, ct.c_ulong,
        np.ctypeslib.ndpointer(dtype=np.float64,
            ndim=2,
            flags='C_CONTIGUOUS'
            )
        )
libtest2d.print_2d_list.restype = None

arr2d = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 11))[0]

libtest2d.print_2d_list(arr2d.shape[0], arr2d.shape[1], arr2d)