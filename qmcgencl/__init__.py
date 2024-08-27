import ctypes
import glob
import os

qmcgencl_clib = ctypes.CDLL(glob.glob(os.path.dirname(os.path.abspath(__file__))+"/clib*")[0], mode=ctypes.RTLD_GLOBAL)
