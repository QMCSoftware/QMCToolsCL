from .util import get_qmcseqcl_program_from_context

import ctypes
import numpy as np 
import time
import glob
import os

def opencl_c_func(func):
    func_name = func.__name__
    def wrapped_func(*args, **kwargs):
        if "backend" in kwargs: 
            assert kwargs["backend"].lower() in ["cl","c"] 
            backend = kwargs["backend"].lower()
        else: 
            backend = "c"
        args = list(args)
        if backend=="c":
            t0_perf = time.perf_counter()
            t0_process = time.process_time()
            args = args[:3]+args[:3]+args[3:] # repeat the first 3 args to the batch sizes
            eval("%s_c(*args)"%func_name)
            tdelta_process = time.process_time()-t0_process 
            tdelta_perf = time.perf_counter()-t0_perf 
            return tdelta_perf,tdelta_process
        else: # backend=="cl"
            t0_perf = time.perf_counter()
            import pyopencl as cl
            if "PYOPENCL_CTX" in kwargs:
                os.environ["PYOPENCL_CTX"] = kwargs["PYOPENCL_CTX"]
            context = kwargs["context"] if "context" in kwargs else cl.create_some_context()
            program = program if "program" in kwargs else get_qmcseqcl_program_from_context(context)
            queue = kwargs["queue"] if "queue" in kwargs else cl.CommandQueue(context,properties=cl.command_queue_properties.PROFILING_ENABLE)
            assert "global_size" in kwargs 
            global_size = kwargs["global_size"]
            local_size = kwargs["local_size"] if "local_size" in kwargs else None
            cl_func = getattr(program,func_name)
            args_device = [cl.Buffer(context,cl.mem_flags.READ_WRITE|cl.mem_flags.COPY_HOST_PTR,hostbuf=arg) if isinstance(arg,np.ndarray) else arg for arg in args]
            batch_size = [np.uint64(np.ceil(args[i]/global_size[i])) for i in range(3)]
            args_device = args_device[:3]+batch_size+args_device[3:] # repeat the first 3 args to the batch sizes
            event = cl_func(queue,global_size,local_size,*args_device)
            if "wait" in kwargs and kwargs["wait"]:
                event.wait()
                tdelta_process = (event.profile.end - event.profile.start)*1e-9
            else:
                tdelta_process = -1
            if isinstance(args[-1],np.ndarray):
                cl.enqueue_copy(queue,args[-1],args_device[-1])
            tdelta_perf = time.perf_counter()-t0_perf
            return tdelta_perf,tdelta_process
    return wrapped_func

c_lib = ctypes.CDLL(glob.glob(os.path.dirname(os.path.abspath(__file__))+"/c_lib*")[0], mode=ctypes.RTLD_GLOBAL)

lattice_linear_c = c_lib.lattice_linear
lattice_linear_c.argtypes = [
    ctypes.c_uint64, # r_x
    ctypes.c_uint64, # n
    ctypes.c_uint64, # d
    ctypes.c_uint64, # batch_size_r_x
    ctypes.c_uint64, # batch_size_n
    ctypes.c_uint64, # batch_size_d
    np.ctypeslib.ndpointer(ctypes.c_uint64,flags='C_CONTIGUOUS'),  # g_d
    np.ctypeslib.ndpointer(ctypes.c_double,flags='C_CONTIGUOUS')]  # x_d

@opencl_c_func
def lattice_linear(): pass
