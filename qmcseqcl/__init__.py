import ctypes
import numpy as np
import time
import glob
import os

qmcgencl_clib = ctypes.CDLL(glob.glob(os.path.dirname(os.path.abspath(__file__))+"/clib*")[0], mode=ctypes.RTLD_GLOBAL)

def get_qmcseqcl_program_from_context(context):
    import pyopencl as cl
    FILEDIR = os.path.dirname(os.path.realpath(__file__))
    with open(FILEDIR+"/qmcseqcl/qmcseqcl.cl","r") as kernel_file:
        kernelsource = kernel_file.read()
    program = cl.Program(context,kernelsource).build()
    return program

def opencl_c_func(func):
    func_name = func.__name__
    def wrapped_func(*args, **kwargs):
        if "backend" in kwargs: 
            assert kwargs["backend"].lower() in ["cl","c"] 
            backend = kwargs["backend"].lower()
        else: 
            backend = "c"
        if backend=="c":
            t0_perf = time.perf_counter()
            t0_process = time.process_time()
            eval("%s_c(*args)"%func_name)
            tdelta_process = time.process_time()-t0_process 
            tdelta_perf = time.perf_counter()-t0_perf 
            return tdelta_perf,tdelta_process
        else: # backend=="cl"
            t0_perf = time.perf_counter()
            import pyopencl as cl
            context = kwargs["context"] if "context" in kwargs else cl.create_some_context()
            program = program if "program" in kwargs else get_qmcseqcl_program_from_context(context)
            queue = kwargs["queue"] if "queue" in kwargs else cl.CommandQueue(context,properties=cl.command_queue_properties.PROFILING_ENABLE)
            assert "global_size" in kwargs 
            global_size = kwargs["global_size"]
            local_size = kwargs["local_size"] if "local_size" in kwargs else None
            cl_func = getattr(program,func_name)
            args_device = (cl.Buffer(context,cl.mem_flags.READ_WRITE|cl.mem_flags.COPY_HOST_PTR,hostbuf=arg) if isinstance(arg,np.ndarray) else arg for arg in args)
            event = cl_func(queue,global_size,local_size,*args_device)
            tdelta_process = (event.profile.end - event.profile.start)*1e-9
            if isinstance(args[-1],np.ndarray):
                cl.enqueue_copy(queue,args[-1],args_device[-1])
            tdelta_perf = time.perf_counter()-t0_perf
            return tdelta_perf,tdelta_process
