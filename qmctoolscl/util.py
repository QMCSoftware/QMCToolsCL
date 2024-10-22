import numpy as np
import time
import os
import ctypes 
import glob 
import os

c_lib = ctypes.CDLL(glob.glob(os.path.dirname(os.path.abspath(__file__))+"/c_lib*")[0], mode=ctypes.RTLD_GLOBAL)

from .c_funcs import *

def print_opencl_device_info():
    """ Print OpenCL devices info. Copied from https://github.com/HandsOnOpenCL/Exercises-Solutions/blob/master/Exercises/Exercise01/Python/DeviceInfo.py """
    import pyopencl as cl
    platforms = cl.get_platforms()
    for i,p in enumerate(platforms):
        print("Platform %d -------------------------"%i)
        print("\tName:",p.name)
        print("\tVendor:", p.vendor)
        print("\tVersion:", p.version)
        devices = p.get_devices()
        for j,d in enumerate(devices):
            print("\n\tDevice %d -------------------------"%j)
            print("\t\tName:", d.name)
            print("\t\tVersion:", d.opencl_c_version)
            print("\t\tMax. Compute Units:", d.max_compute_units)
            print("\t\tLocal Memory Size:", d.local_mem_size/1024, "KB")
            print("\t\tGlobal Memory Size:", d.global_mem_size/(1024*1024), "MB")
            print("\t\tMax Alloc Size:", d.max_mem_alloc_size/(1024*1024), "MB")
            print("\t\tMax Work-group Total Size:", d.max_work_group_size)
            dim = d.max_work_item_sizes
            print("\t\tMax Work-group Dims:(", dim[0], " ".join(map(str, dim[1:])), ")")
        print()

bs_plugin_indices = {
    #"lat_gen_gray": [3,5,3,4,5],
    #"dnb2_gen_gray": [3,5,7],
}

def get_qmctoolscl_program_from_context(context, func_name, args_device):
    import pyopencl as cl
    FILEDIR = os.path.dirname(os.path.realpath(__file__))
    with open(FILEDIR+"/cl_kernels/%s.cl"%func_name,"r") as kernel_file:
        kernelsource = kernel_file.read()
    if func_name in bs_plugin_indices:
        insert_ints = tuple([int(args_device[i]) for i in bs_plugin_indices[func_name]])
        kernelsource = kernelsource%insert_ints
    program = cl.Program(context,kernelsource).build()
    return program

def _parse_kwargs_backend_queue_program(kwargs):
    if "backend" in kwargs: 
        kwargs["backend"] = kwargs["backend"].lower()
        assert kwargs["backend"] in ["cl","c"] 
    else: 
        kwargs["backend"] = "c"
    if kwargs["backend"]=="cl":
        try:
            import pyopencl as cl
        except:
            raise ImportError("install pyopencl to access these capabilities in qmctoolscl")
        if "context" not in kwargs:
            platform = cl.get_platforms()[kwargs["platform_id"] if "platform_id" in kwargs else 0]
            device = platform.get_devices()[kwargs["device_id"] if "device_id" in kwargs else 0]
            kwargs["context"] = cl.Context([device])
        if "queue" not in kwargs:
            if "profile" in kwargs and kwargs["profile"]:
                kwargs["queue"] = cl.CommandQueue(kwargs["context"],properties=cl.command_queue_properties.PROFILING_ENABLE)
            else:
                kwargs["queue"] = cl.CommandQueue(kwargs["context"])

def _preprocess_fft_bro_1d_radix2(*args_device,kwargs):
    if kwargs["backend"]=="cl" and (kwargs["local_size"] is None or kwargs["local_size"][2]!=kwargs["global_size"][2]):
        raise Exception("fft_bro_1d_radix2 requires local_size is not None and local_size[2] = %d equals global_size[2] = %d"%(kwargs["local_size"][2],kwargs["global_size"][2]))

def _preprocess_ifft_bro_1d_radix2(*args_device,kwargs):
    if kwargs["backend"]=="cl" and (kwargs["local_size"] is None or kwargs["local_size"][2]!=kwargs["global_size"][2]):
        raise Exception("fft_bro_1d_radix2 requires local_size is not None and local_size[2] = %d equals global_size[2] = %d"%(kwargs["local_size"][2],kwargs["global_size"][2]))

def _preprocess_fwht_1d_radix2(*args_device,kwargs):
    if kwargs["backend"]=="cl" and (kwargs["local_size"] is None or kwargs["local_size"][2]!=kwargs["global_size"][2]):
        raise Exception("fwht_1d_radix2 requires local_size is not None and local_size[2] = %d equals global_size[2] = %d"%(kwargs["local_size"][2],kwargs["global_size"][2]))

def _preprocess_lat_gen_natural(r,n,d,bs_r,bs_n,bs_d,n_start,g,x,kwargs):
    if not ((n_start==0 or np.log2(n_start)%1==0) and ((n+n_start)==0 or np.log2(n+n_start)%1==0)):
        raise Exception("lat_gen_natural requires n_start and n+n_start are either 0 or powers of 2")

_overwrite_args = {
    "fft_bro_1d_radix2": 2, 
    "ifft_bro_1d_radix2": 2, 
}

def parse_gs_bs(gs, args3):
    gs = [min(gs[i],args3[i]) for i in range(3)]
    bs = [np.uint64(np.ceil(args3[i]/gs[i])) for i in range(3)]
    gs = [np.uint64(np.ceil(args3[i]/bs[i])) for i in range(3)]
    return gs,bs

def _opencl_c_func(func):
    func_name = func.__name__
    def wrapped_func(*args, **kwargs):
        _parse_kwargs_backend_queue_program(kwargs)
        args = list(args)
        if kwargs["backend"]=="c":
            t0_perf = time.perf_counter()
            t0_process = time.process_time()
            args = args[:3]+args[:3]+args[3:] # repeat the first 3 args to the batch sizes
            try:
                eval('_preprocess_%s(*args,kwargs=kwargs)'%func_name)
            except NameError: pass
            eval("%s_c(*args)"%func_name)
            tdelta_process = time.process_time()-t0_process 
            tdelta_perf = time.perf_counter()-t0_perf 
            return tdelta_perf,tdelta_process
        else: # kwargs["backend"]=="cl"
            import pyopencl as cl
            t0_perf = time.perf_counter()
            assert "global_size" in kwargs 
            kwargs["global_size"],bs = parse_gs_bs(kwargs["global_size"],args)
            if "local_size" not in kwargs:
                kwargs["local_size"] = None
            args_device = [cl.Buffer(kwargs["context"],cl.mem_flags.READ_WRITE|cl.mem_flags.COPY_HOST_PTR,hostbuf=arg) if isinstance(arg,np.ndarray) else arg for arg in args]
            args_device = args_device[:3]+bs+args_device[3:] # repeat the first 3 args to the batch sizes
            if "program" not in kwargs:
                kwargs["program"] =  get_qmctoolscl_program_from_context(kwargs["context"],func_name,args_device)
            cl_func = getattr(kwargs["program"],func_name)
            try:
                eval('_preprocess_%s(*args_device,kwargs=kwargs)'%func_name)
            except NameError: pass
            t0_process = time.perf_counter()
            event = cl_func(kwargs["queue"],kwargs["global_size"],kwargs["local_size"],*args_device)
            if "wait" not in kwargs or kwargs["wait"]:
                event.wait()
                tdelta_process = time.perf_counter()-t0_process
            else:
                tdelta_process = -1
            if isinstance(args[-1],np.ndarray):
                num_overwrite_args = _overwrite_args[func_name] if func_name in _overwrite_args else 1
                for i in range(-1,-1-num_overwrite_args,-1):
                    cl.enqueue_copy(kwargs["queue"],args[i],args_device[i])
            tdelta_perf = time.perf_counter()-t0_perf
            return tdelta_perf,tdelta_process
    wrapped_func.__doc__ = func.__doc__
    return wrapped_func