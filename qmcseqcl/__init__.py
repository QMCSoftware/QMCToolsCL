import ctypes
import numpy as np
import re
import time
import glob
import os

def random_tbit_uint64s(rng, t, shape):
    """Generate the desired shape of random integers with t bits
    
    Args:
        rng (np.random._generator.Generator): random number generator with rng.integers method
        t: (int): number of bits with 0 <= t <= 64
        shape (tuple of ints): shape of resulting integer array"""
    assert 0<=t<=64, "t must be between 0 and 64"
    if t<64: 
        x = rng.integers(0,1<<t,shape,dtype=np.uint64)
    else: # t==64
        x = rng.integers(-(1<<63),1<<63,shape,dtype=np.int64)
        negs = x<0
        x[negs] = x[negs]-(-(1<<63))
        x = x.astype(np.uint64)
        x[~negs] = x[~negs]+((1<<63))
    return x

def get_qmcseqcl_program_from_context(context):
    import pyopencl as cl
    FILEDIR = os.path.dirname(os.path.realpath(__file__))
    with open(FILEDIR+"/qmcseqcl.cl","r") as kernel_file:
        kernelsource = kernel_file.read()
    program = cl.Program(context,kernelsource).build()
    return program

def print_opencl_device_info():
    """
    Copied from https://github.com/HandsOnOpenCL/Exercises-Solutions/blob/master/Exercises/Exercise01/Python/DeviceInfo.py
    """
    import pyopencl as cl
    platforms = cl.get_platforms()
    print("\nNumber of OpenCL platforms:", len(platforms))
    print("\n-------------------------")
    for p in platforms:
        print("Platform:", p.name)
        print("Vendor:", p.vendor)
        print("Version:", p.version)
        devices = p.get_devices()
        for d in devices:
            print("\t-------------------------")
            print("\t\tName:", d.name)
            print("\t\tName:", d.name)
            print("\t\tVersion:", d.opencl_c_version)
            print("\t\tMax. Compute Units:", d.max_compute_units)
            print("\t\tLocal Memory Size:", d.local_mem_size/1024, "KB")
            print("\t\tGlobal Memory Size:", d.global_mem_size/(1024*1024), "MB")
            print("\t\tMax Alloc Size:", d.max_mem_alloc_size/(1024*1024), "MB")
            print("\t\tMax Work-group Total Size:", d.max_work_group_size)
            dim = d.max_work_item_sizes
            print("\t\tMax Work-group Dims:(", dim[0], " ".join(map(str, dim[1:])), ")")
            print("\t-------------------------")
        print("\n-------------------------")

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
            try:
                import pyopencl as cl
            except:
                raise ImportError("install pyopencl to access these capabilities in QMCseqCL")
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
    wrapped_func.__doc__ = func.__doc__
    return wrapped_func

c_lib = ctypes.CDLL(glob.glob(os.path.dirname(os.path.abspath(__file__))+"/c_lib*")[0], mode=ctypes.RTLD_GLOBAL)

c_to_ctypes_map = {
    "ulong": "uint64",
    "double": "double",
    "char": "uint8",
}

with open("./qmcseqcl/qmcseqcl.cl","r") as f:
    code = f.read() 
blocks = re.findall('(?<=void\s).*?(?=\s?\))',code,re.DOTALL)
for block in blocks:
    lines = block.replace("(","").splitlines()
    name = lines[0]
    desc = lines[1].split("// ")[1].strip()
    args = []
    doc_args = []
    for i in range(2,len(lines)):
        input,var_desc = lines[i].split(" // ")
        var_type,var = input.replace(",","").split(" ")[-2:]
        if var_type not in c_to_ctypes_map:
                raise Exception("var_type %s not found in map"%var_type)
        c_var_type = c_to_ctypes_map[var_type]
        if var[0]!="*":
            doc_args += ["%s (np.%s): %s"%(var,c_var_type,var_desc)]
            args += ["ctypes.c_%s"%c_var_type]
        else:
            doc_args += ["%s (np.ndarray of np.%s): %s"%(var[1:],c_var_type,var_desc)]
            args += ["np.ctypeslib.ndpointer(ctypes.c_%s,flags='C_CONTIGUOUS')"%c_var_type]
    doc_args = doc_args[:3]+doc_args[6:] # skip batch size args
    exec("%s_c = c_lib.%s"%(name,name)) 
    exec("%s_c.argtypes = [%s]"%(name,','.join(args)))
    exec('@opencl_c_func\ndef %s():\n    """%s\n\nArgs:\n    %s"""\n    pass'%(name,desc.strip(),"\n    ".join(doc_args)))
