import ctypes
import numpy as np
import re
import time
import glob
import os

def print_opencl_device_info():
    """ Print OpenCL devices info. Copied from https://github.com/HandsOnOpenCL/Exercises-Solutions/blob/master/Exercises/Exercise01/Python/DeviceInfo.py """
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
            if "context" in kwargs:
                context = kwargs["context"]
            else:
                platform = cl.get_platforms()[0]
                device = platform.get_devices()[0]
                context = cl.Context([device])
            program = program if "program" in kwargs else get_qmcseqcl_program_from_context(context)
            queue = kwargs["queue"] if "queue" in kwargs else cl.CommandQueue(context,properties=cl.command_queue_properties.PROFILING_ENABLE)
            assert "global_size" in kwargs 
            global_size = kwargs["global_size"]
            global_size = [min(global_size[i],args[i]) for i in range(3)]
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
blocks = re.findall(r'(?<=void\s).*?(?=\s?\))',code,re.DOTALL)
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

class NUSNodeB2(object):
    def __init__(self, shift_bits=None, xb=None, left=None, right=None):
        self.shift_bits = shift_bits
        self.xb = xb 
        self.left = left 
        self.right = right

def nested_uniform_scramble_digital_net_b2(
    r,
    n, 
    d,
    r_x,
    tmax,
    tmax_new,
    rngs,
    root_nodes,
    xb,
    xrb):
    """ Nested uniform scramble of digital net b2
    
    Args: 
        r (np.uint64): replications 
        n (np.uint64): points
        d (np.uint64): dimensions
        r_x (np.uint64): replications of x
        tmax (np.uint64): maximum number of bits in each integer
        tmax_new (np.uint64): maximum number of bits in each integer after scrambling
        rngs (np.ndarray of numpy.random._generator.Generator): random number generators of size r*d
        root_nodes (np.ndarray of NUSNodeB2): root nodes of size r*d
        xb (np.ndarray of np.uint64): array of unrandomized points of size r*n*d
        xrb (np.ndarray of np.uint64): array to store scrambled points of size r*n*d
    """
    for l in range(r): 
        for j in range(d):
            rng = rngs[l,j]
            root_node = root_nodes[l,j]
            if root_node.shift_bits is None:
                # initilize root nodes 
                assert root_node.xb is None and root_node.left is None and root_node.right is None
                root_node.xb = np.uint64(0) 
                root_node.shift_bits = random_tbit_uint64s(rng,tmax_new,1)[0]
            for i in range(n):
                _xb_new = xb[l%r_x,i,j]<<(tmax_new-tmax)
                _xb = _xb_new
                node = root_nodes[l,j]
                t = tmax_new
                shift = np.uint64(0)                 
                while t>0:
                    b = (_xb>>(t-1))&1 # leading bit of _xb
                    ones_mask_tm1 = (2**(t-1)-1)
                    _xb_next = _xb&ones_mask_tm1 # drop the leading bit of _xb 
                    if node.xb is None: # this is not a leaf node, so node.shift_bits in [0,1]
                        if node.shift_bits: shift += 2**(t-1) # add node.shift_bits to the shift
                        if b==0: # looking to move left
                            if node.left is None: # left node does not exist
                                shift_bits = int(rng.integers(0,2**(t-1))) # get (t-1) random bits
                                node.left = NUSNodeB2(shift_bits,_xb_next,None,None) # create the left node 
                                shift += shift_bits # add the (t-1) random bits to the shift
                                break
                            else: # left node exists, so move there 
                                node = node.left
                        else: # b==1, looking to move right
                            if node.right is None: # right node does not exist
                                shift_bits = int(rng.integers(0,2**(t-1))) # get (t-1) random bits
                                node.right = NUSNodeB2(shift_bits,_xb_next,None,None) # create the right node
                                shift += shift_bits # add the (t-1) random bits to the shift
                                break 
                            else: # right node exists, so move there
                                node = node.right
                    elif node.xb==_xb: # this is a leaf node we have already seen before!
                        shift += node.shift_bits
                        break
                    else: #  node.xb!=_xb, this is a leaf node where the _xb values don't match
                        node_b = (node.xb>>(t-1))&1 # leading bit of node.xb
                        node_xb_next = node.xb&ones_mask_tm1 # drop the leading bit of node.xb
                        node_shift_bits_next = node.shift_bits&ones_mask_tm1 # drop the leading bit of node.shift_bits
                        node_leading_shift_bit = (node.shift_bits>>(t-1))&1
                        if node_leading_shift_bit: shift += 2**(t-1)
                        if node_b==0 and b==1: # the node will move its contents left and the _xb will go right
                            node.left = NUSNodeB2(node_shift_bits_next,node_xb_next,None,None)  # create the left node from the current node
                            node.xb,node.shift_bits = None,node_leading_shift_bit # reset the existing node
                            # create the right node 
                            shift_bits = int(rng.integers(0,2**(t-1))) # (t-1) random bits for the right node
                            node.right = NUSNodeB2(shift_bits,_xb_next,None,None)
                            shift += shift_bits
                            break
                        elif node_b==1 and b==0: # the node will move its contents right and the _xb will go left
                            node.right = NUSNodeB2(node_shift_bits_next,node_xb_next,None,None)  # create the right node from the current node
                            node.xb,node.shift_bits = None,node_leading_shift_bit # reset the existing node
                            # create the left node 
                            shift_bits = int(rng.integers(0,2**(t-1))) # (t-1) random bits for the left node
                            node.left = NUSNodeB2(shift_bits,_xb_next,None,None)
                            shift += shift_bits
                            break
                        elif node_b==0 and b==0: # move the node contents and _xb to the left
                            node.left = NUSNodeB2(node_shift_bits_next,node_xb_next,None,None) 
                            node.xb,node.shift_bits = None,node_leading_shift_bit # reset the existing node
                            node = node.left
                        elif node_b==1 and b==1: # move the node contents and _xb to the right 
                            node.right = NUSNodeB2(node_shift_bits_next,node_xb_next,None,None) 
                            node.xb,node.shift_bits = None,node_leading_shift_bit # reset the existing node
                            node = node.right
                    t -= 1
                    _xb = _xb_next
                xrb[l,i,j] = _xb_new^shift

