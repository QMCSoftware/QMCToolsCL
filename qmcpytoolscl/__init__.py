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

def get_qmcpytoolscl_program_from_context(context):
    import pyopencl as cl
    FILEDIR = os.path.dirname(os.path.realpath(__file__))
    with open(FILEDIR+"/qmcpytoolscl.cl","r") as kernel_file:
        kernelsource = kernel_file.read()
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
            raise ImportError("install pyopencl to access these capabilities in QMCPyToolsCL")
        if "context" not in kwargs:
            platform = cl.get_platforms()[kwargs["platform_id"] if "platform_id" in kwargs else 0]
            device = platform.get_devices()[kwargs["device_id"] if "device_id" in kwargs else 0]
            kwargs["context"] = cl.Context([device])
        if "queue" not in kwargs:
            if "profile" in kwargs and kwargs["profile"]:
                kwargs["queue"] = cl.CommandQueue(kwargs["context"],properties=cl.command_queue_properties.PROFILING_ENABLE)
            else:
                kwargs["queue"] = cl.CommandQueue(kwargs["context"])

def _preprocess_fwht(*args_device,kwargs):
    if kwargs["local_size"] is None or kwargs["local_size"][2]!=kwargs["global_size"][2]:
        raise Exception("fwht requires local_size is not None and local_size[2] = %d equals global_size[2] = %d"%(kwargs["local_size"][2],kwargs["global_size"][2]))

overwrite_args = {
    "rfft_1d_radix2": 2, 
}

def opencl_c_func(func):
    func_name = func.__name__
    def wrapped_func(*args, **kwargs):
        _parse_kwargs_backend_queue_program(kwargs)
        args = list(args)
        if kwargs["backend"]=="c":
            t0_perf = time.perf_counter()
            t0_process = time.process_time()
            args = args[:3]+args[:3]+args[3:] # repeat the first 3 args to the batch sizes
            eval("%s_c(*args)"%func_name)
            tdelta_process = time.process_time()-t0_process 
            tdelta_perf = time.perf_counter()-t0_perf 
            return tdelta_perf,tdelta_process
        else: # kwargs["backend"]=="cl"
            import pyopencl as cl
            t0_perf = time.perf_counter()
            if "program" not in kwargs:
                kwargs["program"] =  get_qmcpytoolscl_program_from_context(kwargs["context"])
            assert "global_size" in kwargs 
            kwargs["global_size"] = [min(kwargs["global_size"][i],args[i]) for i in range(3)]
            batch_size = [np.uint64(np.ceil(args[i]/kwargs["global_size"][i])) for i in range(3)]
            kwargs["global_size"] = [np.uint64(np.ceil(args[i]/batch_size[i])) for i in range(3)]
            if "local_size" not in kwargs:
                kwargs["local_size"] = None
            cl_func = getattr(kwargs["program"],func_name)
            args_device = [cl.Buffer(kwargs["context"],cl.mem_flags.READ_WRITE|cl.mem_flags.COPY_HOST_PTR,hostbuf=arg) if isinstance(arg,np.ndarray) else arg for arg in args]
            args_device = args_device[:3]+batch_size+args_device[3:] # repeat the first 3 args to the batch sizes
            try:
                eval('_preprocess_%s(*args_device,kwargs=kwargs)'%func_name)
            except NameError: pass
            event = cl_func(kwargs["queue"],kwargs["global_size"],kwargs["local_size"],*args_device)
            if "wait" not in kwargs or kwargs["wait"]:
                event.wait()
                try:
                    tdelta_process = (event.profile.end - event.profile.start)*1e-9
                except cl._cl.RuntimeError:
                    tdelta_process = -1
            else:
                tdelta_process = -1
            if isinstance(args[-1],np.ndarray):
                num_overwrite_args = overwrite_args[func_name] if func_name in overwrite_args else 1
                for i in range(-1,-1-num_overwrite_args,-1):
                    cl.enqueue_copy(kwargs["queue"],args[i],args_device[i])
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

THISDIR = os.path.dirname(os.path.realpath(__file__))

with open("%s/qmcpytoolscl.cl"%THISDIR,"r") as f:
    code = f.read() 
blocks = re.findall(r'(?<=void\s).*?(?=\s?\))',code,re.DOTALL)
for block in blocks:
    lines = block.replace("(","").splitlines()
    name = lines[0]
    desc = [] 
    si = 1
    while lines[si].strip()[:2]=="//":
        desc += [lines[si].split("// ")[1].strip()]
        si += 1
    desc = "\n".join(desc)
    args = []
    doc_args = []
    for i in range(si,len(lines)):
        var_input,var_desc = lines[i].split(" // ")
        var_type,var = var_input.replace(",","").split(" ")[-2:]
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

def random_tbit_uint64s(rng, t, shape):
    """Generate the desired shape of random integers with t bits

Args:
    rng (np.random._generator.Generator): random number generator
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

def random_uint64_permutations(rng, n, b):
    """Generate n permutations of 0,...,b-1 into a size (n,b) np.ndarray of np.uint64

Args:
    rng (np.random._generator.Generator): random number generator
    n (int): number of permutations
    b (int): permute 0,...,b-1"""
    x = np.empty((n,b),dtype=np.uint64)
    for i in range(n): 
        x[i] = rng.permutation(b) 
    return x
def dnb2_get_linear_scramble_matrix(rng, r, d, tmax, tmax_new, print_mats):
    """Return a scrambling matrix for linear matrix scrambling

Args:
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications
    d (np.uint64): dimension
    tmax (np.uint64): bits in each integer
    tmax_new (np.uint64): bits in each integer of the generating matrix after scrambling
    print_mats (np.uint8): flag to print the resulting matrices"""
    S = np.empty((r,d,tmax_new),dtype=np.uint64) 
    for t in range(tmax_new):
        S[:,:,t] = random_tbit_uint64s(rng,min(t,tmax),(r,d))
    S[:,:,:tmax] <<= np.arange(tmax,0,-1,dtype=np.uint64)
    S[:,:,:tmax] += np.uint64(1)<<np.arange(tmax-1,-1,-1,dtype=np.uint64)
    if print_mats:
        print("S with shape (r=%d, d=%d, tmax_new=%d)"%(r,d,tmax_new))
        for l in range(r):
            print("l = %d"%l)
            for j in range(d): 
                print("    j = %d"%j)
                for t in range(tmax_new):
                    b = bin(S[l,j,t])[2:]
                    print("        "+"0"*(tmax-len(b))+b)
    return S

def gdn_get_linear_scramble_matrix(rng, r, d, tmax, tmax_new, r_b, bases):
    """Return a scrambling matrix for linear matrix scrambling

Args:
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications
    d (np.uint64): dimension
    tmax (np.uint64): bits in each integer
    tmax_new (np.uint64): bits in each integer of the generating matrix after scrambling
    r_b (np.uint64): replications of bases 
    bases (np.ndarray of np.uint64): bases of size r_b*d"""
    S = np.empty((r,d,tmax_new,tmax),dtype=np.uint64)
    bases_2d = np.atleast_2d(bases)
    lower_flag = np.tri(int(tmax_new),int(tmax),k=-1,dtype=np.bool)
    n_lower_flags = lower_flag.sum()
    diag_flag = np.eye(tmax_new,tmax,dtype=np.bool)
    for l in range(r):
        for j in range(d):
            b = bases_2d[l%r_b,j]
            Slj = np.zeros((tmax_new,tmax),dtype=np.uint64)
            Slj[lower_flag] = rng.integers(0,b,n_lower_flags)
            Slj[diag_flag] = rng.integers(1,b,tmax)
            S[l,j] = Slj
    return S

def gdn_get_halton_generating_matrix(r,d,mmax):
    """Return the identity matrices comprising the Halton generating matrices
    
Arg:
    r (np.uint64): replications 
    d (np.uint64): dimension 
    mmax (np.uint64): maximum number rows and columns in each generating matrix"""
    return np.tile(np.eye(mmax,dtype=np.uint64)[None,None,:,:],(r,d,1,1))

def gdn_get_digital_shifts(rng, r, d, tmax_new, r_b, bases):
    """Return digital shifts for gdn

Args: 
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications 
    d (np.uint64): dimension 
    tmax_new (np.uint64): number of bits in each shift 
    r_b (np.uint64): replications of bases 
    bases (np.ndarray of np.uint64): bases of size r_b*d"""
    shifts = np.empty((r,d,tmax_new),dtype=np.uint64)
    bases_2d = np.atleast_2d(bases)
    for l in range(r):
         for j in range(d):
             b = bases_2d[l%r_b,j]
             shifts[l,j] = rng.integers(0,b,tmax_new,dtype=np.uint64)
    return shifts

def gdn_get_permutations(rng, r, d, tmax_new, r_b, bases):
    """Return permutations for gdn

Args: 
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications 
    d (np.uint64): dimension 
    tmax_new (np.uint64): number of bits in each shift 
    r_b (np.uint64): replications of bases 
    bases (np.ndarray of np.uint64): bases of size r_b*d"""
    bases_2d = np.atleast_2d(bases)
    bmax = bases_2d.max()
    perms = np.zeros((r,d,tmax_new,bmax),dtype=np.uint64)
    for l in range(r):
        for j in range(d):
            b = bases_2d[l%r_b,j]
            for t in range(tmax_new):
                perms[l,j,t,:b] = rng.permutation(b)
    return perms

class NUSNode_dnb2(object):
    def __init__(self, shift_bits=None, xb=None, left_b2=None, right_b2=None):
        self.shift_bits = shift_bits
        self.xb = xb 
        self.left_b2 = left_b2 
        self.right_b2 = right_b2

def dnb2_nested_uniform_scramble(
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
    """Nested uniform scramble of digital net b2

Args: 
    r (np.uint64): replications 
    n (np.uint64): points
    d (np.uint64): dimensions
    r_x (np.uint64): replications of xb
    tmax (np.uint64): maximum number of bits in each integer
    tmax_new (np.uint64): maximum number of bits in each integer after scrambling
    rngs (np.ndarray of numpy.random._generator.Generator): random number generators of size r*d
    root_nodes (np.ndarray of NUSNode_dnb2): root nodes of size r*d
    xb (np.ndarray of np.uint64): array of unrandomized points of size r*n*d
    xrb (np.ndarray of np.uint64): array to store scrambled points of size r*n*d"""
    t0_perf = time.perf_counter()
    t0_process = time.process_time()
    for l in range(r): 
        for j in range(d):
            rng = rngs[l,j]
            root_node = root_nodes[l,j]
            assert isinstance(root_node,NUSNode_dnb2)
            if root_node.shift_bits is None:
                # initilize root nodes 
                assert root_node.xb is None and root_node.left_b2 is None and root_node.right_b2 is None
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
                            if node.left_b2 is None: # left node does not exist
                                shift_bits = int(rng.integers(0,2**(t-1))) # get (t-1) random bits
                                node.left_b2 = NUSNode_dnb2(shift_bits,_xb_next,None,None) # create the left node 
                                shift += shift_bits # add the (t-1) random bits to the shift
                                break
                            else: # left node exists, so move there 
                                node = node.left_b2
                        else: # b==1, looking to move right
                            if node.right_b2 is None: # right node does not exist
                                shift_bits = int(rng.integers(0,2**(t-1))) # get (t-1) random bits
                                node.right_b2 = NUSNode_dnb2(shift_bits,_xb_next,None,None) # create the right node
                                shift += shift_bits # add the (t-1) random bits to the shift
                                break 
                            else: # right node exists, so move there
                                node = node.right_b2
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
                            node.left_b2 = NUSNode_dnb2(node_shift_bits_next,node_xb_next,None,None)  # create the left node from the current node
                            node.xb,node.shift_bits = None,node_leading_shift_bit # reset the existing node
                            # create the right node 
                            shift_bits = int(rng.integers(0,2**(t-1))) # (t-1) random bits for the right node
                            node.right_b2 = NUSNode_dnb2(shift_bits,_xb_next,None,None)
                            shift += shift_bits
                            break
                        elif node_b==1 and b==0: # the node will move its contents right and the _xb will go left
                            node.right_b2 = NUSNode_dnb2(node_shift_bits_next,node_xb_next,None,None)  # create the right node from the current node
                            node.xb,node.shift_bits = None,node_leading_shift_bit # reset the existing node
                            # create the left node 
                            shift_bits = int(rng.integers(0,2**(t-1))) # (t-1) random bits for the left node
                            node.left_b2 = NUSNode_dnb2(shift_bits,_xb_next,None,None)
                            shift += shift_bits
                            break
                        elif node_b==0 and b==0: # move the node contents and _xb to the left
                            node.left_b2 = NUSNode_dnb2(node_shift_bits_next,node_xb_next,None,None) 
                            node.xb,node.shift_bits = None,node_leading_shift_bit # reset the existing node
                            node = node.left_b2
                        elif node_b==1 and b==1: # move the node contents and _xb to the right 
                            node.right_b2 = NUSNode_dnb2(node_shift_bits_next,node_xb_next,None,None) 
                            node.xb,node.shift_bits = None,node_leading_shift_bit # reset the existing node
                            node = node.right_b2
                    t -= 1
                    _xb = _xb_next
                xrb[l,i,j] = _xb_new^shift
    tdelta_process = time.process_time()-t0_process 
    tdelta_perf = time.perf_counter()-t0_perf
    return tdelta_perf,tdelta_process,

class NUSNode_gdn(object):
    def __init__(self, perm=None, xdig=None, children=None):
        self.perm = perm
        self.xdig = xdig 
        self.children = children

def gdn_nested_uniform_scramble(
    r,
    n, 
    d,
    r_x,
    r_b,
    tmax,
    tmax_new,
    rngs,
    root_nodes,
    bases,
    xdig,
    xrdig):
    """Nested uniform scramble of general digital nets

Args: 
    r (np.uint64): replications 
    n (np.uint64): points
    d (np.uint64): dimensions
    r_x (np.uint64): replications of xb
    r_b (np.uint64): replications of bases
    tmax (np.uint64): maximum number digits in each point representation
    tmax_new (np.uint64): maximum number digits in each point representation after scrambling
    rngs (np.ndarray of numpy.random._generator.Generator): random number generators of size r*d
    root_nodes (np.ndarray of NUSNode_gdn): root nodes of size r*d
    bases (np.ndarray of np.uint64): array of bases of size r*d
    xdig (np.ndarray of np.uint64): array of unrandomized points of size r*n*d*tmax
    xrdig (np.ndarray of np.uint64): array to store scrambled points of size r*n*d*tmax_new"""
    t0_perf = time.perf_counter()
    t0_process = time.process_time()
    for l in range(r): 
        for j in range(d):
            rng = rngs[l,j]
            root_node = root_nodes[l,j]
            b = bases[l%r_b,j]
            assert isinstance(root_node,NUSNode_gdn)
            if root_node.perm is None:
                # initilize root nodes
                assert root_node.xdig is None and root_node.children is None
                root_node.xdig = np.zeros(tmax_new,dtype=np.uint64) 
                root_node.perm = random_uint64_permutations(rng,tmax_new,b)
                root_node.children = [None]*b
            for i in range(n):
                node = root_nodes[l,j]
                t = 0
                perm = np.zeros(tmax_new,dtype=np.uint64)         
                while t<tmax:
                    _xdig = np.zeros(tmax_new-t,dtype=np.uint64)
                    _xdig[:(tmax-t)] = xdig[l%r_x,i,j,t:]
                    dig = _xdig[0]
                    if node.xdig is None: # this is not a leaf node, so node.perm is a single permutation
                        perm[t] = node.perm[dig] # set the permuted value
                        if node.children[dig] is None: # child in dig position does not exist
                            node_perm = random_uint64_permutations(rng,tmax_new-t-1,b)
                            node.children[dig] = NUSNode_gdn(node_perm,_xdig[1:],[None]*b)
                            perm[(t+1):] = node_perm[np.arange(tmax_new-t-1,dtype=np.uint64),_xdig[1:]] # digits in _xdig[1:] index node_perm rows
                            break
                        else: # child in dig position exists, so move there 
                            node = node.children[dig]
                    elif (node.xdig==_xdig).all(): # this is a leaf node we have already seen before!
                        perm[t:] = node.perm[np.arange(tmax_new-t,dtype=np.uint64),_xdig] # digits in _xdig index node_perm rows
                        break
                    else: # node.xdig!=_xdig, this is a leaf node where the _xdig values don't match
                        node_dig = node.xdig[0]
                        perm[t] = node.perm[0,dig]
                        # move node contenst to the child in the dig position
                        node.children[node_dig] = NUSNode_gdn(node.perm[1:],node.xdig[1:],[None]*b) 
                        node.perm = node.perm[0]
                        node.xdig = None
                        if node_dig==dig: 
                            node = node.children[dig] 
                        else: # create child node in the dig position
                            dig_node_perm = random_uint64_permutations(rng,tmax_new-t-1,b)
                            node.children[dig] = NUSNode_gdn(dig_node_perm,_xdig[1:],[None]*b) # create a new leaf node
                            perm[(t+1):] = dig_node_perm[np.arange(tmax_new-t-1,dtype=np.uint64),_xdig[1:]] # digits in _xdig[1:] index node_perm rows
                            break
                    t += 1
                xrdig[l,i,j] = perm
    tdelta_process = time.process_time()-t0_process 
    tdelta_perf = time.perf_counter()-t0_perf
    return tdelta_perf,tdelta_process,
