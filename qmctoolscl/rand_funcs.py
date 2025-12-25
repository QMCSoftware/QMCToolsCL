import numpy as np 
import time

def random_tbit_uint64s(rng, t, shape):
    """Generate the desired shape of random integers with t bits

Args:
    rng (np.random._generator.Generator): random number generator
    t: (int): number of bits with 0 <= t <= 64
    shape (tuple of ints): shape of resulting integer array"""
    assert 0<=t<=64, "t must be between 0 and 64"
    if t<64: 
        x = rng.integers(0,1<<int(t),shape,dtype=np.uint64)
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

def lat_get_shifts(rng, r, d):
    """Get random shifts
Args:
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications
    d (np.uint64): dimension"""
    shifts = rng.random((r,d))
    return shifts 

def dnb2_get_digital_shifts(rng,r,d,tmax_new):
    """Get random shifts
Args:
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications
    d (np.uint64): dimension
    tmax_new (np.uint64): bits in each integer"""
    shifts = random_tbit_uint64s(rng,tmax_new,(r,d))
    return shifts

def dnb2_get_linear_scramble_matrix(rng, r, d, tmax, tmax_new, print_mats):
    """Return a scrambling matrix for linear matrix scrambling

Args:
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications
    d (np.uint64): dimension
    tmax (np.uint64): bits in each integer
    tmax_new (np.uint64): bits in each integer of the generating matrix after scrambling
    print_mats (np.uint8): flag to print the resulting matrices"""
    tmin = int(min(tmax_new,tmax))
    S = random_tbit_uint64s(rng,tmin,(r,d,tmax_new))
    shift = np.arange(tmin,0,-1,dtype=np.uint64)
    S[:,:,:tmin] >>= shift
    S[:,:,:tmin] <<= shift
    S[:,:,:tmin] += np.uint64(1)<<np.arange(int(tmax)-1,-1,-1,dtype=np.uint64)
    if print_mats:
        print("S with shape (r=%d, d=%d, tmax_new=%d)"%(r,d,tmax_new))
        for l in range(r):
            print("l = %d"%l)
            for j in range(d): 
                print("    j = %d"%j)
                for t in range(tmax_new):
                    b = bin(S[l,j,t])[2:]
                    print("        "+"0"*int(tmax-len(b))+b)
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
    lower_flag = np.tri(int(tmax_new),int(tmax),k=-1,dtype=bool)
    n_lower_flags = lower_flag.sum()
    diag_flag = np.eye(tmax_new,tmax,dtype=bool)
    for l in range(r):
        l_b = int(l%r_b)
        for j in range(d):
            b = bases_2d[l_b,j]
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
    return np.tile(np.eye(mmax,dtype=np.uint64)[None,None,:,:],(int(r),int(d),1,1))

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
         l_b = int(l%r_b)
         for j in range(d):
             b = bases_2d[l_b,j]
             shifts[l,j] = rng.integers(0,b,tmax_new,dtype=np.uint64)
    return shifts

def gdn_get_digital_permutations(rng, r, d, tmax_new, r_b, bases):
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
        l_b = int(l%r_b)
        for j in range(d):
            b = bases_2d[l_b,j]
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
    t_delta = np.uint64(tmax_new-tmax)
    for l in range(r):
        l_x = np.uint64(l%r_x)
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
                _xb_new = xb[l_x,i,j]<<t_delta
                _xb = _xb_new
                node = root_nodes[l,j]
                t = int(tmax_new)
                shift = np.uint64(0)                 
                while t>0:
                    b = int(_xb>>np.uint64(t-1))&1 # leading bit of _xb
                    ones_mask_tm1 = np.uint64(2**(t-1)-1)
                    _xb_next = _xb&ones_mask_tm1 # drop the leading bit of _xb 
                    if node.xb is None: # this is not a leaf node, so node.shift_bits in [0,1]
                        if node.shift_bits: shift += np.uint64(2**(t-1)) # add node.shift_bits to the shift
                        if b==0: # looking to move left
                            if node.left_b2 is None: # left node does not exist
                                shift_bits = np.uint64(rng.integers(0,2**(t-1))) # get (t-1) random bits
                                node.left_b2 = NUSNode_dnb2(shift_bits,_xb_next,None,None) # create the left node 
                                shift += shift_bits # add the (t-1) random bits to the shift
                                break
                            else: # left node exists, so move there 
                                node = node.left_b2
                        else: # b==1, looking to move right
                            if node.right_b2 is None: # right node does not exist
                                shift_bits = np.uint64(rng.integers(0,2**(t-1))) # get (t-1) random bits
                                node.right_b2 = NUSNode_dnb2(shift_bits,_xb_next,None,None) # create the right node
                                shift += shift_bits # add the (t-1) random bits to the shift
                                break 
                            else: # right node exists, so move there
                                node = node.right_b2
                    elif node.xb==_xb: # this is a leaf node we have already seen before!
                        shift += node.shift_bits
                        break
                    else: #  node.xb!=_xb, this is a leaf node where the _xb values don't match
                        node_b = int(node.xb>>np.uint64(t-1))&1 # leading bit of node.xb
                        node_xb_next = node.xb&ones_mask_tm1 # drop the leading bit of node.xb
                        node_shift_bits_next = node.shift_bits&ones_mask_tm1 # drop the leading bit of node.shift_bits
                        node_leading_shift_bit = int(node.shift_bits>>np.uint64(t-1))&1
                        if node_leading_shift_bit: shift += np.uint64(2**(t-1))
                        if node_b==0 and b==1: # the node will move its contents left and the _xb will go right
                            node.left_b2 = NUSNode_dnb2(node_shift_bits_next,node_xb_next,None,None)  # create the left node from the current node
                            node.xb,node.shift_bits = None,node_leading_shift_bit # reset the existing node
                            # create the right node 
                            shift_bits = np.uint64(rng.integers(0,2**(t-1))) # (t-1) random bits for the right node
                            node.right_b2 = NUSNode_dnb2(shift_bits,_xb_next,None,None)
                            shift += shift_bits
                            break
                        elif node_b==1 and b==0: # the node will move its contents right and the _xb will go left
                            node.right_b2 = NUSNode_dnb2(node_shift_bits_next,node_xb_next,None,None)  # create the right node from the current node
                            node.xb,node.shift_bits = None,node_leading_shift_bit # reset the existing node
                            # create the left node 
                            shift_bits = np.uint64(rng.integers(0,2**(t-1))) # (t-1) random bits for the left node
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
        l_b = int(l%r_b)
        l_x = int(l%r_x)
        for j in range(d):
            rng = rngs[l,j]
            root_node = root_nodes[l,j]
            b = bases[l_b,j]
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
                while t<=tmax:
                    _xdig = np.zeros(np.uint64(tmax_new-t),dtype=np.uint64)
                    _xdig[:np.uint64(tmax-t)] = xdig[l_x,i,j,t:]
                    dig = _xdig[0]
                    if node.xdig is None: # this is not a leaf node, so node.perm is a single permutation
                        perm[t] = node.perm[dig] # set the permuted value
                        if node.children[dig] is None: # child in dig position does not exist
                            node_perm = random_uint64_permutations(rng,np.uint64(tmax_new-t-1),b)
                            node.children[dig] = NUSNode_gdn(node_perm,_xdig[1:],[None]*b)
                            perm[(t+1):] = node_perm[np.arange(int(tmax_new)-t-1,dtype=np.uint64),_xdig[1:]] # digits in _xdig[1:] index node_perm rows
                            break
                        else: # child in dig position exists, so move there 
                            node = node.children[dig]
                    elif (node.xdig==_xdig).all(): # this is a leaf node we have already seen before!
                        perm[t:] = node.perm[np.arange(int(tmax_new)-t,dtype=np.uint64),_xdig] # digits in _xdig index node_perm rows
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
                            dig_node_perm = random_uint64_permutations(rng,np.uint64(tmax_new-t-1),b)
                            node.children[dig] = NUSNode_gdn(dig_node_perm,_xdig[1:],[None]*b) # create a new leaf node
                            perm[(t+1):] = dig_node_perm[np.arange(int(tmax_new)-t-1,dtype=np.uint64),_xdig[1:]] # digits in _xdig[1:] index node_perm rows
                            break
                    t += 1
                xrdig[l,i,j] = perm
    tdelta_process = time.process_time()-t0_process 
    tdelta_perf = time.perf_counter()-t0_perf
    return tdelta_perf,tdelta_process