import pyopencl as cl
import numpy as np
from time import time 
import os 

Cs = [
    np.load("./qmcpy/discrete_distribution/digital_net_b2/generating_matrices/sobol_mat.21201.32.32.msb.npy").astype(np.uint64),
    np.load("./qmcpy/discrete_distribution/digital_net_b2/generating_matrices/sobol_mat.51.30.30.msb.npy").astype(np.uint64)
]

mmax = 28 
tmaxes_x = np.array([32,30],dtype=np.uint64)
tmax = 64
lshifts = tmax-tmaxes_x

gc = np.uint8(True)
nstart = np.uint64(0)
r_x,r,n,d = 2,4,8,3

if not gc: assert (nstart==0 or (nstart&(nstart-1))==0) and ((nstart+n)&(nstart+n-1))==0, "lattice with natural ordering requires (nstart is 0 or a power of 2) and (nstart+n is a power of 2)"
assert all(d<=len(C) for C in Cs) and all(mmax<C.shape[1] for C in Cs)
C = np.array([C[:d,:mmax] for C in Cs],dtype=np.uint64)

global_size_r_x,global_size_r,global_size_n,global_size_d = np.uint64(np.ceil(r_x/r_x)),np.uint64(np.ceil(r/r)),np.uint64(np.ceil(n/n)),np.uint64(np.ceil(d/d))
batch_size_r_x,batch_size_r,batch_size_n,batch_size_d = np.uint64(np.ceil(r_x/global_size_r_x)),np.uint64(np.ceil(r/global_size_r)),np.uint64(np.ceil(n/global_size_n)),np.uint64(np.ceil(d/global_size_d))
print("global_size_r_x  = %d"%global_size_r_x)
print("global_size_n    = %d"%global_size_n)
print("global_size_d    = %d"%global_size_d)
print()
print("batch_size_r_x = %d"%batch_size_r_x)
print("batch_size_r = %d"%batch_size_r)
print("batch_size_n = %d"%batch_size_n)
print("batch_size_r = %d"%batch_size_d)

os.environ["PYOPENCL_CTX"] = "0"

FILEDIR = os.path.dirname(os.path.realpath(__file__))
with open(FILEDIR+"/digital_net_b2.c","r") as kernel_file:
    kernelsource = kernel_file.read()
    kernelsource = kernelsource.replace("void","__kernel void")

mf = cl.mem_flags
ctx = cl.create_some_context()
prg = cl.Program(ctx,kernelsource).build()
queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)

rng = np.random.default_rng()

C_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C)
xb_d = cl.Buffer(ctx, mf.READ_WRITE, np.dtype(np.uint64).itemsize*r_x*n*d)
x_d = cl.Buffer(ctx, mf.READ_WRITE, np.dtype(np.uint64).itemsize*r_x*n*d)
xrb_d = cl.Buffer(ctx, mf.READ_WRITE, np.dtype(np.uint64).itemsize*r*n*d)
xr_d = cl.Buffer(ctx, mf.WRITE_ONLY, np.dtype(np.uint64).itemsize*r*n*d)
shiftsb_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=(rng.integers(np.iinfo(np.int64).min,np.iinfo(np.int64).max,(r,d),endpoint=True)+np.iinfo(np.int64).min).astype(np.uint64))
tmaxes_x_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmaxes_x)
tmaxes_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tmax*np.ones(r,dtype=np.uint64))
lshifts_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lshifts)

digital_net_b2_binary = prg.digital_net_b2_binary
event_dnb2_b = digital_net_b2_binary(queue,(global_size_r_x,global_size_n,global_size_d),None,np.uint64(r_x),np.uint64(n),np.uint64(d),np.uint64(batch_size_r_x),np.uint64(batch_size_n),np.uint64(batch_size_d),nstart,gc,np.uint64(mmax),C_d,xb_d)
xb_dnb2_b_cl = np.empty((r_x,n,d),dtype=np.uint64)
cl.enqueue_copy(queue,xb_dnb2_b_cl,xb_d)
print(xb_dnb2_b_cl)

digital_net_b2_from_binary = prg.digital_net_b2_from_binary
event_dnb2_fb = digital_net_b2_from_binary(queue,(global_size_r_x,global_size_n,global_size_d),None,np.uint64(r_x),np.uint64(n),np.uint64(d),np.uint64(batch_size_r_x),np.uint64(batch_size_n),np.uint64(batch_size_d),tmaxes_x_d,xb_d,x_d)
xb_dnb2_f_cl = np.empty((r_x,n,d),dtype=np.float64)
cl.enqueue_copy(queue,xb_dnb2_f_cl,x_d)
print(xb_dnb2_f_cl)

digital_net_b2_binary_rdshift = prg.digital_net_b2_binary_rdshift
event_dnb2_rb = digital_net_b2_binary_rdshift(queue,(global_size_r,global_size_n,global_size_d),None,np.uint64(r),np.uint64(n),np.uint64(d),np.uint64(batch_size_r),np.uint64(batch_size_n),np.uint64(batch_size_d),np.uint64(r_x),lshifts_d,xb_d,shiftsb_d,xrb_d)
xb_dnb2_rb_cl = np.empty((r,n,d),dtype=np.uint64)
cl.enqueue_copy(queue,xb_dnb2_rb_cl,xrb_d)
print(xb_dnb2_rb_cl)

digital_net_b2_from_binary = prg.digital_net_b2_from_binary
event_dnb2_rfb = digital_net_b2_from_binary(queue,(global_size_r,global_size_n,global_size_d),None,np.uint64(r),np.uint64(n),np.uint64(d),np.uint64(batch_size_r),np.uint64(batch_size_n),np.uint64(batch_size_d),tmaxes_d,xrb_d,xr_d)
xb_dnb2_rf_cl = np.empty((r,n,d),dtype=np.float64)
cl.enqueue_copy(queue,xb_dnb2_rf_cl,xr_d)
print(xb_dnb2_rf_cl)

C_lsb = np.load("./qmcpy/discrete_distribution/digital_net_b2/generating_matrices/sobol_mat.21201.32.32.lsb.npy").astype(np.uint64)
d = 3
C_lsb = C_lsb[:d]
mmax = 32
tmax = 32
C_d = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C_lsb)
tmaxes_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array([tmax],dtype=np.uint64))
gen_mats_lsb_to_msb = prg.gen_mats_lsb_to_msb
event_lsb_to_msb = gen_mats_lsb_to_msb(queue,(1,1,1),None,np.uint64(1),np.uint64(d),np.uint64(mmax),np.uint64(1),np.uint64(d),np.uint64(mmax),tmaxes_d,C_d,C_d)
C_msb_cl = np.empty((d,mmax),dtype=np.uint64)
cl.enqueue_copy(queue,C_msb_cl,C_d)
C_msb = np.load("./qmcpy/discrete_distribution/digital_net_b2/generating_matrices/sobol_mat.21201.32.32.msb.npy").astype(np.uint64)
assert ((C_msb_cl-C_msb[:d])==0).all()

r_C = 2
tmax_new = 45
slow,shigh = np.uint64(1)<<np.maximum(0,tmax-1-np.arange(tmax_new,dtype=np.uint64)),np.uint64(1)<<np.uint64(tmax)
S = rng.integers(0,np.uint64(1)<<np.minimum(np.arange(tmax_new,dtype=np.uint64),tmax),(r_C,d,tmax_new),dtype=np.uint64)
S[:,:,:tmax] <<= np.arange(tmax,0,-1,dtype=np.uint64)
S[:,:,:tmax] += np.uint64(1)<<np.arange(tmax-1,-1,-1,dtype=np.uint64)
for l in range(r_C):
    for j in range(d): 
        for t in range(tmax_new):
            b = bin(S[l,j,t])[2:]
            print("\t"+"0"*(tmax-len(b))+b)
        print()
S_d = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=S)
C_lms_d = cl.Buffer(ctx, mf.WRITE_ONLY, np.dtype(np.uint64).itemsize*r_C*d*mmax)
gen_mats_linear_matrix_scramble = prg.gen_mats_linear_matrix_scramble 
event_lms = gen_mats_linear_matrix_scramble(queue,(1,1,1),None,np.uint64(r_C),np.uint64(d),np.uint64(mmax),np.uint64(r_C),np.uint64(d),np.uint64(mmax),np.uint64(1),np.uint64(tmax_new),tmaxes_d,S_d,C_d,C_lms_d)
C_lms_cl = np.empty((r_C,d,mmax),dtype=np.uint64)
cl.enqueue_copy(queue,C_lms_cl,C_lms_d)
print(C_lms_cl)
