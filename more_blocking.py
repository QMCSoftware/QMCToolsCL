import numpy as np

import pyopencl as cl

ctx = cl.create_some_context()

prg = cl.Program(ctx, """
__kernel void sum(
  __global double *y,
  ulong n,
  ulong d, 
  ulong batch_size_n,
  ulong batch_size_d)
{   ulong i,j,idx;
    ulong i0 = get_global_id(0)*batch_size_n;
    ulong j0 = get_global_id(1)*batch_size_d;
    for(i=0; i<batch_size_n; i++){
        for(j=0; j<batch_size_d; j++){
            idx = (i0+i)*n+(j0+j);
            y[idx] = idx;
        }
    }
}
""").build()

n,d = 4,8
global_size_n,global_size_d = 4,2
batch_size_n,batch_size_d = n//global_size_n,d//global_size_d
local_size_n,local_size_d = 2,2

queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
y_d = cl.Buffer(ctx, mf.WRITE_ONLY, np.dtype(np.float64).itemsize*n*d)
knl = prg.sum
knl(queue,(global_size_n,global_size_d),(local_size_n,local_size_d),y_d,np.uint64(n),np.uint64(d),np.uint64(batch_size_n),np.uint64(batch_size_d))

y_h = np.empty((n,d),dtype=np.float64)
cl.enqueue_copy(queue,y_h,y_d)

print(y_h)