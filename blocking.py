import numpy as np

import pyopencl as cl

ctx = cl.create_some_context()

prg = cl.Program(ctx, """
__kernel void sum(
  __global double *y)
{
  int i0 = get_global_id(0);
  int i1 = get_global_id(1);
  int i2 = get_global_id(2);
  int d = get_global_size(2);
  int nd = get_global_size(1)*d;
  int i = i0*nd+i1*d+i2;
  y[i] = get_local_size(0);
}
""").build()

r,n,d = 4,4,4

rng = np.random.default_rng()
b_np = rng.random((r,n,d),dtype=np.float64)

queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
y_d = cl.Buffer(ctx, mf.WRITE_ONLY, np.dtype(np.float64).itemsize*r*n*d)
knl = prg.sum
knl(queue,(r,n,d),(2,2,2),y_d)

y_h = np.empty((r,n,d),dtype=np.float64)
cl.enqueue_copy(queue,y_h,y_d)

print(y_h)