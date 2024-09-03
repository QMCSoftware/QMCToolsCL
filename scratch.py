from qmcseqcl import lattice_linear

import numpy as np 

print(lattice_linear.__doc__)

gs = np.array([
    [1,433461,315689,441789,501101],
    [1,182667,469891,498753,110745]],
    dtype=np.uint64)

r_x = np.uint64(gs.shape[0]) 
n = np.uint64(16) 
d = np.uint64(gs.shape[1])

x = np.zeros((r_x,n,d),dtype=np.float64)

time_perf,time_process = lattice_linear(r_x,n,d,gs,x,
    backend = "cl",
    wait = True,
    PYOPENCL_CTX = "0:2",
    global_size = (1,2,1)
)
print("\n   time_perf: %.1e\ntime_process: %.1e\n"%(time_perf,time_process))
print(x)
