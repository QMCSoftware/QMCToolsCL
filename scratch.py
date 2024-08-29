from qmcseqcl import lattice_linear

import numpy as np 

gs = np.array([
    [1,433461,315689,441789,501101],
    [1,182667,469891,498753,110745]],
    dtype=np.uint64)

r_x = np.uint64(2) 
n = np.uint64(8) 
d = np.uint64(len(gs))

x = np.zeros((r_x,n,d),dtype=np.float64)

time_perf,time_process = lattice_linear(r_x,n,d,gs,x,
    backend = "cl",
    wait = True,
    PYOPENCL_CTX = "0:2",
    global_size = (1,1,1)
)
print("   time_perf: %.1e\ntime_process: %.1e"%(time_perf,time_process))
print(x)
