import qmctoolscl
import numpy as np 

g = np.array([
    [1,433461,315689,441789,501101],
    [1,182667,469891,498753,110745]],
    dtype=np.uint64)
r = np.uint64(g.shape[0]) 
n = np.uint64(8) 
d = np.uint64(g.shape[1])
x = np.empty((r,n,d),dtype=np.float64)
time_perf,time_process = qmctoolscl.lat_gen_linear(r,n,d,g,x)
print(x)