import qmcseqcl
import numpy as np 

kwargs = {}

g = np.array([
     [1,433461,315689,441789,501101],
     [1,182667,469891,498753,110745]],
     dtype=np.uint64)
r = np.uint64(g.shape[0]) 
nstart = np.uint64(4)
n = np.uint64(4) 
d = np.uint64(g.shape[1])
x = np.empty((r,n,d),dtype=np.float64)
gc = np.uint8(False)
time_perf,time_process = qmcseqcl.lattice_b2(r,n,d,nstart,gc,g,x,**kwargs)
print(x)