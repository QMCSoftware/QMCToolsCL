import sympy
import numpy as np 
import qmcpytoolscl

def fwht(x):
    assert x.ndim==1
    n = len(x)
    n_half = np.uint64(n//2)
    x_cp = x.copy()
    qmcpytoolscl.fwht_1d_radix2(1,1,n_half,x_cp)
    return x_cp

def fwht_sp(x):
    assert x.ndim==1
    n = len(x) 
    y = np.array(sympy.fwht(x),dtype=np.float64)
    y_ortho = y/np.sqrt(n) 
    return y_ortho

m = 10
n = int(2**m)

y1 = np.random.rand(n) 
y2 = np.random.rand(n) 
y = np.hstack([y1,y2])

k1 = np.random.rand(n) 
k2 = np.random.rand(n) 
k = np.hstack([k1,k2]) 

# constants 

yt = fwht(y) 
kt = fwht(k)
y1t = fwht(y1) 
y2t = fwht(y2) 
k1t = fwht(k1) 
k2t = fwht(k2)
gammat = k1t**2-k2t**2

# matrix vector product
u_sp = fwht_sp(fwht_sp(y)*fwht_sp(k))*np.sqrt(2*n)
u = fwht(yt*kt)*np.sqrt(2*n)
assert np.allclose(u_sp,u,atol=1e-8)
u_hat = np.hstack([fwht(y1t*k1t+y2t*k2t),fwht(y2t*k1t+y1t*k2t)])*np.sqrt(n)
assert np.allclose(u,u_hat,atol=1e-10)

# inverse 
v_sp = fwht_sp(fwht_sp(y)/fwht_sp(k))/np.sqrt(2*n)
v = fwht(yt/kt)/np.sqrt(2*n)
assert np.allclose(v_sp,v,atol=1e-8)
v_hat = np.hstack([fwht((y1t*k1t-y2t*k2t)/gammat),fwht((y2t*k1t-y1t*k2t)/gammat)])/np.sqrt(n)
assert np.allclose(v,v_hat,atol=1e-10)
