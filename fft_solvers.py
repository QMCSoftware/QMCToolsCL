import qmcpytoolscl
import numpy as np 

fft_np = lambda x: np.fft.fft(x,norm="ortho")
ifft_np = lambda x: np.fft.ifft(x,norm="ortho")

def fft(x):
    assert x.ndim==1
    n = len(x)
    n_half = np.uint64(n//2)
    xr = x.real.copy()
    xi = x.imag.copy()
    qmcpytoolscl.fft_bro_1d_radix2(1,1,n_half,np.empty(n,dtype=np.double),np.empty(n,dtype=np.double),xr,xi)
    return xr+1j*xi

def ifft(x):
    assert x.ndim==1
    n = len(x)
    n_half = np.uint64(n//2)
    xr = x.real.copy()
    xi = x.imag.copy()
    qmcpytoolscl.ifft_bro_1d_radix2(1,1,n_half,np.empty(n,dtype=np.double),np.empty(n,dtype=np.double),xr,xi)
    return xr+1j*xi

m = 10
n = 2**m

bitrev = np.vectorize(lambda i,m: int('{:0{m}b}'.format(i,m=m)[::-1],2))
ir = bitrev(np.arange(2*n),m+1)

y1 = np.random.rand(n) 
y2 = np.random.rand(n) 
y = np.hstack([y1,y2])

k1 = np.random.rand(n) 
k2 = np.random.rand(n) 
k = np.hstack([k1,k2]) 

# constants 
yt = fft(y) 
kt = fft(k)
y1t = fft(y1) 
y2t = fft(y2) 
k1t = fft(k1) 
k2t = fft(k2)
wt = np.exp(-np.pi*1j*np.arange(n)/n)
wtsq = wt**2
gammat = k1t**2-wtsq*k2t**2

# matrix vector product
u = ifft(yt*kt)*np.sqrt(2*n)
u_np = ifft_np(fft_np(y[ir])*fft_np(k[ir]))[ir]*np.sqrt(2*n)
assert np.allclose(u,u_np,atol=1e-10)
u_hat = np.hstack([ifft(y1t*k1t+wtsq*y2t*k2t),ifft(y2t*k1t+y1t*k2t)])*np.sqrt(n)
assert np.allclose(u_hat,u,atol=1e-10)

# inverse 
v = ifft(yt/kt)/np.sqrt(2*n)
v_np = ifft_np(fft_np(y[ir])/fft_np(k[ir]))[ir]/np.sqrt(2*n)
assert np.allclose(v,v_np,atol=1e-10)
v_hat = np.hstack([ifft((y1t*k1t-wtsq*y2t*k2t)/gammat),ifft((y2t*k1t-y1t*k2t)/gammat)])/np.sqrt(n)
assert np.allclose(v,v_hat,atol=1e-10)
