<h1>Quasi-Monte Carlo Sequences in OpenCL & C</h1>

The `QMCSeqCL` package provides a Python interface to functions in **OpenCL** and **C** for generating **Quasi-Monte Carlo (QMC) sequences**. Point generation and randomization routines are written as OpenCL kernels. By replacing a few code snippets, these OpenCL kernels are automatically translated to C functions. Python functions provide access to both the OpenCL kernels and C functions in a unified interface. Code is available on <a href="https://github.com/QMCSoftware/QMCseqCL" target="_blank">GitHub</a>. 

# Installation 

For now, install from source via 

```
git clone https://github.com/QMCSoftware/QMCseqCL.git
cd QMCseqCL
pip install -e . 
```

To use OpenCL features, please install <a href="https://pypi.org/project/pyopencl/" target="_blank">PyOpenCL</a>.


# Setup for OpenCL vs C 

Let's start by importing the relevant packages 

```python
>>> import qmcseqcl
>>> import numpy as np
```

and seeding a random number generator used for reproducibility

```python
>>> rng = np.random.Generator(np.random.SFC64(7))
```

To use the **C backend** supply the following keyword arguments to function calls

```python
>>> kwargs = {
...     "backend": "C",
... }
```

To use the **OpenCL backend** supply the following keyword arguments to function calls

```python
>>> kwargs = {
...     "backend": "CL",
...     "wait": True, # required for accurate timing
...     "global_size": (2,2,2), # global size 
... }
```

# Lattice Sequences

## linear order

```python
>>> print(qmcseqcl.lattice_linear.__doc__)
Lattice points in linear order

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    g (np.ndarray of np.uint64): pointer to generating vector of size r*d
    x (np.ndarray of np.double): pointer to point storage of size r*n*d
>>> g = np.array([
...     [1,433461,315689,441789,501101],
...     [1,182667,469891,498753,110745]],
...     dtype=np.uint64)
>>> r = np.uint64(g.shape[0]) 
>>> n = np.uint64(8) 
>>> d = np.uint64(g.shape[1])
>>> x = np.empty((r,n,d),dtype=np.float64)
>>> time_perf,time_process = qmcseqcl.lattice_linear(r,n,d,g,x,**kwargs)
>>> x
array([[[0.   , 0.   , 0.   , 0.   , 0.   ],
        [0.125, 0.625, 0.125, 0.625, 0.625],
        [0.25 , 0.25 , 0.25 , 0.25 , 0.25 ],
        [0.375, 0.875, 0.375, 0.875, 0.875],
        [0.5  , 0.5  , 0.5  , 0.5  , 0.5  ],
        [0.625, 0.125, 0.625, 0.125, 0.125],
        [0.75 , 0.75 , 0.75 , 0.75 , 0.75 ],
        [0.875, 0.375, 0.875, 0.375, 0.375]],

       [[0.   , 0.   , 0.   , 0.   , 0.   ],
        [0.125, 0.375, 0.375, 0.125, 0.125],
        [0.25 , 0.75 , 0.75 , 0.25 , 0.25 ],
        [0.375, 0.125, 0.125, 0.375, 0.375],
        [0.5  , 0.5  , 0.5  , 0.5  , 0.5  ],
        [0.625, 0.875, 0.875, 0.625, 0.625],
        [0.75 , 0.25 , 0.25 , 0.75 , 0.75 ],
        [0.875, 0.625, 0.625, 0.875, 0.875]]])
```

## generate Gray code or natural order

```python 
>>> print(qmcseqcl.lattice_b2.__doc__)
Lattice points in Gray code or natural order

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    n_start (np.uint64): starting index in sequence
    gc (np.uint8): flag to use Gray code or natural order
    g (np.ndarray of np.uint64): pointer to generating vector of size r*d 
    x (np.ndarray of np.double): pointer to point storage of size r*n*d
>>> n_start = np.uint64(2)
>>> n = np.uint64(6) 
>>> x = np.empty((r,n,d),dtype=np.float64)
>>> gc = np.uint8(False)
>>> time_perf,time_process = qmcseqcl.lattice_b2(r,n,d,n_start,gc,g,x,**kwargs)
>>> x
array([[[0.25 , 0.25 , 0.25 , 0.25 , 0.25 ],
        [0.75 , 0.75 , 0.75 , 0.75 , 0.75 ],
        [0.125, 0.625, 0.125, 0.625, 0.625],
        [0.625, 0.125, 0.625, 0.125, 0.125],
        [0.375, 0.875, 0.375, 0.875, 0.875],
        [0.875, 0.375, 0.875, 0.375, 0.375]],

       [[0.25 , 0.75 , 0.75 , 0.25 , 0.25 ],
        [0.75 , 0.25 , 0.25 , 0.75 , 0.75 ],
        [0.125, 0.375, 0.375, 0.125, 0.125],
        [0.625, 0.875, 0.875, 0.625, 0.625],
        [0.375, 0.125, 0.125, 0.375, 0.375],
        [0.875, 0.625, 0.625, 0.875, 0.875]]])
>>> gc = np.uint8(True)
>>> time_perf,time_process = qmcseqcl.lattice_b2(r,n,d,n_start,gc,g,x,**kwargs)
>>> x
array([[[0.75 , 0.75 , 0.75 , 0.75 , 0.75 ],
        [0.25 , 0.25 , 0.25 , 0.25 , 0.25 ],
        [0.375, 0.875, 0.375, 0.875, 0.875],
        [0.875, 0.375, 0.875, 0.375, 0.375],
        [0.625, 0.125, 0.625, 0.125, 0.125],
        [0.125, 0.625, 0.125, 0.625, 0.625]],

       [[0.75 , 0.25 , 0.25 , 0.75 , 0.75 ],
        [0.25 , 0.75 , 0.75 , 0.25 , 0.25 ],
        [0.375, 0.125, 0.125, 0.375, 0.375],
        [0.875, 0.625, 0.625, 0.875, 0.875],
        [0.625, 0.875, 0.875, 0.625, 0.625],
        [0.125, 0.375, 0.375, 0.125, 0.125]]])
```

## shift mod 1 

```python
>>> print(qmcseqcl.lattice_shift.__doc__)
Shift mod 1 for lattice points

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    r_x (np.uint64): replications in x
    x (np.ndarray of np.double): lattice points of size r_x*n*d
    shifts (np.ndarray of np.double): shifts of size r*d
    xr (np.ndarray of np.double): pointer to point storage of size r*n*d
>>> r_x = r 
>>> r = 2*r_x
>>> shifts = rng.random((r,d))
>>> xr = np.empty((r,n,d),dtype=np.float64)
>>> time_perf,time_process = qmcseqcl.lattice_shift(r,n,d,r_x,x,shifts,xr,**kwargs)
>>> xr
array([[[0.79386058, 0.33727432, 0.1191824 , 0.40212985, 0.44669968],
        [0.29386058, 0.83727432, 0.6191824 , 0.90212985, 0.94669968],
        [0.41886058, 0.46227432, 0.7441824 , 0.52712985, 0.57169968],
        [0.91886058, 0.96227432, 0.2441824 , 0.02712985, 0.07169968],
        [0.66886058, 0.71227432, 0.9941824 , 0.77712985, 0.82169968],
        [0.16886058, 0.21227432, 0.4941824 , 0.27712985, 0.32169968]],

       [[0.85605352, 0.88025643, 0.38630282, 0.3468363 , 0.8076251 ],
        [0.35605352, 0.38025643, 0.88630282, 0.8468363 , 0.3076251 ],
        [0.48105352, 0.75525643, 0.26130282, 0.9718363 , 0.4326251 ],
        [0.98105352, 0.25525643, 0.76130282, 0.4718363 , 0.9326251 ],
        [0.73105352, 0.50525643, 0.01130282, 0.2218363 , 0.6826251 ],
        [0.23105352, 0.00525643, 0.51130282, 0.7218363 , 0.1826251 ]],

       [[0.9528797 , 0.97909681, 0.8866783 , 0.50220658, 0.59501765],
        [0.4528797 , 0.47909681, 0.3866783 , 0.00220658, 0.09501765],
        [0.5778797 , 0.10409681, 0.5116783 , 0.62720658, 0.72001765],
        [0.0778797 , 0.60409681, 0.0116783 , 0.12720658, 0.22001765],
        [0.8278797 , 0.35409681, 0.7616783 , 0.87720658, 0.97001765],
        [0.3278797 , 0.85409681, 0.2616783 , 0.37720658, 0.47001765]],

       [[0.31269008, 0.29826852, 0.96308655, 0.55983568, 0.60383675],
        [0.81269008, 0.79826852, 0.46308655, 0.05983568, 0.10383675],
        [0.93769008, 0.17326852, 0.83808655, 0.18483568, 0.22883675],
        [0.43769008, 0.67326852, 0.33808655, 0.68483568, 0.72883675],
        [0.18769008, 0.92326852, 0.58808655, 0.43483568, 0.47883675],
        [0.68769008, 0.42326852, 0.08808655, 0.93483568, 0.97883675]]])
```

# Digital Net Base 2 

## LSB to MSB integer representations 

convert generating matrices from least significant bit (LSB) order to most significant bit (MSB) order

```python
>>> print(qmcseqcl.gen_mats_lsb_to_msb_b2.__doc__)
Convert base 2 generating matrices with integers stored in Least Significant Bit order to Most Significant Bit order

Args:
    r (np.uint64): replications
    d (np.uint64): dimension
    mmax (np.uint64): columns in each generating matrix 
    tmaxes (np.ndarray of np.uint64): length r vector of bits in each integer of the resulting MSB generating matrices
    C_lsb (np.ndarray of np.uint64): original generating matrices of size r*d*mmax
    C_msb (np.ndarray of np.uint64): new generating matrices of size r*d*mmax
>>> C_lsb = np.array([
...     [1,2,4,8],
...     [1,3,5,15],
...     [1,3,6,9],
...     [1,3,4,10]],
...     dtype=np.uint64)
>>> r = np.uint64(1)
>>> d = np.uint64(C_lsb.shape[0])
>>> mmax = np.uint64(C_lsb.shape[1])
>>> tmax = 4
>>> tmaxes = np.tile(np.uint64(tmax),r)
>>> C = np.empty((d,mmax),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.gen_mats_lsb_to_msb_b2(r,d,mmax,tmaxes,C_lsb,C,**kwargs)
>>> C
array([[ 8,  4,  2,  1],
       [ 8, 12, 10, 15],
       [ 8, 12,  6,  9],
       [ 8, 12,  2,  5]], dtype=uint64)
```

## linear matrix scrambling (LMS)

```python
>>> print(qmcseqcl.linear_matrix_scramble_digital_net_b2.__doc__)
Linear matrix scrambling for base 2 generating matrices

Args:
    r (np.uint64): replications
    d (np.uint64): dimension
    mmax (np.uint64): columns in each generating matrix 
    r_C (np.uint64): original generating matrices
    tmax_new (np.uint64): bits in the integers of the resulting generating matrices
    tmaxes (np.ndarray of np.uint64): bits in the integers of the original generating matrices
    S (np.ndarray of np.uint64): scrambling matrices of size r*d*tmax_new
    C (np.ndarray of np.uint64): original generating matrices of size r_C*d*mmax
    C_lms (np.ndarray of np.uint64): resulting generating matrices of size r*d*mmax
>>> r_C = r 
>>> r = 2*r_C
>>> tmax_new = np.uint64(8)
>>> S = rng.integers(0,np.uint64(1)<<np.minimum(np.arange(tmax_new,dtype=np.uint64),tmax),(r,d,tmax_new),dtype=np.uint64)
>>> S[:,:,:tmax] <<= np.arange(tmax,0,-1,dtype=np.uint64)
>>> S[:,:,:tmax] += np.uint64(1)<<np.arange(tmax-1,-1,-1,dtype=np.uint64)
>>> for l in range(r):
...     print("l = %d"%l)
...     for j in range(d): 
...         print("    j = %d"%j)
...         for t in range(tmax_new):
...             b = bin(S[l,j,t])[2:]
...             print("        "+"0"*(tmax-len(b))+b)
l = 0
    j = 0
        1000
        1100
        1110
        0011
        1001
        0011
        0001
        0000
    j = 1
        1000
        0100
        1010
        1001
        0011
        1001
        0011
        0101
    j = 2
        1000
        0100
        0110
        1011
        1111
        1001
        1000
        0100
    j = 3
        1000
        1100
        0110
        0011
        0000
        0101
        0001
        0010
l = 1
    j = 0
        1000
        1100
        1010
        1101
        1000
        1011
        0111
        0111
    j = 1
        1000
        0100
        1010
        1001
        1110
        0000
        0110
        0100
    j = 2
        1000
        0100
        1010
        1011
        1100
        1110
        0000
        0000
    j = 3
        1000
        0100
        0010
        1111
        1111
        1111
        1101
        0111
>>> C_lms = np.empty((r,d,mmax),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.linear_matrix_scramble_digital_net_b2(r,d,mmax,r_C,tmax_new,tmaxes,S,C,C_lms,**kwargs)
>>> C_lms
array([[[232,  96,  52,  30],
        [180, 245, 158, 192],
        [158, 247,  81, 130],
        [192, 164,  49, 114]],

       [[252,  83,  39,  23],
        [184, 243, 146, 201],
        [188, 240, 120, 172],
        [158, 193,  61,  64]]], dtype=uint64)
```

## digital interlacing 

Digital interlacing is used to create **higher order digital nets**.

```python
>>> print(qmcseqcl.interlace_b2.__doc__)
Interlace generating matrices or transpose of point sets to attain higher order digital nets in base 2

Args:
    r (np.uint64): replications
    d_alpha (np.uint64): dimension of resulting generating matrices 
    mmax (np.uint64): columns of generating matrices
    d (np.uint64): dimension of original generating matrices
    tmax (np.uint64): bits in integers of original generating matrices
    tmax_alpha (np.uint64): bits in integers of resulting generating matrices
    alpha (np.uint64): interlacing factor
    C (np.ndarray of np.uint64): original generating matrices of size r*d*mmax
    C_alpha (np.ndarray of np.uint64): resulting interlaced generating matrices of size r*d_alpha*mmax
>>> alpha = np.uint64(2) 
>>> d_alpha = np.uint64(d//alpha)
>>> tmax = tmax_new
>>> tmax_alpha = min(alpha*tmax_new,64)
>>> C_alpha = np.empty((r,d_alpha,mmax),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.interlace_b2(r,d_alpha,mmax,d,tmax,tmax_alpha,alpha,C_lms,C_alpha,**kwargs)
>>> C_alpha
array([[[60816, 32017, 19316, 21160],
        [53928, 60986,  9987, 38156]],

       [[61408, 30479, 18734, 21099],
        [52212, 64001, 12241, 39072]]], dtype=uint64)
``` 

## undo digital interlacing 

```python
>>> print(qmcseqcl.undo_interlace_b2.__doc__)
Undo interlacing of generating matrices

Args:
    r (np.uint64): replications
    d (np.uint64): dimension of resulting generating matrices 
    mmax (np.uint64): columns in generating matrices
    d_alpha (np.uint64): dimension of interlaced generating matrices
    tmax (np.uint64): bits in integers of original generating matrices 
    tmax_alpha (np.uint64): bits in integers of interlaced generating matrices
    alpha (np.uint64): interlacing factor
    C_alpha (np.ndarray of np.uint64): interlaced generating matrices of size r*d_alpha*mmax
    C (np.ndarray of np.uint64): original generating matrices of size r*d*mmax
>>> C_lms_cp = np.empty((r,d,mmax),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.undo_interlace_b2(r,d,mmax,d_alpha,tmax,tmax_alpha,alpha,C_alpha,C_lms_cp,**kwargs)
>>> assert (C_lms_cp==C_lms).all()
```

## generate Gray code or natural order 

```python
>>> print(qmcseqcl.digital_net_b2_binary.__doc__)
Binary representation of digital net in base 2 in either Gray code or natural order

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    n_start (np.uint64): starting index in sequence
    gc (np.uint8): flag to use Gray code or natural order
    mmax (np.uint64): columns in each generating matrix
    C (np.ndarray of np.uint64): generating matrices of size r*d*mmax
    xb (np.ndarray of np.uint64): binary digital net points of size r*n*d
>>> C = C_alpha
>>> d = d_alpha
>>> n_start = np.uint64(2)
>>> n = np.uint64(14)
>>> xb = np.empty((r,n,d),dtype=np.uint64)
>>> gc = np.uint8(False)
>>> time_perf,time_process = qmcseqcl.digital_net_b2_binary(r,n,d,n_start,gc,mmax,C,xb,**kwargs)
>>> xb
array([[[32017, 60986],
        [36993, 15506],
        [19316,  9987],
        [42724, 62891],
        [13925, 51513],
        [56309,  7057],
        [21160, 38156],
        [48952, 18340],
        [12217, 31542],
        [49705, 43422],
        [ 6620, 45583],
        [62540, 24743],
        [25805, 23605],
        [35165, 36509]],

       [[30479, 64001],
        [39151, 12789],
        [18734, 12241],
        [42702, 58405],
        [15905, 54736],
        [53697,  7716],
        [21099, 39072],
        [48523, 21332],
        [ 9572, 25249],
        [51844, 43349],
        [ 6981, 46961],
        [62629, 31877],
        [27722, 19824],
        [33706, 34436]]], dtype=uint64)
>>> gc = np.uint8(True)
>>> time_perf,time_process = qmcseqcl.digital_net_b2_binary(r,n,d,n_start,gc,mmax,C,xb,**kwargs)
>>> xb
array([[[36993, 15506],
        [32017, 60986],
        [13925, 51513],
        [56309,  7057],
        [42724, 62891],
        [19316,  9987],
        [ 6620, 45583],
        [62540, 24743],
        [35165, 36509],
        [25805, 23605],
        [12217, 31542],
        [49705, 43422],
        [48952, 18340],
        [21160, 38156]],

       [[39151, 12789],
        [30479, 64001],
        [15905, 54736],
        [53697,  7716],
        [42702, 58405],
        [18734, 12241],
        [ 6981, 46961],
        [62629, 31877],
        [33706, 34436],
        [27722, 19824],
        [ 9572, 25249],
        [51844, 43349],
        [48523, 21332],
        [21099, 39072]]], dtype=uint64)
```

## digital shift 

```python
>>> print(qmcseqcl.digital_net_b2_digital_shift.__doc__)
Digital shift base 2 digital net

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    r_x (np.uint64): replications of xb
    lshifts (np.ndarray of np.uint64): left shift applied to each element of xb
    xb (np.ndarray of np.uint64): binary base 2 digital net points of size r_x*n*d
    shiftsb (np.ndarray of np.uint64): digital shifts of size r*d
    xrb (np.ndarray of np.uint64): digital shifted digital net points of size r*n*d
>>> tmax = tmax_alpha 
>>> tmax_new = np.uint64(64)
>>> lshifts = np.tile(tmax_new-tmax,r) 
>>> r_x = r 
>>> r = 2*r_x
>>> print(qmcseqcl.random_tbit_uint64s.__doc__)
Generate the desired shape of random integers with t bits

    Args:
        rng (np.random._generator.Generator): random number generator with rng.integers method
        t: (int): number of bits with 0 <= t <= 64
        shape (tuple of ints): shape of resulting integer array
>>> shiftsb = qmcseqcl.random_tbit_uint64s(rng,tmax_new,(r,d))
>>> shiftsb
array([[1440145505151606152, 5686212125327047696],
       [8845106632000742233, 2846419807334012155],
       [1319978926241325836, 1435914391012611701],
       [1035738700154284414, 5954637325829729870]], dtype=uint64)
>>> xrb = np.empty((r,n,d),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.digital_net_b2_digital_shift(r,n,d,r_x,lshifts,xb,shiftsb,xrb,**kwargs)
>>> xrb
array([[[ 9474848715357281672,  8249323263254281232],
        [ 7993164437952388488, 11588742386949504016],
        [ 2709316175139954056,  9786458111071173648],
        [14414171606675873160,  6158808611224239120],
        [13049299444605902216, 13493483554350513168],
        [ 6379468396470197640,  7632048639327812624],
        [  729702663933910408, 18223389062996376592],
        [16694963342962318728,  3336740494723202064],
        [11142306477391207816, 13867845273375685648],
        [ 8588765488672136584,  1359097308354133008],
        [ 4342996939968601480,  3881957524611742736],
        [15120110848266198408, 16678935865785007120],
        [12449194794258783624,   670328040343157776],
        [ 4707507034808901000, 15845206984768044048]],

       [[16298282540322202457,  1618344483945420027],
        [  995051006517257049, 15961183397213607163],
        [ 4963285228184085337, 17460600598151271675],
        [12322167019307475801,  4153589599178298619],
        [15856648301863183193, 14097819051389064443],
        [ 3741965304236548953,   599405068252845307],
        [ 7027059757426615129, 10444273853684749563],
        [10260644289878631257,  6558793275170854139],
        [17972214226820473689, 11602543382849099003],
        [ 1624147579465573209,  7705803805266777339],
        [ 6891670293628789593,  4981407505684337915],
        [12701313812936729433, 10292277366260995323],
        [14360608800646046553,  8418498446298158331],
        [ 2930472946379727705, 13772152503334835451]],

       [[ 9426176780531507980,  3422564776636421749],
        [ 8016550097164542732, 18291198946400114293],
        [ 2608852844599419660, 15768338730142504565],
        [14529881058249122572,   611474184227100277],
        [13021456658056717068, 16593060411904726645],
        [ 6423683203958940428,  3814096469240944245],
        [  832464266578469644, 11664996519654561397],
        [16581552163493094156,  8307562997449856629],
        [11172447536044417804, 11344677996157834869],
        [ 8546848953287418636,  5753459058777364085],
        [ 4460958191355536140,  7555743334655694453],
        [15021899317539349260, 13435192648187876981],
        [12495661401561211660,  6073496107297379957],
        [ 4681916048073401100,  9719160005653796469]],

       [[10858370962426046846,  7157942851267784270],
        [ 8741679137561913726, 12151308938114821710],
        [ 3494422621721864574,  9760179010957798990],
        [16113508777613994366,  5514410462254263886],
        [12146681930830719358, 13152233955297914446],
        [ 5148088109896968574,  9039321595601808974],
        [ 1520720085026744702, 16560332973310537294],
        [18084959514495428990,  3325379568375492174],
        [10229274389477730686, 15287221653648240206],
        [ 7067747451063642494,  2293210828777516622],
        [ 3115275828092610942,  3459361657289764430],
        [14185123712169290110, 18155733141306535502],
        [12958174288687540606,   141616106801262158],
        [ 6644127611114105214, 14556512614107377230]]], dtype=uint64)
```

## convert digits to doubles

```python 
>>> print(qmcseqcl.digital_net_b2_from_binary.__doc__)
Convert base 2 binary digital net points to floats

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    tmaxes (np.ndarray of np.uint64): bits in integers of each generating matrix of size r
    xb (np.ndarray of np.uint64): binary digital net points of size r*n*d
    x (np.ndarray of np.double): float digital net points of size r*n*d
>>> x = np.empty((r,n,d),dtype=np.float64)
>>> tmaxes_new = np.tile(tmax_new,r)
>>> time_perf,time_process = qmcseqcl.digital_net_b2_from_binary(r,n,d,tmaxes_new,xrb,x,**kwargs)
>>> x
array([[[0.51363258, 0.44719671],
        [0.43331031, 0.62822698],
        [0.14687232, 0.53052496],
        [0.78139381, 0.33386968],
        [0.70740394, 0.73148321],
        [0.34583167, 0.41373419],
        [0.03955726, 0.9878919 ],
        [0.90503578, 0.18088506],
        [0.60402564, 0.7517774 ],
        [0.46559791, 0.07367681],
        [0.23543434, 0.21044134],
        [0.81966285, 0.90416693],
        [0.6748722 , 0.03633856],
        [0.25519447, 0.85897039]],

       [[0.88353167, 0.08773063],
        [0.05394182, 0.86525749],
        [0.26906023, 0.94654106],
        [0.66798601, 0.22516654],
        [0.85959063, 0.7642443 ],
        [0.20285235, 0.03249381],
        [0.38093767, 0.56618522],
        [0.55623064, 0.3555529 ],
        [0.97427569, 0.62897514],
        [0.08804522, 0.41773246],
        [0.3735982 , 0.27004264],
        [0.6885396 , 0.55794547],
        [0.77849016, 0.45636772],
        [0.15886126, 0.74658988]],

       [[0.51099407, 0.18553761],
        [0.43457805, 0.99156788],
        [0.1414262 , 0.85480336],
        [0.78766643, 0.03314808],
        [0.70589458, 0.89951161],
        [0.34822856, 0.20676258],
        [0.04512798, 0.63236073],
        [0.89888774, 0.45035389],
        [0.60565959, 0.61499623],
        [0.46332561, 0.31189564],
        [0.24182903, 0.40959767],
        [0.81433879, 0.72832325],
        [0.67739116, 0.32924488],
        [0.25380718, 0.52687672]],

       [[0.58863347, 0.38803286],
        [0.47388738, 0.65872378],
        [0.18943303, 0.52910036],
        [0.87351506, 0.29893679],
        [0.65847295, 0.71298403],
        [0.27907842, 0.49002261],
        [0.0824384 , 0.89773745],
        [0.98038762, 0.18026919],
        [0.55453008, 0.82872195],
        [0.38314336, 0.12431521],
        [0.16887944, 0.18753237],
        [0.7689771 , 0.98422427],
        [0.70246404, 0.00767702],
        [0.36017888, 0.78911013]]])
```

# Generalized Digital Net 

Accommodates both Halton sequences and digital nets in any prime base

## Linear Matrix Scramble 

```python
>>> print(qmcseqcl.linear_matrix_scramble_generalized_digital_net.__doc__)
Linear matrix scramble for generalized digital net

Args:
    r (np.uint64): replications 
    d (np.uint64): dimension 
    mmax (np.uint64): columns in each generating matrix
    r_C (np.uint64): number of replications of C 
    r_b (np.uint64): number of replications of bases
    tmax (np.uint64): number of rows in each generating matrix 
    tmax_new (np.uint64): new number of rows in each generating matrix 
    bases (np.ndarray of np.uint64): bases for each dimension of size r*d 
    S (np.ndarray of np.uint64): scramble matrices of size r*d*tmax_new*tmax
    C (np.ndarray of np.uint64): generating matrices of size r_C*d*mmax*tmax 
    C_lms (np.ndarray of np.uint64): new generating matrices of size r*d*mmax*tmax_new
>>> bases = np.array([[2,3],[5,7]],dtype=np.uint64)
>>> r_C = r_b = np.uint64(bases.shape[0])
>>> d = np.uint64(bases.shape[1]) 
>>> mmax = np.uint64(5)
>>> tmax = mmax
>>> C = np.tile(np.eye(mmax,dtype=np.uint64)[None,None,:,:],(r_C,d,1,1))
>>> tmax_new = np.uint64(2*tmax)
>>> r = np.uint64(2*r_C)
>>> S = np.empty((r,d,tmax_new,tmax),dtype=np.uint64)
>>> lower_flag = np.tri(int(tmax_new),int(tmax),k=-1,dtype=np.bool)
>>> n_lower_flags = lower_flag.sum()
>>> diag_flag = np.eye(tmax_new,tmax,dtype=np.bool)
>>> for l in range(r):
...     for j in range(d):
...         b = bases[l%r_b,j]
...         Slj = np.zeros((tmax_new,tmax),dtype=np.uint64)
...         Slj[lower_flag] = rng.integers(0,b,n_lower_flags)
...         Slj[diag_flag] = rng.integers(1,b,tmax)
...         S[l,j] = Slj
>>> S
array([[[[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 1, 1],
         [0, 1, 0, 1, 1],
         [1, 0, 1, 0, 0],
         [1, 0, 1, 0, 0],
         [1, 1, 1, 0, 1],
         [0, 1, 1, 0, 1]],

        [[2, 0, 0, 0, 0],
         [2, 1, 0, 0, 0],
         [1, 0, 1, 0, 0],
         [2, 2, 0, 2, 0],
         [2, 0, 1, 0, 1],
         [2, 2, 0, 1, 2],
         [0, 0, 1, 0, 2],
         [0, 0, 2, 0, 1],
         [2, 1, 2, 1, 0],
         [1, 2, 2, 2, 2]]],


       [[[3, 0, 0, 0, 0],
         [2, 1, 0, 0, 0],
         [0, 4, 4, 0, 0],
         [2, 3, 3, 4, 0],
         [3, 1, 3, 3, 1],
         [3, 1, 2, 0, 4],
         [3, 1, 1, 0, 3],
         [0, 4, 1, 0, 4],
         [1, 0, 0, 3, 2],
         [3, 3, 1, 2, 2]],

        [[1, 0, 0, 0, 0],
         [3, 2, 0, 0, 0],
         [5, 1, 1, 0, 0],
         [5, 2, 0, 2, 0],
         [4, 3, 1, 4, 4],
         [0, 0, 1, 3, 6],
         [0, 4, 3, 6, 2],
         [0, 2, 1, 1, 6],
         [6, 4, 1, 4, 0],
         [0, 0, 1, 5, 6]]],


       [[[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 1, 1, 1, 1],
         [1, 0, 0, 0, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 0, 1]],

        [[2, 0, 0, 0, 0],
         [1, 2, 0, 0, 0],
         [0, 2, 2, 0, 0],
         [2, 0, 0, 1, 0],
         [2, 0, 2, 2, 1],
         [1, 0, 2, 0, 2],
         [0, 1, 0, 2, 0],
         [0, 2, 0, 1, 0],
         [1, 2, 0, 2, 0],
         [0, 2, 0, 1, 0]]],


       [[[4, 0, 0, 0, 0],
         [3, 4, 0, 0, 0],
         [3, 3, 1, 0, 0],
         [4, 4, 1, 3, 0],
         [2, 4, 2, 1, 1],
         [4, 0, 1, 3, 1],
         [3, 3, 1, 4, 2],
         [4, 2, 0, 4, 1],
         [4, 1, 1, 3, 3],
         [3, 2, 3, 4, 4]],

        [[4, 0, 0, 0, 0],
         [1, 4, 0, 0, 0],
         [5, 2, 6, 0, 0],
         [3, 5, 6, 2, 0],
         [0, 3, 6, 0, 6],
         [2, 5, 5, 6, 1],
         [1, 1, 2, 0, 3],
         [5, 6, 3, 3, 2],
         [2, 5, 4, 2, 5],
         [5, 6, 4, 3, 3]]]], dtype=uint64)
>>> C_lms = np.empty((r,d,mmax,tmax_new),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.linear_matrix_scramble_generalized_digital_net(r,d,mmax,r_C,r_b,tmax,tmax_new,bases,S,C,C_lms,**kwargs)
>>> C_lms 
array([[[[1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
         [0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
         [0, 0, 1, 0, 1, 0, 1, 1, 1, 1],
         [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 1, 1]],

        [[2, 2, 1, 2, 2, 2, 0, 0, 2, 1],
         [0, 1, 0, 2, 0, 2, 0, 0, 1, 2],
         [0, 0, 1, 0, 1, 0, 1, 2, 2, 2],
         [0, 0, 0, 2, 0, 1, 0, 0, 1, 2],
         [0, 0, 0, 0, 1, 2, 2, 1, 0, 2]]],


       [[[3, 2, 0, 2, 3, 3, 3, 0, 1, 3],
         [0, 1, 4, 3, 1, 1, 1, 4, 0, 3],
         [0, 0, 4, 3, 3, 2, 1, 1, 0, 1],
         [0, 0, 0, 4, 3, 0, 0, 0, 3, 2],
         [0, 0, 0, 0, 1, 4, 3, 4, 2, 2]],

        [[1, 3, 5, 5, 4, 0, 0, 0, 6, 0],
         [0, 2, 1, 2, 3, 0, 4, 2, 4, 0],
         [0, 0, 1, 0, 1, 1, 3, 1, 1, 1],
         [0, 0, 0, 2, 4, 3, 6, 1, 4, 5],
         [0, 0, 0, 0, 4, 6, 2, 6, 0, 6]]],


       [[[1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
         [0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
         [0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
         [0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]],

        [[2, 1, 0, 2, 2, 1, 0, 0, 1, 0],
         [0, 2, 2, 0, 0, 0, 1, 2, 2, 2],
         [0, 0, 2, 0, 2, 2, 0, 0, 0, 0],
         [0, 0, 0, 1, 2, 0, 2, 1, 2, 1],
         [0, 0, 0, 0, 1, 2, 0, 0, 0, 0]]],


       [[[4, 3, 3, 4, 2, 4, 3, 4, 4, 3],
         [0, 4, 3, 4, 4, 0, 3, 2, 1, 2],
         [0, 0, 1, 1, 2, 1, 1, 0, 1, 3],
         [0, 0, 0, 3, 1, 3, 4, 4, 3, 4],
         [0, 0, 0, 0, 1, 1, 2, 1, 3, 4]],

        [[4, 1, 5, 3, 0, 2, 1, 5, 2, 5],
         [0, 4, 2, 5, 3, 5, 1, 6, 5, 6],
         [0, 0, 6, 6, 6, 5, 2, 3, 4, 4],
         [0, 0, 0, 2, 0, 6, 0, 3, 2, 3],
         [0, 0, 0, 0, 6, 1, 3, 2, 5, 3]]]], dtype=uint64)
```

## generate natural order

```python
>>> print(qmcseqcl.generalized_digital_net_digits.__doc__)
Generalized digital net where the base can be different for each dimension e.g. for the Halton sequence

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    r_b (np.uint64): number of replications of bases
    mmax (np.uint64): columns in each generating matrix
    tmax (np.uint64): rows of each generating matrix
    n_start (np.uint64): starting index in sequence
    bases (np.ndarray of np.uint64): bases for each dimension of size r_b*d
    C (np.ndarray of np.uint64): generating matrices of size r*d*mmax*tmax
    xdig (np.ndarray of np.uint64): generalized digital net sequence of digits of size r*n*d*tmax
>>> tmax = tmax_new
>>> C = C_lms
>>> n_start = np.uint64(2)
>>> n = np.uint64(6)
>>> xdig = np.empty((r,n,d,tmax),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.generalized_digital_net_digits(r,n,d,r_b,mmax,tmax,n_start,bases,C,xdig,**kwargs)
>>> xdig
array([[[[0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
         [1, 1, 2, 1, 1, 1, 0, 0, 1, 2]],

        [[1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
         [2, 0, 0, 2, 1, 0, 1, 0, 2, 0]],

        [[0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
         [1, 2, 1, 1, 0, 2, 1, 0, 1, 1]],

        [[1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
         [0, 1, 2, 0, 2, 1, 1, 0, 0, 2]],

        [[0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
         [1, 0, 0, 1, 2, 0, 2, 0, 1, 0]],

        [[1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
         [0, 2, 1, 0, 1, 2, 2, 0, 0, 1]]],


       [[[1, 4, 0, 4, 1, 1, 1, 0, 2, 1],
         [2, 6, 3, 3, 1, 0, 0, 0, 5, 0]],

        [[4, 1, 0, 1, 4, 4, 4, 0, 3, 4],
         [3, 2, 1, 1, 5, 0, 0, 0, 4, 0]],

        [[2, 3, 0, 3, 2, 2, 2, 0, 4, 2],
         [4, 5, 6, 6, 2, 0, 0, 0, 3, 0]],

        [[3, 3, 0, 1, 3, 0, 1, 4, 3, 1],
         [5, 1, 4, 4, 6, 0, 0, 0, 2, 0]],

        [[1, 0, 0, 3, 1, 3, 4, 4, 4, 4],
         [6, 4, 2, 2, 3, 0, 0, 0, 1, 0]],

        [[4, 2, 0, 0, 4, 1, 2, 4, 0, 2],
         [0, 0, 0, 6, 0, 0, 2, 1, 2, 3]]],


       [[[0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
         [1, 2, 0, 1, 1, 2, 0, 0, 2, 0]],

        [[1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
         [1, 0, 0, 1, 0, 0, 2, 2, 0, 0]],

        [[0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
         [0, 1, 0, 0, 2, 1, 2, 2, 1, 0]],

        [[1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
         [2, 2, 0, 2, 1, 2, 2, 2, 2, 0]],

        [[0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
         [2, 0, 0, 2, 0, 0, 1, 1, 0, 0]],

        [[1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
         [1, 1, 0, 1, 2, 1, 1, 1, 1, 0]]],


       [[[3, 1, 1, 3, 4, 3, 1, 3, 3, 1],
         [1, 2, 3, 6, 0, 4, 2, 3, 4, 3]],

        [[2, 4, 4, 2, 1, 2, 4, 2, 2, 4],
         [5, 3, 1, 2, 0, 6, 3, 1, 6, 1]],

        [[1, 2, 2, 1, 3, 1, 2, 1, 1, 2],
         [2, 4, 6, 5, 0, 1, 4, 6, 1, 6]],

        [[4, 3, 4, 4, 3, 0, 4, 3, 4, 4],
         [6, 5, 4, 1, 0, 3, 5, 4, 3, 4]],

        [[3, 1, 2, 3, 0, 4, 2, 2, 3, 2],
         [3, 6, 2, 4, 0, 5, 6, 2, 5, 2]],

        [[2, 4, 0, 2, 2, 3, 0, 1, 2, 0],
         [2, 1, 5, 2, 5, 0, 4, 2, 5, 3]]]], dtype=uint64)
```

## digital shift 

```python
>>> print(qmcseqcl.generalized_digital_net_digital_shift.__doc__)
Digital shift a generalized digital net

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    r_x (np.uint64): replications of xdig
    r_b (np.uint64): replications of bases
    tmax (np.uint64): rows of each generating matrix
    tmax_new (np.uint64): rows of each new generating matrix
    bases (np.ndarray of np.uint64): bases for each dimension of size r_b*d
    shifts (np.ndarray of np.uint64): digital shifts of size r*d*tmax_new
    xdig (np.ndarray of np.uint64): binary digital net points of size r_x*n*d*tmax
    xdig_new (np.ndarray of np.uint64): float digital net points of size r*n*d*tmax_new
>>> tmax_new = np.uint64(12)
>>> shifts = np.empty((r,d,tmax_new),dtype=np.uint64)
>>> for l in range(r):
...     for j in range(d):
...         b = bases[l%r_b,j]
...         shifts[l,j] = rng.integers(0,b,tmax_new,dtype=np.uint64)
>>> shifts
array([[[1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 2, 1, 1, 0, 1, 2, 2]],

       [[0, 1, 3, 1, 0, 4, 4, 4, 3, 3, 1, 3],
        [1, 4, 6, 4, 6, 2, 5, 0, 4, 2, 2, 1]],

       [[1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 1, 0, 1, 0, 0, 2, 2, 0, 1]],

       [[0, 3, 3, 2, 2, 0, 2, 1, 3, 2, 4, 1],
        [2, 3, 2, 4, 1, 0, 2, 5, 4, 2, 4, 3]]], dtype=uint64)
>>> xdig_new = np.empty((r,n,d,tmax_new),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.generalized_digital_net_digital_shift(r,n,d,r_x,r_b,tmax,tmax_new,bases,shifts,xdig,xdig_new,**kwargs)
>>> xdig_new
array([[[[1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
         [2, 1, 2, 2, 1, 0, 1, 1, 1, 0, 2, 2]],

        [[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 2, 2, 1, 2, 1, 2, 2]],

        [[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
         [2, 2, 1, 2, 0, 1, 2, 1, 1, 2, 2, 2]],

        [[0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
         [1, 1, 2, 1, 2, 0, 2, 1, 0, 0, 2, 2]],

        [[1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
         [2, 0, 0, 2, 2, 2, 0, 1, 1, 1, 2, 2]],

        [[0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [1, 2, 1, 1, 1, 1, 0, 1, 0, 2, 2, 2]]],


       [[[1, 0, 3, 0, 1, 0, 0, 4, 0, 4, 1, 3],
         [3, 3, 2, 0, 0, 2, 5, 0, 2, 2, 2, 1]],

        [[4, 2, 3, 2, 4, 3, 3, 4, 1, 2, 1, 3],
         [4, 6, 0, 5, 4, 2, 5, 0, 1, 2, 2, 1]],

        [[2, 4, 3, 4, 2, 1, 1, 4, 2, 0, 1, 3],
         [5, 2, 5, 3, 1, 2, 5, 0, 0, 2, 2, 1]],

        [[3, 4, 3, 2, 3, 4, 0, 3, 1, 4, 1, 3],
         [6, 5, 3, 1, 5, 2, 5, 0, 6, 2, 2, 1]],

        [[1, 1, 3, 4, 1, 2, 3, 3, 2, 2, 1, 3],
         [0, 1, 1, 6, 2, 2, 5, 0, 5, 2, 2, 1]],

        [[4, 3, 3, 1, 4, 0, 1, 3, 3, 0, 1, 3],
         [1, 4, 6, 3, 6, 2, 0, 1, 6, 5, 2, 1]]],


       [[[1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
         [2, 2, 0, 2, 1, 2, 0, 0, 0, 1, 0, 1]],

        [[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 1, 1, 1, 0, 1, 2, 0, 1]],

        [[1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
         [2, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 1]],

        [[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [1, 2, 0, 1, 2, 2, 1, 0, 2, 1, 0, 1]],

        [[1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
         [2, 1, 1, 2, 2, 1, 2, 0, 0, 2, 0, 1]],

        [[0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
         [1, 0, 2, 1, 1, 0, 2, 0, 2, 0, 0, 1]]],


       [[[1, 2, 3, 1, 3, 1, 3, 1, 0, 3, 4, 1],
         [4, 2, 5, 0, 2, 0, 2, 5, 2, 2, 4, 3]],

        [[4, 4, 3, 3, 1, 4, 1, 1, 1, 1, 4, 1],
         [5, 5, 3, 5, 6, 0, 2, 5, 1, 2, 4, 3]],

        [[2, 1, 3, 0, 4, 2, 4, 1, 2, 4, 4, 1],
         [6, 1, 1, 3, 3, 0, 2, 5, 0, 2, 4, 3]],

        [[3, 1, 3, 3, 0, 0, 3, 0, 1, 3, 4, 1],
         [0, 4, 6, 1, 0, 0, 2, 5, 6, 2, 4, 3]],

        [[1, 3, 3, 0, 3, 3, 1, 0, 2, 1, 4, 1],
         [1, 0, 4, 6, 4, 0, 2, 5, 5, 2, 4, 3]],

        [[4, 0, 3, 2, 1, 1, 4, 0, 3, 4, 4, 1],
         [2, 3, 2, 3, 1, 0, 4, 6, 6, 5, 4, 3]]]], dtype=uint64)
```

## digital permutation 

```python
>>> print(qmcseqcl.generalized_digital_net_permutation.__doc__)
Permutation of digits for a generalized digital net

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    r_x (np.uint64): replications of xdig
    r_b (np.uint64): replications of bases
    tmax (np.uint64): rows of each generating matrix
    tmax_new (np.uint64): rows of each new generating matrix
    bmax (np.uint64): common permutation size, typically the maximum basis
    perms (np.ndarray of np.uint64): permutations of size r*d*tmax_new*bmax
    xdig (np.ndarray of np.uint64): binary digital net points of size r_x*n*d*tmax
    xdig_new (np.ndarray of np.uint64): float digital net points of size r*n*d*tmax_new
>>> bmax = bases.max()
>>> perms = np.zeros((r,d,tmax_new,bmax),dtype=np.uint64)
>>> for l in range(r):
...     for j in range(d):
...         b = bases[l%r_b,j]
...         for t in range(tmax_new):
...             perms[l,j,t,:b] = rng.permutation(b)
>>> perms
array([[[[1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0]],

        [[2, 0, 1, 0, 0, 0, 0],
         [1, 0, 2, 0, 0, 0, 0],
         [1, 2, 0, 0, 0, 0, 0],
         [2, 1, 0, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0, 0],
         [2, 1, 0, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0],
         [1, 0, 2, 0, 0, 0, 0],
         [0, 2, 1, 0, 0, 0, 0],
         [1, 0, 2, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0, 0],
         [1, 2, 0, 0, 0, 0, 0]]],


       [[[1, 0, 3, 4, 2, 0, 0],
         [3, 2, 1, 4, 0, 0, 0],
         [1, 0, 3, 4, 2, 0, 0],
         [1, 3, 0, 4, 2, 0, 0],
         [0, 4, 1, 2, 3, 0, 0],
         [2, 4, 3, 1, 0, 0, 0],
         [2, 1, 4, 0, 3, 0, 0],
         [2, 0, 4, 1, 3, 0, 0],
         [1, 4, 2, 3, 0, 0, 0],
         [3, 1, 2, 0, 4, 0, 0],
         [3, 2, 4, 0, 1, 0, 0],
         [3, 2, 1, 4, 0, 0, 0]],

        [[6, 2, 0, 5, 1, 4, 3],
         [2, 0, 3, 5, 6, 4, 1],
         [0, 6, 2, 1, 4, 3, 5],
         [3, 2, 5, 4, 6, 0, 1],
         [1, 4, 3, 6, 5, 0, 2],
         [5, 6, 4, 2, 3, 0, 1],
         [3, 0, 1, 4, 2, 5, 6],
         [3, 2, 6, 1, 0, 4, 5],
         [1, 4, 3, 2, 5, 0, 6],
         [6, 0, 4, 1, 5, 3, 2],
         [6, 3, 4, 2, 5, 1, 0],
         [1, 6, 2, 3, 5, 0, 4]]],


       [[[1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0]],

        [[0, 1, 2, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0],
         [0, 2, 1, 0, 0, 0, 0],
         [2, 1, 0, 0, 0, 0, 0],
         [1, 2, 0, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0],
         [1, 0, 2, 0, 0, 0, 0],
         [1, 0, 2, 0, 0, 0, 0],
         [1, 2, 0, 0, 0, 0, 0],
         [2, 1, 0, 0, 0, 0, 0]]],


       [[[0, 1, 4, 3, 2, 0, 0],
         [4, 0, 2, 3, 1, 0, 0],
         [4, 1, 0, 2, 3, 0, 0],
         [1, 3, 2, 4, 0, 0, 0],
         [3, 2, 0, 4, 1, 0, 0],
         [2, 4, 0, 1, 3, 0, 0],
         [2, 3, 1, 4, 0, 0, 0],
         [4, 0, 2, 1, 3, 0, 0],
         [0, 1, 3, 4, 2, 0, 0],
         [2, 0, 1, 3, 4, 0, 0],
         [0, 4, 1, 2, 3, 0, 0],
         [1, 0, 4, 2, 3, 0, 0]],

        [[4, 5, 0, 3, 1, 6, 2],
         [4, 2, 1, 5, 3, 0, 6],
         [3, 1, 4, 5, 6, 0, 2],
         [0, 1, 3, 6, 2, 4, 5],
         [4, 0, 2, 6, 1, 5, 3],
         [0, 6, 4, 2, 1, 3, 5],
         [5, 1, 3, 4, 0, 6, 2],
         [4, 6, 0, 1, 3, 5, 2],
         [3, 2, 6, 5, 0, 4, 1],
         [3, 4, 2, 6, 1, 5, 0],
         [2, 6, 1, 3, 0, 4, 5],
         [2, 4, 0, 6, 1, 5, 3]]]], dtype=uint64)
>>> time_perf,time_process = qmcseqcl.generalized_digital_net_permutation(r,n,d,r_x,r_b,tmax,tmax_new,bmax,perms,xdig,xdig_new,**kwargs)
>>> xdig_new
array([[[[1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0, 1, 0, 1, 2, 2, 2, 1]],

        [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
         [1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 2, 1]],

        [[1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
         [0, 2, 2, 1, 2, 0, 1, 1, 2, 0, 2, 1]],

        [[0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
         [2, 0, 0, 2, 1, 1, 1, 1, 0, 2, 2, 1]],

        [[1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
         [0, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1]],

        [[0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
         [2, 2, 2, 2, 0, 0, 2, 1, 0, 0, 2, 1]]],


       [[[0, 0, 1, 2, 4, 4, 1, 2, 2, 1, 3, 3],
         [0, 1, 1, 4, 4, 5, 3, 3, 0, 6, 6, 1]],

        [[2, 2, 1, 3, 3, 0, 3, 2, 3, 4, 3, 3],
         [5, 3, 6, 2, 0, 5, 3, 3, 5, 6, 6, 1]],

        [[3, 4, 1, 4, 1, 3, 4, 2, 0, 2, 3, 3],
         [1, 4, 5, 1, 3, 5, 3, 3, 2, 6, 6, 1]],

        [[4, 4, 1, 3, 2, 2, 1, 3, 3, 1, 3, 3],
         [4, 0, 4, 6, 2, 5, 3, 3, 3, 6, 6, 1]],

        [[0, 3, 1, 4, 4, 1, 3, 3, 0, 4, 3, 3],
         [3, 6, 2, 5, 6, 5, 3, 3, 4, 6, 6, 1]],

        [[2, 1, 1, 1, 3, 4, 4, 3, 1, 2, 3, 3],
         [6, 2, 0, 1, 1, 5, 1, 2, 3, 1, 6, 1]]],


       [[[1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 2, 1, 0, 0, 0, 2, 1, 2]],

        [[0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
         [2, 0, 0, 0, 2, 0, 1, 0, 2, 1, 1, 2]],

        [[1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
         [1, 2, 2, 1, 1, 2, 1, 0, 0, 0, 1, 2]],

        [[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 2, 0, 1, 1, 0, 1, 2, 1, 2]],

        [[1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
         [1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 1, 2]],

        [[0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
         [0, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1, 2]]],


       [[[1, 1, 4, 0, 2, 4, 3, 4, 3, 0, 0, 1],
         [0, 6, 5, 6, 0, 0, 5, 4, 4, 3, 2, 2]],

        [[2, 0, 4, 3, 1, 3, 0, 4, 4, 4, 0, 1],
         [3, 1, 1, 1, 5, 0, 5, 4, 0, 3, 2, 2]],

        [[4, 3, 4, 4, 0, 0, 1, 4, 2, 1, 0, 1],
         [1, 0, 2, 5, 2, 0, 5, 4, 5, 3, 2, 2]],

        [[3, 3, 4, 3, 4, 2, 3, 3, 4, 0, 0, 1],
         [6, 2, 6, 2, 3, 0, 5, 4, 6, 3, 2, 2]],

        [[1, 4, 4, 4, 2, 1, 0, 3, 2, 4, 0, 1],
         [2, 3, 4, 3, 6, 0, 5, 4, 2, 3, 2, 2]],

        [[2, 2, 4, 1, 1, 4, 1, 3, 0, 1, 0, 1],
         [4, 4, 3, 5, 4, 0, 3, 6, 6, 6, 2, 2]]]], dtype=uint64)
```

## convert digits to doubles 

```python 
>>> print(qmcseqcl.generalized_digital_net_from_digits.__doc__)
Convert digits of generalized digital net to floats

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    r_b (np.uint64): replications of bases 
    tmax (np.uint64): rows of each generating matrix
    bases (np.ndarray of np.uint64): bases for each dimension of size r_b*d
    xdig (np.ndarray of np.uint64): binary digital net points of size r*n*d*tmax
    x (np.ndarray of np.double): float digital net points of size r*n*d
>>> x = np.empty((r,n,d),dtype=np.float64)
>>> time_perf,time_process = qmcseqcl.generalized_digital_net_from_digits(r,n,d,r_b,tmax_new,bases,xdig_new,x,**kwargs)
>>> x
array([[[0.58935547, 0.01401849],
        [0.33349609, 0.48491554],
        [0.72216797, 0.31759687],
        [0.48193359, 0.6975017 ],
        [0.78955078, 0.16855117],
        [0.03759766, 0.9887344 ]],

       [[0.01275512, 0.02527427],
        [0.49380554, 0.79388271],
        [0.7749686 , 0.23970878],
        [0.97359019, 0.5857551 ],
        [0.13579056, 0.55933756],
        [0.45087567, 0.89847932]],

       [[0.52148438, 0.50347263],
        [0.2734375 , 0.67548232],
        [0.66992188, 0.64930067],
        [0.40625   , 0.17476258],
        [0.84863281, 0.34661985],
        [0.10449219, 0.3329363 ]],

       [[0.27294618, 0.13953207],
        [0.4373247 , 0.45261581],
        [0.95842417, 0.15089641],
        [0.75825613, 0.9164703 ],
        [0.39911312, 0.36021388],
        [0.51419659, 0.6641329 ]]])
```
