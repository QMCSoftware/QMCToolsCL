# Quasi-Monte Carlo Tools in OpenCL & C

The `qmctoolscl` package provides a **Python** interface to tools in **OpenCL** and **C** for performing **Quasi-Monte Carlo (QMC)**. Routines are written as OpenCL kernels. By replacing a few code snippets, these OpenCL kernels are automatically translated to C functions. Python functions provide access to both the OpenCL kernels and C functions in a unified interface. Code is available on <a href="https://github.com/QMCSoftware/qmctoolscl" target="_blank">GitHub</a>. 

[TOC]

## Installation 

```
pip install qmctoolscl
```

To use OpenCL features, please install <a href="https://pypi.org/project/pyopencl/" target="_blank">PyOpenCL</a>. Commands used to install PyOpenCL on Linux, MacOS, and Windows for our automated tests can be found <a href="https://github.com/QMCSoftware/qmctoolscl/blob/main/.github/workflows/doctests.yml" target="_blank">here</a>. Note that Apple has deprecated support for OpenCL (see <a href="https://developer.apple.com/opencl/" target="_blank">this post</a>), but installation is still possible in many cases. If you cannot install PyOpenCL, the C backend to this package will still work independently. 

<h1>Full Doctests for qmctoolscl</h1>

## Setup for OpenCL vs C 

Let's start by importing the relevant packages 

```python
>>> import qmctoolscl
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
...     "wait": True, ## required for accurate timing
...     "global_size": (2,2,2), ## global size 
... }
```

## Lattice Sequences

### linear order

```python
>>> print(qmctoolscl.lat_gen_linear.__doc__)
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
>>> time_perf,time_process = qmctoolscl.lat_gen_linear(r,n,d,g,x,**kwargs)
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

### natural order

```python 
>>> print(qmctoolscl.lat_gen_natural.__doc__)
Lattice points in natural order

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    n_start (np.uint64): starting index in sequence
    g (np.ndarray of np.uint64): pointer to generating vector of size r*d 
    x (np.ndarray of np.double): pointer to point storage of size r*n*d
>>> n_start = np.uint64(2)
>>> n = np.uint64(6) 
>>> x = np.empty((r,n,d),dtype=np.float64)
>>> time_perf,time_process = qmctoolscl.lat_gen_natural(r,n,d,n_start,g,x,**kwargs)
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
```

### Gray code order

```python 
>>> print(qmctoolscl.lat_gen_gray.__doc__)
Lattice points in Gray code order

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    n_start (np.uint64): starting index in sequence
    g (np.ndarray of np.uint64): pointer to generating vector of size r*d 
    x (np.ndarray of np.double): pointer to point storage of size r*n*d
>>> time_perf,time_process = qmctoolscl.lat_gen_gray(r,n,d,n_start,g,x,**kwargs)
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

### shift mod 1 

```python
>>> print(qmctoolscl.lat_shift_mod_1.__doc__)
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
>>> r = np.uint64(2*r_x)
>>> print(qmctoolscl.lat_get_shifts.__doc__)
Get random shifts
Args:
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications
    d (np.uint64): dimension
>>> shifts = qmctoolscl.lat_get_shifts(rng,r,d)
>>> xr = np.empty((r,n,d),dtype=np.float64)
>>> time_perf,time_process = qmctoolscl.lat_shift_mod_1(r,n,d,r_x,x,shifts,xr,**kwargs)
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

## Digital Net Base 2 

### LSB to MSB integer representations 

convert generating matrices from least significant bit (LSB) order to most significant bit (MSB) order

```python
>>> print(qmctoolscl.dnb2_gmat_lsb_to_msb.__doc__)
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
>>> tmaxes = np.tile(np.uint64(tmax),int(r))
>>> C = np.empty((d,mmax),dtype=np.uint64)
>>> time_perf,time_process = qmctoolscl.dnb2_gmat_lsb_to_msb(r,d,mmax,tmaxes,C_lsb,C,**kwargs)
>>> C
array([[ 8,  4,  2,  1],
       [ 8, 12, 10, 15],
       [ 8, 12,  6,  9],
       [ 8, 12,  2,  5]], dtype=uint64)
```

### linear matrix scrambling (LMS)

```python
>>> print(qmctoolscl.dnb2_linear_matrix_scramble.__doc__)
Linear matrix scrambling for base 2 generating matrices

Args:
    r (np.uint64): replications
    d (np.uint64): dimension
    mmax (np.uint64): columns in each generating matrix 
    r_C (np.uint64): original generating matrices
    tmax_new (np.uint64): bits in the integers of the resulting generating matrices
    S (np.ndarray of np.uint64): scrambling matrices of size r*d*tmax_new
    C (np.ndarray of np.uint64): original generating matrices of size r_C*d*mmax
    C_lms (np.ndarray of np.uint64): resulting generating matrices of size r*d*mmax
>>> r_C = r 
>>> r = np.uint64(2*r_C)
>>> tmax_new = np.uint64(6)
>>> print(qmctoolscl.dnb2_get_linear_scramble_matrix.__doc__)
Return a scrambling matrix for linear matrix scrambling

Args:
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications
    d (np.uint64): dimension
    tmax (np.uint64): bits in each integer
    tmax_new (np.uint64): bits in each integer of the generating matrix after scrambling
    print_mats (np.uint8): flag to print the resulting matrices
>>> print_mats = np.uint8(True)
>>> S = qmctoolscl.dnb2_get_linear_scramble_matrix(rng,r,d,tmax,tmax_new,print_mats)
S with shape (r=2, d=4, tmax_new=6)
l = 0
    j = 0
        1000
        1100
        0010
        1001
        0011
        0001
    j = 1
        1000
        0100
        1010
        1001
        0011
        1001
    j = 2
        1000
        0100
        0110
        0101
        1010
        1111
    j = 3
        1000
        1100
        0110
        1001
        0110
        0010
l = 1
    j = 0
        1000
        0100
        0010
        0011
        1101
        1010
    j = 1
        1000
        1100
        1010
        0111
        0111
        0010
    j = 2
        1000
        1100
        1110
        0001
        0110
        0100
    j = 3
        1000
        1100
        1010
        1101
        1110
        0000
>>> C_lms = np.empty((r,d,mmax),dtype=np.uint64)
>>> time_perf,time_process = qmctoolscl.dnb2_linear_matrix_scramble(r,d,mmax,r_C,tmax_new,S,C,C_lms,**kwargs)
>>> C_lms
array([[[52, 16, 10,  7],
        [45, 61, 39, 48],
        [35, 62, 22, 38],
        [52, 46, 11, 30]],
<BLANKLINE>
       [[35, 18, 13,  6],
        [56, 46, 55, 39],
        [56, 35, 17, 60],
        [62, 40, 10, 18]]], dtype=uint64)
```

### digital interlacing 

Digital interlacing is used to create **higher order digital nets**.

```python
>>> print(qmctoolscl.dnb2_interlace.__doc__)
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
>>> time_perf,time_process = qmctoolscl.dnb2_interlace(r,d_alpha,mmax,d,tmax,tmax_alpha,alpha,C_lms,C_alpha,**kwargs)
>>> C_alpha
array([[[3697, 1873, 1181, 1322],
        [3354, 3836,  621, 2428]],
<BLANKLINE>
       [[3402, 1628, 1463, 1085],
        [4052, 3146,  582, 2980]]], dtype=uint64)
``` 

### undo digital interlacing 

```python
>>> print(qmctoolscl.dnb2_undo_interlace.__doc__)
Undo interlacing of generating matrices in base 2

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
>>> time_perf,time_process = qmctoolscl.dnb2_undo_interlace(r,d,mmax,d_alpha,tmax,tmax_alpha,alpha,C_alpha,C_lms_cp,**kwargs)
>>> print((C_lms_cp==C_lms).all())
True
```

### natural order 

```python
>>> print(qmctoolscl.dnb2_gen_natural.__doc__)
Binary representation of digital net in base 2 in natural order

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    n_start (np.uint64): starting index in sequence
    mmax (np.uint64): columns in each generating matrix
    C (np.ndarray of np.uint64): generating matrices of size r*d*mmax
    xb (np.ndarray of np.uint64): binary digital net points of size r*n*d
>>> C = C_alpha
>>> d = d_alpha
>>> n_start = np.uint64(2)
>>> n = np.uint64(14)
>>> xb = np.empty((r,n,d),dtype=np.uint64)
>>> time_perf,time_process = qmctoolscl.dnb2_gen_natural(r,n,d,n_start,mmax,C,xb,**kwargs)
>>> xb
array([[[1873, 3836],
        [2336,  998],
        [1181,  621],
        [2796, 3959],
        [ 972, 3217],
        [3517,  395],
        [1322, 2428],
        [2907, 1126],
        [ 635, 1920],
        [3082, 2714],
        [ 439, 2833],
        [4038, 1547],
        [1766, 1517],
        [2199, 2295]],
<BLANKLINE>
       [[1628, 3146],
        [2838,  926],
        [1463,  582],
        [2301, 3474],
        [1003, 3596],
        [3745,  472],
        [1085, 2980],
        [2423, 1136],
        [ 609, 2030],
        [3883, 2106],
        [ 394, 2530],
        [3264, 1590],
        [2006, 1448],
        [2716, 2684]]], dtype=uint64)
```

### Gray code order

```python
>>> print(qmctoolscl.dnb2_gen_gray.__doc__)
Binary representation of digital net in base 2 in Gray code order

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    n_start (np.uint64): starting index in sequence
    mmax (np.uint64): columns in each generating matrix
    C (np.ndarray of np.uint64): generating matrices of size r*d*mmax
    xb (np.ndarray of np.uint64): binary digital net points of size r*n*d
>>> time_perf,time_process = qmctoolscl.dnb2_gen_gray(r,n,d,n_start,mmax,C,xb,**kwargs)
>>> xb
array([[[2336,  998],
        [1873, 3836],
        [ 972, 3217],
        [3517,  395],
        [2796, 3959],
        [1181,  621],
        [ 439, 2833],
        [4038, 1547],
        [2199, 2295],
        [1766, 1517],
        [ 635, 1920],
        [3082, 2714],
        [2907, 1126],
        [1322, 2428]],
<BLANKLINE>
       [[2838,  926],
        [1628, 3146],
        [1003, 3596],
        [3745,  472],
        [2301, 3474],
        [1463,  582],
        [ 394, 2530],
        [3264, 1590],
        [2716, 2684],
        [2006, 1448],
        [ 609, 2030],
        [3883, 2106],
        [2423, 1136],
        [1085, 2980]]], dtype=uint64)
```

### digital shift 

```python
>>> print(qmctoolscl.dnb2_digital_shift.__doc__)
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
>>> lshifts = np.tile(tmax_new-tmax,int(r)) 
>>> r_x = r 
>>> r = np.uint64(2*r_x)
>>> print(qmctoolscl.dnb2_get_digital_shifts.__doc__)
Get random shifts
Args:
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications
    d (np.uint64): dimension
    tmax_new (np.uint64): bits in each integer
>>> shiftsb = qmctoolscl.dnb2_get_digital_shifts(rng,r,d,tmax_new)
>>> shiftsb
array([[ 8506091904275275003, 16209688535125722604],
       [17447177601860792486,  9021077006476608718],
       [ 1440145505151606152,  5686212125327047696],
       [ 8845106632000742233,  2846419807334012155]], dtype=uint64)
>>> xrb = np.empty((r,n,d),dtype=np.uint64)
>>> time_perf,time_process = qmctoolscl.dnb2_digital_shift(r,n,d,r_x,lshifts,xb,shiftsb,xrb,**kwargs)
>>> xrb
array([[[16432427248447347963, 16038551749285643756],
        [  223972189540932859,  1095608185670338028],
        [ 5389600962134891771,  3018645226557539820],
        [12527806371517127931, 17889531196134917612],
        [15621779315520658683,  1694586936110613996],
        [ 4601471027345054971, 14277644294983779820],
        [ 7889098755325517051,  5900948988074657260],
        [ 9974265382798056699,  9242619911583565292],
        [18409507484862995707,  8035655211448272364],
        [ 1759699662474271995, 13701183542680356332],
        [ 5889500520773016827, 11021541764394911212],
        [13162813918976367867,  5283955839124899308],
        [14104066241096801531, 12003326483161679340],
        [ 2642405189438889211,  8589597965614843372]],
<BLANKLINE>
       [[ 4846105844478144678,  4958830142588421326],
        [10943979739937796262, 13371554246516507854],
        [14740514225811124390, 11380963211218748622],
        [ 1743125701219872934,  6967435576395662542],
        [ 9074985894579040422, 11822315974701057230],
        [12200484035974164646,  6436010820365944014],
        [16897738447321591974, 16361944399090517198],
        [ 4476810675033764006,  2184612772128195790],
        [ 6620524097662120102, 15776476447532352718],
        [10322482991360667814,  2860152716233770190],
        [15289953380350324902,   275086530123105486],
        [   40765042073825446, 18343528235133535438],
        [ 7300567641395064998,  4193218205935437006],
        [12821980784551293094, 14371353363792757966]],
<BLANKLINE>
       [[ 9366480849323679112,  8109148724852374544],
        [ 7416422210672254344, 11612949234946620432],
        [ 3403714942685142408,  9797998585116310544],
        [14424023230860746120,  6222140480984136720],
        [13635893296070909320, 13373856689248484368],
        [ 6497687886688673160,  7510169974412098576],
        [  615986773342805384, 18444909869667662864],
        [17265794595731529096,  3339836719467019280],
        [11136395502880284040, 13950317441551907856],
        [ 9051228875407744392,  1169101699074440208],
        [ 3768506512502152584,  3956829868416777232],
        [15230167564160064904, 16665988016856316944],
        [11983072232825937288,   615158944907869200],
        [ 4709758834622586248, 15648174501070584848]],
<BLANKLINE>
       [[14672764549818164057,  2188894261737919739],
        [ 2233822379020854105, 16366225888700241147],
        [ 4931478555815781209, 14357620454892999931],
        [10434877300462527321,  4179485297035678971],
        [17658651102764802905, 18347809724743259387],
        [ 2427477162997785433,   279368019732829435],
        [ 7088702777326248793, 13375835736126231803],
        [13168562274276418393,  4963111632198145275],
        [15204189305847882585,  9241531278200116475],
        [  549476118384288601,  9007344097576850683],
        [ 6687882410490274649,  6440292309975667963],
        [ 9831394950394880857, 11826597464310781179],
        [17127226346735084377,  6953702667495904507],
        [ 4111823423634350937, 11367230302318990587]]], dtype=uint64)
```

### convert digits to doubles

```python 
>>> print(qmctoolscl.dnb2_integer_to_float.__doc__)
Convert base 2 binary digital net points to floats

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    tmaxes (np.ndarray of np.uint64): bits in integers of each generating matrix of size r
    xb (np.ndarray of np.uint64): binary digital net points of size r*n*d
    x (np.ndarray of np.double): float digital net points of size r*n*d
>>> x = np.empty((r,n,d),dtype=np.float64)
>>> tmaxes_new = np.tile(tmax_new,int(r))
>>> time_perf,time_process = qmctoolscl.dnb2_integer_to_float(r,n,d,tmaxes_new,xrb,x,**kwargs)
>>> x
array([[[0.89080367, 0.86945163],
        [0.01214156, 0.05939304],
        [0.29217085, 0.16364109],
        [0.67913374, 0.96979343],
        [0.84685835, 0.09186374],
        [0.24944624, 0.77399265],
        [0.4276689 , 0.31989109],
        [0.54070601, 0.50104343],
        [0.9979814 , 0.43561374],
        [0.09539351, 0.74274265],
        [0.31927046, 0.59747898],
        [0.71355757, 0.28644382],
        [0.76458296, 0.65070163],
        [0.14324507, 0.46564304]],
<BLANKLINE>
       [[0.26270792, 0.26881872],
        [0.59327433, 0.72487341],
        [0.79908488, 0.61696325],
        [0.09449503, 0.37770544],
        [0.49195597, 0.64088903],
        [0.66138956, 0.34889685],
        [0.91602824, 0.88698278],
        [0.24268839, 0.1184281 ],
        [0.35889933, 0.8552445 ],
        [0.55958292, 0.15504919],
        [0.82887003, 0.01491247],
        [0.00220988, 0.99440466],
        [0.39576456, 0.22731481],
        [0.69508097, 0.77907263]],
<BLANKLINE>
       [[0.50775794, 0.43959783],
        [0.40204505, 0.62953924],
        [0.18451576, 0.53115057],
        [0.78192787, 0.33730291],
        [0.73920326, 0.72499822],
        [0.35224037, 0.40712713],
        [0.03339271, 0.99990057],
        [0.9359806 , 0.18105291],
        [0.60370521, 0.75624822],
        [0.4906681 , 0.06337713],
        [0.20429115, 0.21450018],
        [0.82562904, 0.90346502],
        [0.64960365, 0.03334783],
        [0.25531654, 0.84828924]],
<BLANKLINE>
       [[0.79541216, 0.1186602 ],
        [0.12109575, 0.88721488],
        [0.26733599, 0.77832816],
        [0.56567583, 0.22657035],
        [0.95727739, 0.99463676],
        [0.1315938 , 0.01514457],
        [0.38427935, 0.72510551],
        [0.71386919, 0.26905082],
        [0.82422075, 0.50098441],
        [0.02978716, 0.4882891 ],
        [0.36255083, 0.34912895],
        [0.53296099, 0.64112113],
        [0.9284688 , 0.37696098],
        [0.22290239, 0.61621879]]])
```

### nested uniform scrambling 

```python
>>> r = np.uint64(1) 
>>> n_start = np.uint64(0)
>>> n = np.uint64(8)
>>> d = np.uint64(4)
>>> C = np.array([
...   [ 8,  4,  2,  1],
...   [ 8, 12, 10, 15],
...   [ 8, 12,  6,  9],
...   [ 8, 12,  2,  5]], 
...   dtype=np.uint64)
>>> mmax = np.uint64(C.shape[1])
>>> tmax = np.uint64(np.ceil(np.log2(np.max(C))))
>>> xb = np.empty((r,n,d),dtype=np.uint64)
>>> time_perf,time_process = qmctoolscl.dnb2_gen_natural(r,n,d,n_start,mmax,C,xb,**kwargs)
>>> print(qmctoolscl.dnb2_nested_uniform_scramble.__doc__)
Nested uniform scramble of digital net b2

Args: 
    r (np.uint64): replications 
    n (np.uint64): points
    d (np.uint64): dimensions
    r_x (np.uint64): replications of xb
    tmax (np.uint64): maximum number of bits in each integer
    tmax_new (np.uint64): maximum number of bits in each integer after scrambling
    rngs (np.ndarray of numpy.random._generator.Generator): random number generators of size r*d
    root_nodes (np.ndarray of NUSNode_dnb2): root nodes of size r*d
    xb (np.ndarray of np.uint64): array of unrandomized points of size r*n*d
    xrb (np.ndarray of np.uint64): array to store scrambled points of size r*n*d
>>> tmax_new = np.uint64(2*tmax) 
>>> r_x = r
>>> r = np.uint64(2*r_x)
>>> base_seed_seq = np.random.SeedSequence(7)
>>> seeds = base_seed_seq.spawn(r*d)
>>> rngs = np.array([np.random.Generator(np.random.SFC64(seed)) for seed in seeds]).reshape(r,d)
>>> root_nodes = np.array([qmctoolscl.NUSNode_dnb2() for i in range(r*d)]).reshape(r,d)
>>> xrb = np.zeros((r,n,d),dtype=np.uint64)
>>> time_perf,time_process = qmctoolscl.dnb2_nested_uniform_scramble(r,n,d,r_x,tmax,tmax_new,rngs,root_nodes,xb,xrb)
>>> xrb
array([[[ 94,  54, 235,  30],
        [174, 243,  26, 129],
        [ 30, 172,  85, 222],
        [254, 127, 143,  68],
        [ 96, 207, 171,  46],
        [131,  21, 125, 165],
        [ 38,  79,  47, 237],
        [199, 133, 199,  99]],

       [[ 37, 152, 170, 143],
        [219,  56,  68,  29],
        [ 65,  98,  34, 111],
        [181, 211, 196, 220],
        [ 26,   9, 242, 180],
        [245, 167,   2,  42],
        [116, 225, 105,  93],
        [138,  93, 139, 230]]], dtype=uint64)
```

## Generalized Digital Net 

Accommodates both Halton sequences and digital nets in any prime base

### linear Matrix Scramble 

```python
>>> print(qmctoolscl.gdn_linear_matrix_scramble.__doc__)
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
>>> print(qmctoolscl.gdn_get_halton_generating_matrix.__doc__)
Return the identity matrices comprising the Halton generating matrices

Arg:
    r (np.uint64): replications 
    d (np.uint64): dimension 
    mmax (np.uint64): maximum number rows and columns in each generating matrix
>>> C = qmctoolscl.gdn_get_halton_generating_matrix(r_C,d,mmax)
>>> tmax_new = np.uint64(2*tmax)
>>> r = np.uint64(2*r_C)
>>> print(qmctoolscl.gdn_get_linear_scramble_matrix.__doc__)
Return a scrambling matrix for linear matrix scrambling

Args:
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications
    d (np.uint64): dimension
    tmax (np.uint64): bits in each integer
    tmax_new (np.uint64): bits in each integer of the generating matrix after scrambling
    r_b (np.uint64): replications of bases 
    bases (np.ndarray of np.uint64): bases of size r_b*d
>>> S = qmctoolscl.gdn_get_linear_scramble_matrix(rng,r,d,tmax,tmax_new,r_b,bases)
>>> S
array([[[[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 1, 0, 1],
         [0, 0, 1, 0, 0],
         [1, 1, 1, 0, 1],
         [0, 1, 1, 1, 0],
         [1, 0, 0, 1, 0],
         [1, 0, 0, 1, 1]],
<BLANKLINE>
        [[2, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 2, 1, 0, 0],
         [0, 2, 2, 2, 0],
         [0, 2, 2, 1, 2],
         [0, 2, 2, 0, 2],
         [0, 1, 0, 2, 2],
         [0, 1, 2, 0, 0],
         [1, 0, 2, 0, 0],
         [2, 0, 1, 2, 1]]],
<BLANKLINE>
<BLANKLINE>
       [[[1, 0, 0, 0, 0],
         [4, 3, 0, 0, 0],
         [3, 3, 3, 0, 0],
         [3, 1, 1, 3, 0],
         [4, 2, 2, 0, 3],
         [4, 2, 3, 3, 3],
         [1, 3, 3, 3, 1],
         [2, 0, 4, 3, 1],
         [1, 0, 3, 0, 4],
         [1, 0, 4, 1, 0]],
<BLANKLINE>
        [[2, 0, 0, 0, 0],
         [1, 5, 0, 0, 0],
         [2, 3, 1, 0, 0],
         [4, 1, 6, 1, 0],
         [5, 1, 3, 5, 1],
         [1, 5, 2, 0, 4],
         [3, 1, 4, 0, 0],
         [1, 3, 6, 0, 4],
         [3, 6, 2, 0, 2],
         [1, 1, 6, 6, 4]]],
<BLANKLINE>
<BLANKLINE>
       [[[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 1, 0, 0, 1],
         [0, 1, 1, 1, 0],
         [0, 0, 1, 0, 0],
         [1, 1, 1, 0, 1],
         [1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1]],
<BLANKLINE>
        [[1, 0, 0, 0, 0],
         [2, 2, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 2, 2, 1, 0],
         [0, 2, 1, 0, 2],
         [2, 2, 0, 0, 2],
         [0, 2, 2, 1, 0],
         [2, 0, 2, 0, 1],
         [0, 2, 0, 0, 2],
         [0, 1, 0, 1, 2]]],
<BLANKLINE>
<BLANKLINE>
       [[[2, 0, 0, 0, 0],
         [1, 3, 0, 0, 0],
         [1, 0, 3, 0, 0],
         [2, 3, 3, 3, 0],
         [0, 1, 3, 3, 2],
         [3, 4, 4, 1, 2],
         [4, 2, 1, 4, 0],
         [1, 3, 1, 3, 3],
         [1, 4, 2, 4, 2],
         [0, 4, 1, 4, 1]],
<BLANKLINE>
        [[4, 0, 0, 0, 0],
         [4, 2, 0, 0, 0],
         [6, 5, 5, 0, 0],
         [6, 6, 1, 5, 0],
         [3, 1, 1, 5, 6],
         [2, 3, 5, 6, 0],
         [3, 6, 0, 2, 5],
         [5, 6, 1, 1, 1],
         [2, 0, 3, 5, 6],
         [3, 3, 2, 2, 5]]]], dtype=uint64)
>>> C_lms = np.empty((r,d,mmax,tmax_new),dtype=np.uint64)
>>> time_perf,time_process = qmctoolscl.gdn_linear_matrix_scramble(r,d,mmax,r_C,r_b,tmax,tmax_new,bases,S,C,C_lms,**kwargs)
>>> C_lms 
array([[[[1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
         [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 1, 0, 1, 0, 0, 1]],
<BLANKLINE>
        [[2, 1, 0, 0, 0, 0, 0, 0, 1, 2],
         [0, 1, 2, 2, 2, 2, 1, 1, 0, 0],
         [0, 0, 1, 2, 2, 2, 0, 2, 2, 1],
         [0, 0, 0, 2, 1, 0, 2, 0, 0, 2],
         [0, 0, 0, 0, 2, 2, 2, 0, 0, 1]]],
<BLANKLINE>
<BLANKLINE>
       [[[1, 4, 3, 3, 4, 4, 1, 2, 1, 1],
         [0, 3, 3, 1, 2, 2, 3, 0, 0, 0],
         [0, 0, 3, 1, 2, 3, 3, 4, 3, 4],
         [0, 0, 0, 3, 0, 3, 3, 3, 0, 1],
         [0, 0, 0, 0, 3, 3, 1, 1, 4, 0]],
<BLANKLINE>
        [[2, 1, 2, 4, 5, 1, 3, 1, 3, 1],
         [0, 5, 3, 1, 1, 5, 1, 3, 6, 1],
         [0, 0, 1, 6, 3, 2, 4, 6, 2, 6],
         [0, 0, 0, 1, 5, 0, 0, 0, 0, 6],
         [0, 0, 0, 0, 1, 4, 0, 4, 2, 4]]],
<BLANKLINE>
<BLANKLINE>
       [[[1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
         [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
         [0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
         [0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
         [0, 0, 0, 0, 1, 0, 0, 1, 0, 1]],
<BLANKLINE>
        [[1, 2, 1, 1, 0, 2, 0, 2, 0, 0],
         [0, 2, 1, 2, 2, 2, 2, 0, 2, 1],
         [0, 0, 1, 2, 1, 0, 2, 2, 0, 0],
         [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
         [0, 0, 0, 0, 2, 2, 0, 1, 2, 2]]],
<BLANKLINE>
<BLANKLINE>
       [[[2, 1, 1, 2, 0, 3, 4, 1, 1, 0],
         [0, 3, 0, 3, 1, 4, 2, 3, 4, 4],
         [0, 0, 3, 3, 3, 4, 1, 1, 2, 1],
         [0, 0, 0, 3, 3, 1, 4, 3, 4, 4],
         [0, 0, 0, 0, 2, 2, 0, 3, 2, 1]],
<BLANKLINE>
        [[4, 4, 6, 6, 3, 2, 3, 5, 2, 3],
         [0, 2, 5, 6, 1, 3, 6, 6, 0, 3],
         [0, 0, 5, 1, 1, 5, 0, 1, 3, 2],
         [0, 0, 0, 5, 5, 6, 2, 1, 5, 2],
         [0, 0, 0, 0, 6, 0, 5, 1, 6, 5]]]], dtype=uint64)
```

### natural order

```python
>>> print(qmctoolscl.gdn_gen_natural.__doc__)
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
>>> time_perf,time_process = qmctoolscl.gdn_gen_natural(r,n,d,r_b,mmax,tmax,n_start,bases,C,xdig,**kwargs)
>>> xdig
array([[[[0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
         [1, 2, 0, 0, 0, 0, 0, 0, 2, 1]],
<BLANKLINE>
        [[1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
         [0, 1, 2, 2, 2, 2, 1, 1, 0, 0]],
<BLANKLINE>
        [[0, 0, 1, 0, 1, 1, 1, 1, 0, 0],
         [2, 2, 2, 2, 2, 2, 1, 1, 1, 2]],
<BLANKLINE>
        [[1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
         [1, 0, 2, 2, 2, 2, 1, 1, 2, 1]],
<BLANKLINE>
        [[0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 2, 1, 1, 1, 1, 2, 2, 0, 0]],
<BLANKLINE>
        [[1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
         [2, 0, 1, 1, 1, 1, 2, 2, 1, 2]]],
<BLANKLINE>
<BLANKLINE>
       [[[2, 3, 1, 1, 3, 3, 2, 4, 2, 2],
         [4, 2, 4, 1, 3, 2, 6, 2, 6, 2]],
<BLANKLINE>
        [[3, 2, 4, 4, 2, 2, 3, 1, 3, 3],
         [6, 3, 6, 5, 1, 3, 2, 3, 2, 3]],
<BLANKLINE>
        [[4, 1, 2, 2, 1, 1, 4, 3, 4, 4],
         [1, 4, 1, 2, 6, 4, 5, 4, 5, 4]],
<BLANKLINE>
        [[0, 3, 3, 1, 2, 2, 3, 0, 0, 0],
         [3, 5, 3, 6, 4, 5, 1, 5, 1, 5]],
<BLANKLINE>
        [[1, 2, 1, 4, 1, 1, 4, 2, 1, 1],
         [5, 6, 5, 3, 2, 6, 4, 6, 4, 6]],
<BLANKLINE>
        [[2, 1, 4, 2, 0, 0, 0, 4, 2, 2],
         [0, 5, 3, 1, 1, 5, 1, 3, 6, 1]]],
<BLANKLINE>
<BLANKLINE>
       [[[0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
         [2, 1, 2, 2, 0, 1, 0, 1, 0, 0]],
<BLANKLINE>
        [[1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 2, 1, 2, 2, 2, 2, 0, 2, 1]],
<BLANKLINE>
        [[0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
         [1, 1, 2, 0, 2, 1, 2, 2, 2, 1]],
<BLANKLINE>
        [[1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
         [2, 0, 0, 1, 2, 0, 2, 1, 2, 1]],
<BLANKLINE>
        [[0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
         [0, 1, 2, 1, 1, 1, 1, 0, 1, 2]],
<BLANKLINE>
        [[1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
         [1, 0, 0, 2, 1, 0, 1, 2, 1, 2]]],
<BLANKLINE>
<BLANKLINE>
       [[[4, 2, 2, 4, 0, 1, 3, 2, 2, 0],
         [1, 1, 5, 5, 6, 4, 6, 3, 4, 6]],
<BLANKLINE>
        [[1, 3, 3, 1, 0, 4, 2, 3, 3, 0],
         [5, 5, 4, 4, 2, 6, 2, 1, 6, 2]],
<BLANKLINE>
        [[3, 4, 4, 3, 0, 2, 1, 4, 4, 0],
         [2, 2, 3, 3, 5, 1, 5, 6, 1, 5]],
<BLANKLINE>
        [[0, 3, 0, 3, 1, 4, 2, 3, 4, 4],
         [6, 6, 2, 2, 1, 3, 1, 4, 3, 1]],
<BLANKLINE>
        [[2, 4, 1, 0, 1, 2, 1, 4, 0, 4],
         [3, 3, 1, 1, 4, 5, 4, 2, 5, 4]],
<BLANKLINE>
        [[4, 0, 2, 2, 1, 0, 0, 0, 1, 4],
         [0, 2, 5, 6, 1, 3, 6, 6, 0, 3]]]], dtype=uint64)
```

### digital shift 

```python
>>> print(qmctoolscl.gdn_digital_shift.__doc__)
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
>>> r_x = r
>>> tmax_new = np.uint64(12)
>>> print(qmctoolscl.gdn_get_digital_shifts.__doc__)
Return digital shifts for gdn

Args: 
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications 
    d (np.uint64): dimension 
    tmax_new (np.uint64): number of bits in each shift 
    r_b (np.uint64): replications of bases 
    bases (np.ndarray of np.uint64): bases of size r_b*d
>>> shifts = qmctoolscl.gdn_get_digital_shifts(rng,r,d,tmax_new,r_b,bases)
>>> shifts
array([[[1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 2, 0, 0, 1, 1, 0, 0, 1]],
<BLANKLINE>
       [[0, 3, 2, 1, 1, 2, 4, 3, 0, 1, 3, 1],
        [1, 6, 6, 6, 4, 5, 2, 4, 1, 4, 6, 4]],
<BLANKLINE>
       [[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1],
        [2, 0, 1, 1, 0, 2, 1, 0, 1, 1, 1, 1]],
<BLANKLINE>
       [[1, 1, 1, 1, 4, 3, 0, 2, 0, 3, 3, 2],
        [3, 0, 3, 2, 4, 3, 6, 2, 2, 3, 2, 4]]], dtype=uint64)
>>> xdig_new = np.empty((r,n,d,tmax_new),dtype=np.uint64)
>>> time_perf,time_process = qmctoolscl.gdn_digital_shift(r,n,d,r_x,r_b,tmax,tmax_new,bases,shifts,xdig,xdig_new,**kwargs)
>>> xdig_new
array([[[[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
         [2, 2, 0, 1, 2, 0, 0, 1, 0, 1, 0, 1]],
<BLANKLINE>
        [[0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
         [1, 1, 2, 0, 1, 2, 1, 2, 1, 0, 0, 1]],
<BLANKLINE>
        [[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 2, 2, 0, 1, 2, 1, 2, 2, 2, 0, 1]],
<BLANKLINE>
        [[0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
         [2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 1]],
<BLANKLINE>
        [[1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
         [1, 2, 1, 2, 0, 1, 2, 0, 1, 0, 0, 1]],
<BLANKLINE>
        [[0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
         [0, 0, 1, 2, 0, 1, 2, 0, 2, 2, 0, 1]]],
<BLANKLINE>
<BLANKLINE>
       [[[2, 1, 3, 2, 4, 0, 1, 2, 2, 3, 3, 1],
         [5, 1, 3, 0, 0, 0, 1, 6, 0, 6, 6, 4]],
<BLANKLINE>
        [[3, 0, 1, 0, 3, 4, 2, 4, 3, 4, 3, 1],
         [0, 2, 5, 4, 5, 1, 4, 0, 3, 0, 6, 4]],
<BLANKLINE>
        [[4, 4, 4, 3, 2, 3, 3, 1, 4, 0, 3, 1],
         [2, 3, 0, 1, 3, 2, 0, 1, 6, 1, 6, 4]],
<BLANKLINE>
        [[0, 1, 0, 2, 3, 4, 2, 3, 0, 1, 3, 1],
         [4, 4, 2, 5, 1, 3, 3, 2, 2, 2, 6, 4]],
<BLANKLINE>
        [[1, 0, 3, 0, 2, 3, 3, 0, 1, 2, 3, 1],
         [6, 5, 4, 2, 6, 4, 6, 3, 5, 3, 6, 4]],
<BLANKLINE>
        [[2, 4, 1, 3, 1, 2, 4, 2, 2, 3, 3, 1],
         [1, 4, 2, 0, 5, 3, 3, 0, 0, 5, 6, 4]]],
<BLANKLINE>
<BLANKLINE>
       [[[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
         [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]],
<BLANKLINE>
        [[0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1],
         [2, 2, 2, 0, 2, 1, 0, 0, 0, 2, 1, 1]],
<BLANKLINE>
        [[1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
         [0, 1, 0, 1, 2, 0, 0, 2, 0, 2, 1, 1]],
<BLANKLINE>
        [[0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
         [1, 0, 1, 2, 2, 2, 0, 1, 0, 2, 1, 1]],
<BLANKLINE>
        [[1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1],
         [2, 1, 0, 2, 1, 0, 2, 0, 2, 0, 1, 1]],
<BLANKLINE>
        [[0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
         [0, 0, 1, 0, 1, 2, 2, 2, 2, 0, 1, 1]]],
<BLANKLINE>
<BLANKLINE>
       [[[0, 3, 3, 0, 4, 4, 3, 4, 2, 3, 3, 2],
         [4, 1, 1, 0, 3, 0, 5, 5, 6, 2, 2, 4]],
<BLANKLINE>
        [[2, 4, 4, 2, 4, 2, 2, 0, 3, 3, 3, 2],
         [1, 5, 0, 6, 6, 2, 1, 3, 1, 5, 2, 4]],
<BLANKLINE>
        [[4, 0, 0, 4, 4, 0, 1, 1, 4, 3, 3, 2],
         [5, 2, 6, 5, 2, 4, 4, 1, 3, 1, 2, 4]],
<BLANKLINE>
        [[1, 4, 1, 4, 0, 2, 2, 0, 4, 2, 3, 2],
         [2, 6, 5, 4, 5, 6, 0, 6, 5, 4, 2, 4]],
<BLANKLINE>
        [[3, 0, 2, 1, 0, 0, 1, 1, 0, 2, 3, 2],
         [6, 3, 4, 3, 1, 1, 3, 4, 0, 0, 2, 4]],
<BLANKLINE>
        [[0, 1, 3, 3, 0, 3, 0, 2, 1, 2, 3, 2],
         [3, 2, 1, 1, 5, 6, 5, 1, 2, 6, 2, 4]]]], dtype=uint64)
```

### digital permutation 

```python
>>> print(qmctoolscl.gdn_digital_permutation.__doc__)
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
>>> print(qmctoolscl.gdn_get_digital_permutations.__doc__)
Return permutations for gdn

Args: 
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications 
    d (np.uint64): dimension 
    tmax_new (np.uint64): number of bits in each shift 
    r_b (np.uint64): replications of bases 
    bases (np.ndarray of np.uint64): bases of size r_b*d
>>> perms = qmctoolscl.gdn_get_digital_permutations(rng,r,d,tmax_new,r_b,bases)
>>> perms
array([[[[0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0]],
<BLANKLINE>
        [[2, 1, 0, 0, 0, 0, 0],
         [0, 2, 1, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0, 0],
         [1, 0, 2, 0, 0, 0, 0],
         [1, 2, 0, 0, 0, 0, 0],
         [2, 1, 0, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0, 0],
         [2, 1, 0, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0],
         [1, 0, 2, 0, 0, 0, 0],
         [0, 2, 1, 0, 0, 0, 0]]],
<BLANKLINE>
<BLANKLINE>
       [[[3, 0, 1, 4, 2, 0, 0],
         [1, 3, 2, 0, 4, 0, 0],
         [1, 0, 3, 4, 2, 0, 0],
         [3, 2, 1, 4, 0, 0, 0],
         [1, 0, 3, 4, 2, 0, 0],
         [1, 3, 0, 4, 2, 0, 0],
         [0, 4, 1, 2, 3, 0, 0],
         [2, 4, 3, 1, 0, 0, 0],
         [2, 1, 4, 0, 3, 0, 0],
         [2, 0, 4, 1, 3, 0, 0],
         [1, 4, 2, 3, 0, 0, 0],
         [3, 1, 2, 0, 4, 0, 0]],
<BLANKLINE>
        [[3, 2, 6, 5, 0, 4, 1],
         [6, 2, 0, 5, 1, 4, 3],
         [2, 0, 3, 5, 6, 4, 1],
         [0, 6, 2, 1, 4, 3, 5],
         [3, 2, 5, 4, 6, 0, 1],
         [1, 4, 3, 6, 5, 0, 2],
         [5, 6, 4, 2, 3, 0, 1],
         [3, 0, 1, 4, 2, 5, 6],
         [3, 2, 6, 1, 0, 4, 5],
         [1, 4, 3, 2, 5, 0, 6],
         [6, 0, 4, 1, 5, 3, 2],
         [6, 3, 4, 2, 5, 1, 0]]],
<BLANKLINE>
<BLANKLINE>
       [[[1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0]],
<BLANKLINE>
        [[1, 0, 2, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0],
         [0, 2, 1, 0, 0, 0, 0],
         [2, 1, 0, 0, 0, 0, 0],
         [1, 2, 0, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0],
         [1, 0, 2, 0, 0, 0, 0],
         [1, 0, 2, 0, 0, 0, 0]]],
<BLANKLINE>
<BLANKLINE>
       [[[3, 1, 4, 2, 0, 0, 0],
         [0, 1, 4, 3, 2, 0, 0],
         [4, 0, 2, 3, 1, 0, 0],
         [4, 1, 0, 2, 3, 0, 0],
         [1, 3, 2, 4, 0, 0, 0],
         [3, 2, 0, 4, 1, 0, 0],
         [2, 4, 0, 1, 3, 0, 0],
         [2, 3, 1, 4, 0, 0, 0],
         [4, 0, 2, 1, 3, 0, 0],
         [0, 1, 3, 4, 2, 0, 0],
         [2, 0, 1, 3, 4, 0, 0],
         [0, 4, 1, 2, 3, 0, 0]],
<BLANKLINE>
        [[1, 6, 0, 4, 2, 3, 5],
         [5, 6, 0, 4, 3, 1, 2],
         [4, 2, 1, 5, 3, 0, 6],
         [3, 1, 4, 5, 6, 0, 2],
         [0, 1, 3, 6, 2, 4, 5],
         [4, 0, 2, 6, 1, 5, 3],
         [0, 6, 4, 2, 1, 3, 5],
         [5, 1, 3, 4, 0, 6, 2],
         [4, 6, 0, 1, 3, 5, 2],
         [3, 2, 6, 5, 0, 4, 1],
         [3, 4, 2, 6, 1, 5, 0],
         [2, 6, 1, 3, 0, 4, 5]]]], dtype=uint64)
>>> bmax = bases.max().astype(np.uint64)
>>> time_perf,time_process = qmctoolscl.gdn_digital_permutation(r,n,d,r_x,r_b,tmax,tmax_new,bmax,perms,xdig,xdig_new,**kwargs)
>>> xdig_new
array([[[[0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
         [1, 1, 2, 2, 1, 1, 2, 2, 0, 1, 1, 0]],
<BLANKLINE>
        [[1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
         [2, 2, 1, 1, 2, 0, 1, 0, 2, 0, 1, 0]],
<BLANKLINE>
        [[0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
         [0, 1, 1, 1, 2, 0, 1, 0, 1, 2, 1, 0]],
<BLANKLINE>
        [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
         [1, 0, 1, 1, 2, 0, 1, 0, 0, 1, 1, 0]],
<BLANKLINE>
        [[0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
         [2, 1, 0, 0, 0, 2, 0, 1, 2, 0, 1, 0]],
<BLANKLINE>
        [[1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 2, 0, 1, 1, 2, 1, 0]]],
<BLANKLINE>
<BLANKLINE>
       [[[1, 0, 0, 2, 4, 4, 1, 0, 4, 4, 1, 3],
         [0, 0, 6, 6, 4, 3, 1, 1, 5, 3, 6, 6]],
<BLANKLINE>
        [[4, 2, 2, 0, 3, 0, 2, 4, 0, 1, 1, 3],
         [1, 5, 1, 3, 2, 6, 4, 4, 6, 2, 6, 6]],
<BLANKLINE>
        [[2, 3, 3, 1, 0, 3, 3, 1, 3, 3, 1, 3],
         [2, 1, 0, 2, 1, 5, 0, 2, 4, 5, 6, 6]],
<BLANKLINE>
        [[3, 0, 4, 2, 3, 0, 2, 2, 2, 2, 1, 3],
         [5, 4, 5, 5, 6, 0, 6, 5, 2, 0, 6, 6]],
<BLANKLINE>
        [[0, 2, 0, 0, 0, 3, 3, 3, 1, 0, 1, 3],
         [4, 3, 4, 1, 5, 2, 3, 6, 0, 6, 6, 6]],
<BLANKLINE>
        [[1, 3, 2, 1, 1, 1, 0, 0, 4, 4, 1, 3],
         [3, 4, 5, 6, 2, 0, 6, 4, 5, 4, 6, 6]]],
<BLANKLINE>
<BLANKLINE>
       [[[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
         [2, 0, 1, 2, 0, 1, 1, 1, 0, 0, 1, 1]],
<BLANKLINE>
        [[0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
         [1, 1, 0, 2, 1, 0, 0, 0, 2, 1, 1, 1]],
<BLANKLINE>
        [[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 1, 1, 0, 2, 2, 1, 1, 1]],
<BLANKLINE>
        [[0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
         [2, 2, 2, 1, 1, 2, 0, 1, 2, 1, 1, 1]],
<BLANKLINE>
        [[1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
         [1, 0, 1, 1, 2, 1, 2, 0, 1, 2, 1, 1]],
<BLANKLINE>
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
         [0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1]]],
<BLANKLINE>
<BLANKLINE>
       [[[0, 4, 2, 3, 1, 2, 1, 1, 2, 0, 2, 0],
         [6, 6, 0, 0, 5, 1, 5, 4, 3, 1, 3, 2]],
<BLANKLINE>
        [[1, 3, 3, 1, 1, 1, 0, 4, 1, 0, 2, 0],
         [3, 1, 3, 6, 3, 3, 4, 1, 2, 6, 3, 2]],
<BLANKLINE>
        [[2, 2, 1, 2, 1, 0, 4, 0, 3, 0, 2, 0],
         [0, 0, 5, 5, 4, 0, 3, 2, 6, 4, 3, 2]],
<BLANKLINE>
        [[3, 3, 4, 2, 3, 1, 0, 4, 3, 2, 2, 0],
         [5, 2, 1, 4, 1, 6, 6, 0, 1, 2, 3, 2]],
<BLANKLINE>
        [[4, 2, 0, 4, 3, 0, 4, 0, 4, 2, 2, 0],
         [4, 4, 2, 1, 2, 5, 1, 3, 5, 0, 3, 2]],
<BLANKLINE>
        [[0, 0, 2, 0, 3, 3, 2, 2, 0, 2, 2, 0],
         [1, 0, 0, 2, 1, 6, 5, 2, 4, 5, 3, 2]]]], dtype=uint64)
```

### convert digits to doubles 

```python 
>>> print(qmctoolscl.gdn_integer_to_float.__doc__)
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
>>> time_perf,time_process = qmctoolscl.gdn_integer_to_float(r,n,d,r_b,tmax_new,bases,xdig_new,x,**kwargs)
>>> x
array([[[0.19482422, 0.54993875],
        [0.69970703, 0.94706656],
        [0.49169922, 0.16927185],
        [0.99658203, 0.39142633],
        [0.12060547, 0.78078093],
        [0.60986328, 0.00298622]],
<BLANKLINE>
       [[0.20475129, 0.02025669],
        [0.89699598, 0.2492386 ],
        [0.54583484, 0.3070579 ],
        [0.63619198, 0.81294329],
        [0.08023862, 0.64505057],
        [0.33798649, 0.52740742]],
<BLANKLINE>
       [[0.5090332 , 0.73038399],
        [0.13989258, 0.4733771 ],
        [0.78149414, 0.04295491],
        [0.41235352, 0.98244584],
        [0.64672852, 0.39332494],
        [0.01586914, 0.33327312]],
<BLANKLINE>
       [[0.18126442, 0.97990468],
        [0.34599479, 0.46043401],
        [0.49157278, 0.01690188],
        [0.75623602, 0.75980128],
        [0.88741349, 0.65947198],
        [0.01718297, 0.14380716]]])
```

### nested uniform scramble 

```python 
>>> r = np.uint64(1)
>>> n_start = np.uint64(0) 
>>> n = np.uint64(9)
>>> bases = np.array([[2,3,5]],dtype=np.uint64)
>>> r_b = np.uint64(bases.shape[0])
>>> d = np.uint64(bases.shape[1])
>>> mmax = np.uint64(3)
>>> tmax = mmax
>>> C = np.tile(np.eye(mmax,dtype=np.uint64)[None,None,:,:],(int(r),int(d),1,1))
>>> xdig = np.empty((r,n,d,tmax),dtype=np.uint64)
>>> time_perf,time_process = qmctoolscl.gdn_gen_natural(r,n,d,r_b,mmax,tmax,n_start,bases,C,xdig)
>>> r_x = r 
>>> r = np.uint64(2) 
>>> tmax_new = np.uint64(8)
>>> base_seed_seq = np.random.SeedSequence(7)
>>> seeds = base_seed_seq.spawn(r*d)
>>> rngs = np.array([np.random.Generator(np.random.SFC64(seed)) for seed in seeds]).reshape(r,d)
>>> root_nodes = np.array([qmctoolscl.NUSNode_gdn() for i in range(r*d)]).reshape(r,d)
>>> xrdig = np.zeros((r,n,d,tmax_new),dtype=np.uint64)
>>> print(qmctoolscl.gdn_nested_uniform_scramble.__doc__)
Nested uniform scramble of general digital nets

Args: 
    r (np.uint64): replications 
    n (np.uint64): points
    d (np.uint64): dimensions
    r_x (np.uint64): replications of xb
    r_b (np.uint64): replications of bases
    tmax (np.uint64): maximum number digits in each point representation
    tmax_new (np.uint64): maximum number digits in each point representation after scrambling
    rngs (np.ndarray of numpy.random._generator.Generator): random number generators of size r*d
    root_nodes (np.ndarray of NUSNode_gdn): root nodes of size r*d
    bases (np.ndarray of np.uint64): array of bases of size r*d
    xdig (np.ndarray of np.uint64): array of unrandomized points of size r*n*d*tmax
    xrdig (np.ndarray of np.uint64): array to store scrambled points of size r*n*d*tmax_new
>>> time_perf,time_process = qmctoolscl.gdn_nested_uniform_scramble(r,n,d,r_x,r_b,tmax,tmax_new,rngs,root_nodes,bases,xdig,xrdig)
>>> xrdig 
array([[[[0, 0, 1, 0, 0, 1, 0, 0],
         [2, 2, 2, 2, 1, 0, 2, 1],
         [3, 4, 2, 0, 3, 4, 1, 0]],

        [[1, 0, 1, 0, 1, 0, 1, 1],
         [1, 0, 1, 0, 1, 0, 1, 2],
         [1, 0, 3, 3, 3, 0, 1, 2]],

        [[0, 1, 1, 1, 1, 0, 0, 0],
         [0, 2, 2, 0, 0, 2, 0, 2],
         [4, 1, 2, 1, 4, 3, 0, 0]],

        [[1, 1, 0, 0, 0, 1, 0, 1],
         [2, 1, 1, 1, 0, 2, 2, 2],
         [2, 3, 2, 4, 2, 2, 2, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0],
         [1, 2, 2, 0, 1, 1, 2, 0],
         [0, 1, 0, 4, 3, 1, 3, 0]],

        [[1, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 2, 0, 0, 1, 0, 2],
         [3, 2, 1, 4, 2, 1, 4, 1]],

        [[0, 1, 0, 0, 0, 0, 0, 0],
         [2, 0, 1, 1, 1, 0, 1, 2],
         [1, 1, 4, 4, 4, 0, 1, 1]],

        [[1, 1, 1, 1, 1, 0, 1, 1],
         [1, 1, 1, 2, 0, 0, 0, 0],
         [4, 4, 1, 4, 1, 2, 3, 0]],

        [[1, 0, 1, 0, 0, 0, 0, 0],
         [0, 1, 2, 0, 2, 1, 0, 1],
         [2, 1, 0, 0, 1, 2, 3, 0]]],


       [[[0, 0, 0, 0, 1, 1, 0, 0],
         [0, 1, 1, 1, 0, 0, 2, 1],
         [2, 4, 0, 0, 1, 1, 0, 0]],

        [[1, 0, 1, 0, 1, 0, 0, 1],
         [1, 0, 1, 2, 0, 0, 0, 1],
         [1, 1, 1, 1, 3, 0, 1, 0]],

        [[0, 1, 0, 1, 0, 1, 0, 1],
         [2, 0, 1, 0, 0, 2, 1, 0],
         [0, 0, 2, 2, 0, 0, 4, 0]],

        [[1, 1, 0, 0, 1, 1, 1, 0],
         [0, 2, 1, 1, 2, 1, 2, 2],
         [4, 1, 4, 3, 4, 0, 1, 3]],

        [[0, 0, 1, 1, 0, 1, 0, 0],
         [1, 2, 2, 2, 1, 1, 0, 1],
         [3, 1, 0, 4, 2, 0, 1, 2]],

        [[1, 0, 0, 1, 1, 0, 0, 0],
         [2, 1, 0, 2, 0, 1, 1, 1],
         [2, 3, 2, 0, 4, 3, 2, 1]],

        [[0, 1, 1, 0, 1, 0, 1, 1],
         [0, 0, 0, 1, 2, 1, 1, 0],
         [1, 3, 3, 4, 4, 4, 0, 4]],

        [[1, 1, 1, 1, 0, 1, 0, 0],
         [1, 1, 1, 1, 0, 1, 0, 0],
         [0, 2, 1, 4, 3, 3, 2, 0]],

        [[1, 0, 1, 0, 0, 0, 0, 0],
         [2, 2, 2, 2, 2, 2, 2, 1],
         [4, 3, 0, 0, 4, 2, 0, 4]]]], dtype=uint64)
>>> xr = np.empty((r,n,d),dtype=np.float64) 
>>> time_perf,time_process = qmctoolscl.gdn_integer_to_float(r,n,d,r_b,tmax_new,bases,xrdig,xr)
>>> xr
array([[[0.140625  , 0.99283646, 0.7772288 ],
        [0.66796875, 0.37524768, 0.22977792],
        [0.46875   , 0.29934461, 0.859072  ],
        [0.76953125, 0.8311233 , 0.5431936 ],
        [0.015625  , 0.63603109, 0.0474624 ],
        [0.55859375, 0.07575065, 0.69515776],
        [0.25      , 0.72092669, 0.27969536],
        [0.98046875, 0.50617284, 0.9748864 ],
        [0.625     , 0.1949398 , 0.4404864 ]],

       [[0.046875  , 0.16156074, 0.560384  ],
        [0.66015625, 0.39521414, 0.2505728 ],
        [0.33203125, 0.70690444, 0.0192512 ],
        [0.8046875 , 0.28242646, 0.87810048],
        [0.203125  , 0.65996037, 0.64705792],
        [0.59375   , 0.80445054, 0.53750016],
        [0.41796875, 0.02240512, 0.35194624],
        [0.953125  , 0.4951989 , 0.0955776 ],
        [0.625     , 0.99969517, 0.92141824]]])
```

### digital interlacing 

```python
>>> print(qmctoolscl.gdn_interlace.__doc__)
Interlace generating matrices or transpose of point sets to attain higher order digital nets

Args:
    r (np.uint64): replications
    d_alpha (np.uint64): dimension of resulting generating matrices 
    mmax (np.uint64): columns of generating matrices
    d (np.uint64): dimension of original generating matrices
    tmax (np.uint64): rows of original generating matrices
    tmax_alpha (np.uint64): rows of interlaced generating matrices
    alpha (np.uint64): interlacing factor
    C (np.ndarray of np.uint64): original generating matrices of size r*d*mmax*tmax
    C_alpha (np.ndarray of np.uint64): resulting interlaced generating matrices of size r*d_alpha*mmax*tmax_alpha
>>> r = np.uint64(2)
>>> d = np.uint64(6)
>>> alpha = np.uint64(3)
>>> d_alpha = np.uint64(d//alpha)
>>> mmax = np.uint64(3)
>>> tmax = np.uint64(4)
>>> tmax_alpha = np.uint64(alpha*tmax)
>>> C = rng.integers(0,5,(r,d,mmax,tmax),dtype=np.uint64)
>>> C
array([[[[0, 4, 4, 0],
         [0, 2, 4, 0],
         [1, 2, 4, 4]],

        [[2, 4, 3, 4],
         [4, 0, 1, 2],
         [0, 2, 0, 0]],

        [[1, 1, 3, 1],
         [3, 3, 1, 1],
         [1, 3, 2, 0]],

        [[1, 1, 4, 2],
         [4, 2, 4, 4],
         [1, 0, 3, 1]],

        [[1, 4, 3, 0],
         [2, 4, 4, 4],
         [1, 4, 2, 4]],

        [[0, 2, 3, 1],
         [3, 1, 1, 0],
         [3, 0, 1, 0]]],


       [[[4, 4, 3, 2],
         [3, 3, 3, 0],
         [4, 3, 3, 3]],

        [[2, 3, 2, 0],
         [0, 4, 0, 2],
         [2, 2, 1, 0]],

        [[3, 0, 1, 4],
         [4, 1, 3, 3],
         [0, 3, 1, 0]],

        [[2, 0, 3, 1],
         [4, 1, 3, 1],
         [3, 1, 0, 4]],

        [[2, 0, 2, 0],
         [4, 0, 1, 3],
         [0, 1, 1, 1]],

        [[4, 4, 0, 1],
         [2, 3, 0, 3],
         [0, 1, 4, 0]]]], dtype=uint64)
>>> C_alpha = np.empty((r,d_alpha,mmax,tmax_alpha),dtype=np.uint64)
>>> time_perf,time_process = qmctoolscl.gdn_interlace(r,d_alpha,mmax,d,tmax,tmax_alpha,alpha,C,C_alpha)
>>> C_alpha
array([[[[0, 2, 1, 4, 4, 1, 4, 3, 3, 0, 4, 1],
         [0, 4, 3, 2, 0, 3, 4, 1, 1, 0, 2, 1],
         [1, 0, 1, 2, 2, 3, 4, 0, 2, 4, 0, 0]],

        [[1, 1, 0, 1, 4, 2, 4, 3, 3, 2, 0, 1],
         [4, 2, 3, 2, 4, 1, 4, 4, 1, 4, 4, 0],
         [1, 1, 3, 0, 4, 0, 3, 2, 1, 1, 4, 0]]],


       [[[4, 2, 3, 4, 3, 0, 3, 2, 1, 2, 0, 4],
         [3, 0, 4, 3, 4, 1, 3, 0, 3, 0, 2, 3],
         [4, 2, 0, 3, 2, 3, 3, 1, 1, 3, 0, 0]],

        [[2, 2, 4, 0, 0, 4, 3, 2, 0, 1, 0, 1],
         [4, 4, 2, 1, 0, 3, 3, 1, 0, 1, 3, 3],
         [3, 0, 0, 1, 1, 1, 0, 1, 4, 4, 1, 0]]]], dtype=uint64)
```

### undo digital interlacing 

```python 
>>> print(qmctoolscl.gdn_undo_interlace.__doc__)
Undo interlacing of generating matrices

Args:
    r (np.uint64): replications
    d (np.uint64): dimension of resulting generating matrices 
    mmax (np.uint64): columns in generating matrices
    d_alpha (np.uint64): dimension of interlaced generating matrices
    tmax (np.uint64): rows of original generating matrices
    tmax_alpha (np.uint64): rows of interlaced generating matrices
    alpha (np.uint64): interlacing factor
    C_alpha (np.ndarray of np.uint64): interlaced generating matrices of size r*d_alpha*mmax*tmax_alpha
    C (np.ndarray of np.uint64): original generating matrices of size r*d*mmax*tmax
>>> C_cp = np.empty((r,d,mmax,tmax),dtype=np.uint64)
>>> time_perf,time_process = qmctoolscl.gdn_undo_interlace(r,d,mmax,d_alpha,tmax,tmax_alpha,alpha,C_alpha,C_cp) 
>>> print((C==C_cp).all())
True
```

### natural order with the same base 

```python
>>> print(qmctoolscl.gdn_gen_natural_same_base.__doc__)
Generalized digital net with the same base for each dimension e.g. a digital net in base greater than 2

Args:
    r (np.uint64): replications
    n (np.uint64): points
    d (np.uint64): dimension
    mmax (np.uint64): columns in each generating matrix
    tmax (np.uint64): rows of each generating matrix
    n_start (np.uint64): starting index in sequence
    b (np.uint64): common base
    C (np.ndarray of np.uint64): generating matrices of size r*d*mmax*tmax
    xdig (np.ndarray of np.uint64): generalized digital net sequence of digits of size r*n*d*tmax
>>> n_start = np.uint64(0)
>>> n = np.uint64(9)
>>> b = np.uint64(3) 
>>> C = np.array([[
...     [
...         [1,1,0,0],
...         [1,1,1,1],
...     ],
...     [
...         [2,1,0,0],
...         [2,1,2,1],
...     ]
...     ]],
...     dtype=np.uint64)
>>> r = np.uint64(C.shape[0])
>>> d = np.uint64(C.shape[1])
>>> mmax = np.uint64(C.shape[2])
>>> tmax = np.uint64(C.shape[3])
>>> xdig = np.empty((r,n,d,tmax),dtype=np.uint64) 
>>> time_perf,time_process = qmctoolscl.gdn_gen_natural_same_base(r,n,d,mmax,tmax,n_start,b,C,xdig)
>>> xdig
array([[[[0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[1, 1, 0, 0],
         [2, 1, 0, 0]],

        [[2, 2, 0, 0],
         [1, 2, 0, 0]],

        [[1, 1, 1, 1],
         [2, 1, 2, 1]],

        [[2, 2, 1, 1],
         [1, 2, 2, 1]],

        [[0, 0, 1, 1],
         [0, 0, 2, 1]],

        [[2, 2, 2, 2],
         [1, 2, 1, 2]],

        [[0, 0, 2, 2],
         [0, 0, 1, 2]],

        [[1, 1, 2, 2],
         [2, 1, 1, 2]]]], dtype=uint64)
```

## Fast Transforms

Fast transforms require the use of a single work group for the final dimension i.e. it is required that `global_size[2]==local_size[2]`

```python 
>>> if kwargs["backend"]=="CL":
...     kwargs_ft = kwargs.copy()
...     kwargs_ft["global_size"] = (2,2,2)
...     kwargs_ft["local_size"] = (1,1,2)
... else: ## kwargs["backend"]=="C"
...     kwargs_ft = kwargs 
```

### (inverse) fast Fourier transform 

```python 
>>> print(qmctoolscl.fft_bro_1d_radix2.__doc__)
Fast Fourier Transform for inputs in bit reversed order.
FFT is done in place along the last dimension where the size is required to be a power of 2.
Follows a decimation-in-time procedure described in https://www.cmlab.csie.ntu.edu.tw/cml/dsp/training/coding/transform/fft.html.

Args:
    d1 (np.uint64): first dimension
    d2 (np.uint64): second dimension
    n_half (np.uint64): half of the last dimension of size n = 2n_half along which FFT is performed
    twiddler (np.ndarray of np.double): size n vector used to store real twiddle factors
    twiddlei (np.ndarray of np.double): size n vector used to store imaginary twiddle factors 
    xr (np.ndarray of np.double): real array of size d1*d2*n on which to perform FFT in place
    xi (np.ndarray of np.double): imaginary array of size d1*d2*n on which to perform FFT in place
>>> print(qmctoolscl.ifft_bro_1d_radix2.__doc__)
Inverse Fast Fourier Transform with outputs in bit reversed order.
FFT is done in place along the last dimension where the size is required to be a power of 2.
Follows a procedure described in https://www.expertsmind.com/learning/inverse-dft-using-the-fft-algorithm-assignment-help-7342873886.aspx.

Args:
    d1 (np.uint64): first dimension
    d2 (np.uint64): second dimension
    n_half (np.uint64): half of the last dimension of size n = 2n_half along which FFT is performed
    twiddler (np.ndarray of np.double): size n vector used to store real twiddle factors
    twiddlei (np.ndarray of np.double): size n vector used to store imaginary twiddle factors 
    xr (np.ndarray of np.double): real array of size d1*d2*n on which to perform FFT in place
    xi (np.ndarray of np.double): imaginary array of size d1*d2*n on which to perform FFT in place
>>> d1 = np.uint64(1) 
>>> d2 = np.uint64(1)
>>> xr = np.array([1,0,1,0,0,1,1,0],dtype=np.double)
>>> xi = np.array([0,1,0,1,1,0,0,1],dtype=np.double)
>>> xr_og = xr.copy()
>>> xi_og = xi.copy()
>>> twiddler = np.empty_like(xr,dtype=np.double)
>>> twiddlei = np.empty_like(xr,dtype=np.double)
>>> n_half = np.uint64(len(xr)//2)
>>> time_perf,time_process = qmctoolscl.fft_bro_1d_radix2(d1,d2,n_half,twiddler,twiddlei,xr,xi,**kwargs_ft)
>>> xr
array([ 1.41421356, -0.5       ,  0.        ,  1.20710678,  0.        ,
        0.5       ,  0.        ,  0.20710678])
>>> xi
array([ 1.41421356, -0.20710678,  0.        , -0.5       ,  0.        ,
       -1.20710678,  0.        ,  0.5       ])
>>> time_perf,time_process = qmctoolscl.ifft_bro_1d_radix2(d1,d2,n_half,twiddler,twiddlei,xr,xi,**kwargs_ft)
>>> np.allclose(xr,xr_og,atol=1e-8)
True
>>> np.allclose(xi,xi_og,atol=1e-8)
True
```

```python
>>> d1 = np.uint(5) 
>>> d2 = np.uint(7) 
>>> m = 8
>>> n = 2**m
>>> n_half = np.uint(n//2)
>>> bitrev = np.vectorize(lambda i,m: int('{:0{m}b}'.format(i,m=m)[::-1],2))
>>> ir = bitrev(np.arange(n),m) 
>>> xr = rng.uniform(0,1,(d1,d2,int(2*n_half))).astype(np.double)
>>> xi = rng.uniform(0,1,(d1,d2,int(2*n_half))).astype(np.double)
>>> twiddler = np.empty_like(xr,dtype=np.double)
>>> twiddlei = np.empty_like(xr,dtype=np.double)
>>> x_np = xr+1j*xi
>>> x_bro_np = x_np[:,:,ir]
>>> y = xr+1j*xi
>>> yt_np = np.fft.fft(x_bro_np,norm="ortho")
>>> time_perf,time_process = qmctoolscl.fft_bro_1d_radix2(d1,d2,n_half,twiddler,twiddlei,xr,xi,**kwargs_ft)
>>> yt = xr+1j*xi
>>> np.allclose(yt,yt_np,atol=1e-8)
True
>>> time_perf,time_process = qmctoolscl.ifft_bro_1d_radix2(d1,d2,n_half,twiddler,twiddlei,xr,xi,**kwargs_ft)
>>> np.allclose(xr+1j*xi,y,atol=1e-8)
True
```

```python
>>> fft_np = lambda x: np.fft.fft(x,norm="ortho")
>>> ifft_np = lambda x: np.fft.ifft(x,norm="ortho")
>>> def fft(x):
...     assert x.ndim==1
...     n = len(x)
...     n_half = np.uint64(n//2)
...     xr = x.real.copy()
...     xi = x.imag.copy()
...     qmctoolscl.fft_bro_1d_radix2(1,1,n_half,np.empty(n,dtype=np.double),np.empty(n,dtype=np.double),xr,xi)
...     return xr+1j*xi
>>> def ifft(x):
...     assert x.ndim==1
...     n = len(x)
...     n_half = np.uint64(n//2)
...     xr = x.real.copy()
...     xi = x.imag.copy()
...     qmctoolscl.ifft_bro_1d_radix2(1,1,n_half,np.empty(n,dtype=np.double),np.empty(n,dtype=np.double),xr,xi)
...     return xr+1j*xi
>>> # parameters 
>>> m = 10
>>> n = 2**m
>>> # bit reverse used for reference solver 
>>> bitrev = np.vectorize(lambda i,m: int('{:0{m}b}'.format(i,m=m)[::-1],2))
>>> ir = bitrev(np.arange(2*n),m+1)
>>> # points 
>>> y1 = np.random.rand(n)+1j*np.random.rand(n)
>>> y2 = np.random.rand(n)+1j*np.random.rand(n)
>>> y = np.hstack([y1,y2])
>>> # kernel evaluations
>>> k1 = np.random.rand(n) 
>>> k2 = np.random.rand(n) 
>>> k = np.hstack([k1,k2]) 
>>> # fast transforms 
>>> yt = fft(y) 
>>> kt = fft(k)
>>> y1t = fft(y1) 
>>> y2t = fft(y2) 
>>> k1t = fft(k1) 
>>> k2t = fft(k2)
>>> wt = np.exp(-np.pi*1j*np.arange(n)/n)
>>> wtsq = wt**2
>>> gammat = k1t**2-wtsq*k2t**2
>>> # matrix vector product
>>> u = ifft(yt*kt)*np.sqrt(2*n)
>>> u_np = ifft_np(fft_np(y[ir])*fft_np(k[ir]))[ir]*np.sqrt(2*n)
>>> np.allclose(u,u_np,atol=1e-10)
True
>>> u_hat = np.hstack([ifft(y1t*k1t+wtsq*y2t*k2t),ifft(y2t*k1t+y1t*k2t)])*np.sqrt(n)
>>> np.allclose(u_hat,u,atol=1e-10)
True
>>> # inverse 
>>> v = ifft(yt/kt)/np.sqrt(2*n)
>>> v_np = ifft_np(fft_np(y[ir])/fft_np(k[ir]))[ir]/np.sqrt(2*n)
>>> np.allclose(v,v_np,atol=1e-10)
True
>>> v_hat = np.hstack([ifft((y1t*k1t-wtsq*y2t*k2t)/gammat),ifft((y2t*k1t-y1t*k2t)/gammat)])/np.sqrt(n)
>>> np.allclose(v,v_hat,atol=1e-10)
True
```

### fast Walsh-Hadamard transform

```python 
>>> print(qmctoolscl.fwht_1d_radix2.__doc__)
Fast Walsh-Hadamard Transform for real valued inputs.
FWHT is done in place along the last dimension where the size is required to be a power of 2.
Follows the divide-and-conquer algorithm described in https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform

Args:
    d1 (np.uint64): first dimension
    d2 (np.uint64): second dimension
    n_half (np.uint64): half of the last dimension along which FWHT is performed
    x (np.ndarray of np.double): array of size d1*d2*2n_half on which to perform FWHT in place
>>> d1 = np.uint64(1) 
>>> d2 = np.uint64(1)
>>> x = np.array([1,0,1,0,0,1,1,0],dtype=np.double)
>>> n_half = np.uint64(len(x)//2)
>>> time_perf,time_process = qmctoolscl.fwht_1d_radix2(d1,d2,n_half,x,**kwargs_ft)
>>> x
array([ 1.41421356,  0.70710678,  0.        , -0.70710678,  0.        ,
        0.70710678,  0.        ,  0.70710678])
```

```python 
>>> d1 = np.uint(5) 
>>> d2 = np.uint(7)
>>> n = 2**8
>>> n_half = np.uint(n//2) 
>>> x = rng.uniform(0,1,(d1,d2,int(2*n_half))).astype(np.double)
>>> x_og = x.copy()
>>> import sympy
>>> y_sympy = np.empty_like(x,dtype=np.double) 
>>> for i in range(d1):
...     for j in range(d2): 
...         y_sympy[i,j] = np.array(sympy.fwht(x[i,j])/np.sqrt(n),dtype=np.double)
>>> time_perf,time_process = qmctoolscl.fwht_1d_radix2(d1,d2,n_half,x,**kwargs_ft)
>>> np.allclose(x,y_sympy,atol=1e-8)
True
>>> time_perf,time_process = qmctoolscl.fwht_1d_radix2(d1,d2,n_half,x,**kwargs_ft)
>>> np.allclose(x,x_og,atol=1e-8)
True
```

```python
>>> def fwht_sp(x):
...     assert x.ndim==1
...     n = len(x) 
...     y = np.array(sympy.fwht(x),dtype=np.float64)
...     y_ortho = y/np.sqrt(n) 
...     return y_ortho
>>> def fwht(x):
...     assert x.ndim==1
...     n = len(x)
...     n_half = np.uint64(n//2)
...     x_cp = x.copy()
...     qmctoolscl.fwht_1d_radix2(1,1,n_half,x_cp)
...     return x_cp
>>> # parameters
>>> m = 10
>>> n = int(2**m)
>>> # points
>>> y1 = np.random.rand(n) 
>>> y2 = np.random.rand(n) 
>>> y = np.hstack([y1,y2])
>>> # kernel evaluations
>>> k1 = np.random.rand(n) 
>>> k2 = np.random.rand(n) 
>>> k = np.hstack([k1,k2]) 
>>> # fast transforms
>>> yt = fwht(y) 
>>> kt = fwht(k)
>>> y1t = fwht(y1) 
>>> y2t = fwht(y2) 
>>> k1t = fwht(k1) 
>>> k2t = fwht(k2)
>>> gammat = k1t**2-k2t**2
>>> # matrix vector product
>>> u_sp = fwht_sp(fwht_sp(y)*fwht_sp(k))*np.sqrt(2*n)
>>> u = fwht(yt*kt)*np.sqrt(2*n)
>>> np.allclose(u_sp,u,atol=1e-8)
True
>>> u_hat = np.hstack([fwht(y1t*k1t+y2t*k2t),fwht(y2t*k1t+y1t*k2t)])*np.sqrt(n)
>>> np.allclose(u,u_hat,atol=1e-10)
True
>>> # inverse 
>>> v_sp = fwht_sp(fwht_sp(y)/fwht_sp(k))/np.sqrt(2*n)
>>> v = fwht(yt/kt)/np.sqrt(2*n)
>>> np.allclose(v_sp,v,atol=1e-8)
True
>>> v_hat = np.hstack([fwht((y1t*k1t-y2t*k2t)/gammat),fwht((y2t*k1t-y1t*k2t)/gammat)])/np.sqrt(n)
>>> np.allclose(v,v_hat,atol=1e-10)
True
```
