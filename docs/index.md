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

<h1>Full Doctests for QMCseqCL</h1>

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
>>> print(qmcseqcl.lat_gen_linear.__doc__)
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
>>> time_perf,time_process = qmcseqcl.lat_gen_linear(r,n,d,g,x,**kwargs)
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
>>> print(qmcseqcl.lat_gen_natural_gray.__doc__)
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
>>> time_perf,time_process = qmcseqcl.lat_gen_natural_gray(r,n,d,n_start,gc,g,x,**kwargs)
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
>>> time_perf,time_process = qmcseqcl.lat_gen_natural_gray(r,n,d,n_start,gc,g,x,**kwargs)
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
>>> print(qmcseqcl.lat_shift_mod_1.__doc__)
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
>>> time_perf,time_process = qmcseqcl.lat_shift_mod_1(r,n,d,r_x,x,shifts,xr,**kwargs)
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
>>> print(qmcseqcl.dnb2_gmat_lsb_to_msb.__doc__)
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
>>> time_perf,time_process = qmcseqcl.dnb2_gmat_lsb_to_msb(r,d,mmax,tmaxes,C_lsb,C,**kwargs)
>>> C
array([[ 8,  4,  2,  1],
       [ 8, 12, 10, 15],
       [ 8, 12,  6,  9],
       [ 8, 12,  2,  5]], dtype=uint64)
```

## linear matrix scrambling (LMS)

```python
>>> print(qmcseqcl.dnb2_linear_matrix_scramble.__doc__)
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
>>> r = 2*r_C
>>> tmax_new = np.uint64(6)
>>> print(qmcseqcl.dnb2_get_linear_scramble_matrix.__doc__)
Return a scrambling matrix for linear matrix scrambling

Args:
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications
    d (np.uint64): dimension
    tmax (np.uint64): bits in each integer
    tmax_new (np.uint64): bits in each integer of the generating matrix after scrambling
    print_mats (np.uint8): flag to print the resulting matrices
>>> print_mats = np.uint8(True)
>>> S = qmcseqcl.dnb2_get_linear_scramble_matrix(rng,r,d,tmax,tmax_new,print_mats)
S with shape (r=2, d=4, tmax_new=6)
l = 0
    j = 0
        1000
        1100
        1010
        1011
        0000
        1011
    j = 1
        1000
        1100
        1010
        1111
        0101
        0111
    j = 2
        1000
        0100
        0010
        1001
        0001
        0111
    j = 3
        1000
        1100
        1010
        1001
        0010
        0010
l = 1
    j = 0
        1000
        0100
        0010
        0101
        1101
        1000
    j = 1
        1000
        0100
        0110
        1001
        1010
        1001
    j = 2
        1000
        0100
        0110
        0111
        1101
        1110
    j = 3
        1000
        0100
        0110
        0011
        1000
        0000
>>> C_lms = np.empty((r,d,mmax),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.dnb2_linear_matrix_scramble(r,d,mmax,r_C,tmax_new,S,C,C_lms,**kwargs)
>>> C_lms
array([[[61, 16, 13,  5],
        [60, 43, 49, 33],
        [36, 53, 24, 35],
        [60, 44, 11, 20]],

       [[35, 22,  8,  6],
        [39, 63, 45, 48],
        [35, 60, 18, 37],
        [34, 58, 12, 28]]], dtype=uint64)
```

## digital interlacing 

Digital interlacing is used to create **higher order digital nets**.

```python
>>> print(qmcseqcl.dnb2_interlace.__doc__)
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
>>> time_perf,time_process = qmcseqcl.dnb2_interlace(r,d_alpha,mmax,d,tmax,tmax_alpha,alpha,C_lms,C_alpha,**kwargs)
>>> C_alpha
array([[[4082, 1605, 1443, 1059],
        [3440, 3698,  709, 2330]],

       [[3103, 1917, 1233, 1320],
        [3086, 4068,  600, 2418]]], dtype=uint64)
``` 

## undo digital interlacing 

```python
>>> print(qmcseqcl.dnb2_undo_interlace.__doc__)
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
>>> time_perf,time_process = qmcseqcl.dnb2_undo_interlace(r,d,mmax,d_alpha,tmax,tmax_alpha,alpha,C_alpha,C_lms_cp,**kwargs)
>>> print((C_lms_cp==C_lms).all())
True
```

## generate Gray code or natural order 

```python
>>> print(qmcseqcl.dnb2_gen_natural_gray.__doc__)
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
>>> time_perf,time_process = qmcseqcl.dnb2_gen_natural_gray(r,n,d,n_start,gc,mmax,C,xb,**kwargs)
>>> xb
array([[[1605, 3698],
        [2487,  770],
        [1443,  709],
        [2641, 4021],
        [ 998, 3255],
        [3092,  455],
        [1059, 2330],
        [3025, 1130],
        [ 614, 1896],
        [3476, 2584],
        [ 384, 3039],
        [3698, 1711],
        [1989, 1453],
        [2103, 2269]],

       [[1917, 4068],
        [2914, 1002],
        [1233,  600],
        [2254, 3670],
        [ 940, 3516],
        [4019,  434],
        [1320, 2418],
        [2359, 1404],
        [ 597, 1686],
        [3658, 2712],
        [ 505, 2858],
        [3558, 1828],
        [1668, 1230],
        [2715, 2240]]], dtype=uint64)
>>> gc = np.uint8(True)
>>> time_perf,time_process = qmcseqcl.dnb2_gen_natural_gray(r,n,d,n_start,gc,mmax,C,xb,**kwargs)
>>> xb
array([[[2487,  770],
        [1605, 3698],
        [ 998, 3255],
        [3092,  455],
        [2641, 4021],
        [1443,  709],
        [ 384, 3039],
        [3698, 1711],
        [2103, 2269],
        [1989, 1453],
        [ 614, 1896],
        [3476, 2584],
        [3025, 1130],
        [1059, 2330]],

       [[2914, 1002],
        [1917, 4068],
        [ 940, 3516],
        [4019,  434],
        [2254, 3670],
        [1233,  600],
        [ 505, 2858],
        [3558, 1828],
        [2715, 2240],
        [1668, 1230],
        [ 597, 1686],
        [3658, 2712],
        [2359, 1404],
        [1320, 2418]]], dtype=uint64)
```

## digital shift 

```python
>>> print(qmcseqcl.dnb2_digital_shift.__doc__)
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
    rng (np.random._generator.Generator): random number generator
    t: (int): number of bits with 0 <= t <= 64
    shape (tuple of ints): shape of resulting integer array
>>> shiftsb = qmcseqcl.random_tbit_uint64s(rng,tmax_new,(r,d))
>>> shiftsb
array([[ 5310653692329262186, 13368902947290702765],
       [14338104955770306630,   858526716083693676],
       [ 8506091904275275003, 16209688535125722604],
       [17447177601860792486,  9021077006476608718]], dtype=uint64)
>>> xrb = np.empty((r,n,d),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.dnb2_digital_shift(r,n,d,r_x,lshifts,xb,shiftsb,xrb,**kwargs)
>>> xrb
array([[[15187047675152759914,  9919145632724902829],
        [ 3306551858149391466,  6820669089094001581],
        [ 8634310217328688234,  8284338967989412781],
        [ 9868296515228204138, 11959276263923737517],
        [17051537920884145258,  4816567254914130861],
        [ 1406032815399042154, 10797347560062149549],
        [ 5887114444632685674,   321974826798375853],
        [12579463490905242730, 15237896792649458605],
        [14610586922849336426,  3771732141364175789],
        [ 3883012610452814954, 16381811098001564589],
        [ 8057849465025264746, 14918141219106153389],
        [10444757267531627626,  1731601510165341101],
        [17627998673187568746, 18385912932181435309],
        [  829572063095618666,  2893530214026929069]],

       [[ 8132144669253763142,  3839909669402962028],
        [12766348685818003526, 17701989322449348716],
        [18175171838289969222, 14999829546027051116],
        [ 4452703773692067910,  1209807487018592364],
        [ 5339912900284055622, 17188578964929112172],
        [10082203307905187910,  3344513710392207468],
        [15666666845844602950, 13351512082409449580],
        [ 1773061995406622790,  8766847661746284652],
        [ 8019554678569500742,  9793668376786757740],
        [12590708300350554182,  5118931963576182892],
        [16405257184733364294,  7100515799619201132],
        [ 2475623537276420166, 11703194618791848044],
        [ 6164071632092856390,  6641148637627410540],
        [10699196456854945862, 11297870652328503404]],

       [[17112470792180292859, 15047759831264134636],
        [ 1322850498619333883,   564183429640619500],
        [ 5218464176294812923,  3135738816869172716],
        [13207849915250072827, 18195775970796111340],
        [15211951749429943547,  1991824511517066732],
        [ 3187340744350719227, 14746018656230311404],
        [ 7929631151971851515,  6702589721746605548],
        [10460654142554070267,  9945181453453362668],
        [17688931544483716347,  7864518425608193516],
        [  746389746315910395, 13412953166528644588],
        [ 5794924928598236411, 10841397779300091372],
        [12631389162946649339,  4716502286076216812],
        [14635490997126520059, 11985312084652197356],
        [ 3763801496654142715,  8166259600642016748]],

       [[ 4900149040006590630,  4868758150041011406],
        [ 9651446646882463910,  9471436969213658318],
        [14474801847796265126, 12029481557560100046],
        [  653254591396212902,  7354745144349525198],
        [ 9133532689734856870, 10975639244755403982],
        [13776743905553838246,  6390974824092239054],
        [17127422028317487270, 14956821315350922446],
        [ 3188781181605802150,  1112756060814017742],
        [ 6598006099525267622, 17379757914876249294],
        [11124123725032616102,  3589735855867790542],
        [15524140560973590694,  1464036831748916430],
        [ 1621528511280869542, 15326116484795303118],
        [ 7012337265243353254,  3094339896857035982],
        [11574483687769665702, 16866347557356012750]]], dtype=uint64)
```

## convert digits to doubles

```python 
>>> print(qmcseqcl.dnb2_integer_to_float.__doc__)
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
>>> time_perf,time_process = qmcseqcl.dnb2_integer_to_float(r,n,d,tmaxes_new,xrb,x,**kwargs)
>>> x
array([[[0.8232915 , 0.53771796],
        [0.17924854, 0.36974921],
        [0.46806689, 0.44909492],
        [0.53496143, 0.64831367],
        [0.92436572, 0.26110663],
        [0.07622119, 0.58532538],
        [0.31914111, 0.01745429],
        [0.68193408, 0.82604804],
        [0.7920415 , 0.20446601],
        [0.21049854, 0.88805976],
        [0.43681689, 0.80871406],
        [0.56621143, 0.09387031],
        [0.95561572, 0.99670234],
        [0.04497119, 0.15685859]],

       [[0.44084445, 0.20816192],
        [0.69206515, 0.95962676],
        [0.98527804, 0.81314239],
        [0.24138156, 0.0655838 ],
        [0.28947726, 0.93179473],
        [0.54655734, 0.18130645],
        [0.84929171, 0.72378692],
        [0.09611788, 0.47525176],
        [0.43474093, 0.53091583],
        [0.68254366, 0.27749786],
        [0.88933077, 0.38491973],
        [0.13420382, 0.63443145],
        [0.33415499, 0.36001739],
        [0.5800046 , 0.6124588 ]],

       [[0.9276689 , 0.8157407 ],
        [0.07171187, 0.03058445],
        [0.28289351, 0.16998874],
        [0.71599898, 0.98639499],
        [0.82464156, 0.10797702],
        [0.17278609, 0.79938327],
        [0.42986617, 0.36334812],
        [0.5670732 , 0.53912937],
        [0.9589189 , 0.4263364 ],
        [0.04046187, 0.72711765],
        [0.31414351, 0.58771335],
        [0.68474898, 0.2556821 ],
        [0.79339156, 0.64972507],
        [0.20403609, 0.44269382]],

       [[0.26563761, 0.26393591],
        [0.52320597, 0.51344763],
        [0.78468058, 0.6521195 ],
        [0.035413  , 0.39870153],
        [0.4951298 , 0.5949906 ],
        [0.74683878, 0.34645544],
        [0.92847941, 0.81081091],
        [0.17286417, 0.06032263],
        [0.35767863, 0.94215856],
        [0.60303996, 0.19459997],
        [0.84156535, 0.0793656 ],
        [0.08790324, 0.83083044],
        [0.38013956, 0.1677445 ],
        [0.62745402, 0.91432653]]])
```

## nested uniform scrambling 

```python
>>> r = np.uint64(1) 
>>> n_start = np.uint64(0)
>>> n = np.uint64(8)
>>> d = np.uint64(4)
>>> gc = np.uint8(False)
>>> C = np.array([
...   [ 8,  4,  2,  1],
...   [ 8, 12, 10, 15],
...   [ 8, 12,  6,  9],
...   [ 8, 12,  2,  5]], 
...   dtype=np.uint64)
>>> mmax = np.uint64(C.shape[1])
>>> tmax = np.uint64(np.ceil(np.log2(np.max(C))))
>>> xb = np.empty((r,n,d),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.dnb2_gen_natural_gray(r,n,d,n_start,gc,mmax,C,xb,**kwargs)
>>> print(qmcseqcl.dnb2_nested_uniform_scramble.__doc__)
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
>>> root_nodes = np.array([qmcseqcl.NUSNode_dnb2() for i in range(r*d)]).reshape(r,d)
>>> xrb = np.zeros((r,n,d),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.dnb2_nested_uniform_scramble(r,n,d,r_x,tmax,tmax_new,rngs,root_nodes,xb,xrb)
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

# Generalized Digital Net 

Accommodates both Halton sequences and digital nets in any prime base

## Linear Matrix Scramble 

```python
>>> print(qmcseqcl.gdn_linear_matrix_scramble.__doc__)
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
>>> print(qmcseqcl.gdn_get_halton_generating_matrix.__doc__)
Return the identity matrices comprising the Halton generating matrices

Arg:
    r (np.uint64): replications 
    d (np.uint64): dimension 
    mmax (np.uint64): maximum number rows and columns in each generating matrix
>>> C = qmcseqcl.gdn_get_halton_generating_matrix(r_C,d,mmax)
>>> tmax_new = np.uint64(2*tmax)
>>> r = np.uint64(2*r_C)
>>> print(qmcseqcl.gdn_get_linear_scramble_matrix.__doc__)
Return a scrambling matrix for linear matrix scrambling

Args:
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications
    d (np.uint64): dimension
    tmax (np.uint64): bits in each integer
    tmax_new (np.uint64): bits in each integer of the generating matrix after scrambling
    r_b (np.uint64): replications of bases 
    bases (np.ndarray of np.uint64): bases of size r_b*d
>>> S = qmcseqcl.gdn_get_linear_scramble_matrix(rng,r,d,tmax,tmax_new,r_b,bases)
>>> S
array([[[[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [1, 0, 0, 0, 1],
         [1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [1, 0, 0, 1, 1],
         [1, 0, 1, 0, 1],
         [1, 1, 0, 1, 0]],

        [[2, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [1, 0, 1, 0, 0],
         [2, 0, 1, 2, 0],
         [2, 1, 1, 0, 1],
         [2, 0, 2, 2, 0],
         [2, 2, 1, 0, 2],
         [2, 0, 2, 0, 1],
         [0, 2, 2, 0, 1],
         [2, 0, 0, 1, 0]]],


       [[[3, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [4, 2, 4, 0, 0],
         [4, 2, 1, 2, 0],
         [3, 4, 4, 3, 1],
         [3, 3, 1, 1, 4],
         [2, 2, 0, 4, 2],
         [3, 3, 3, 1, 3],
         [3, 3, 1, 2, 0],
         [4, 3, 1, 1, 0]],

        [[3, 0, 0, 0, 0],
         [6, 1, 0, 0, 0],
         [1, 0, 2, 0, 0],
         [1, 4, 4, 1, 0],
         [4, 4, 1, 2, 1],
         [3, 4, 1, 6, 5],
         [1, 3, 5, 1, 5],
         [2, 0, 4, 3, 1],
         [4, 0, 0, 1, 3],
         [6, 0, 4, 3, 6]]],


       [[[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 0, 0, 1, 1],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 1],
         [1, 1, 0, 0, 0],
         [1, 0, 0, 1, 1],
         [1, 0, 1, 1, 1]],

        [[1, 0, 0, 0, 0],
         [2, 1, 0, 0, 0],
         [2, 1, 2, 0, 0],
         [0, 0, 1, 1, 0],
         [2, 2, 2, 1, 1],
         [1, 1, 2, 2, 0],
         [2, 1, 0, 2, 2],
         [0, 0, 2, 0, 2],
         [2, 1, 0, 2, 0],
         [2, 0, 1, 0, 2]]],


       [[[2, 0, 0, 0, 0],
         [1, 4, 0, 0, 0],
         [1, 3, 3, 0, 0],
         [1, 3, 1, 1, 0],
         [1, 3, 1, 1, 4],
         [0, 2, 3, 3, 0],
         [1, 3, 3, 3, 4],
         [4, 1, 2, 4, 2],
         [1, 4, 0, 1, 3],
         [1, 3, 3, 1, 4]],

        [[3, 0, 0, 0, 0],
         [1, 5, 0, 0, 0],
         [5, 2, 6, 0, 0],
         [1, 4, 4, 4, 0],
         [4, 3, 4, 6, 3],
         [5, 6, 6, 1, 3],
         [1, 1, 5, 2, 3],
         [5, 6, 0, 3, 6],
         [0, 2, 5, 5, 6],
         [1, 1, 1, 2, 0]]]], dtype=uint64)
>>> C_lms = np.empty((r,d,mmax,tmax_new),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.gdn_linear_matrix_scramble(r,d,mmax,r_C,r_b,tmax,tmax_new,bases,S,C,C_lms,**kwargs)
>>> C_lms 
array([[[[1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
         [0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
         [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
         [0, 0, 0, 0, 1, 0, 0, 1, 1, 0]],

        [[2, 0, 1, 2, 2, 2, 2, 2, 0, 2],
         [0, 1, 0, 0, 1, 0, 2, 0, 2, 0],
         [0, 0, 1, 1, 1, 2, 1, 2, 2, 0],
         [0, 0, 0, 2, 0, 2, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 2, 1, 1, 0]]],


       [[[3, 1, 4, 4, 3, 3, 2, 3, 3, 4],
         [0, 1, 2, 2, 4, 3, 2, 3, 3, 3],
         [0, 0, 4, 1, 4, 1, 0, 3, 1, 1],
         [0, 0, 0, 2, 3, 1, 4, 1, 2, 1],
         [0, 0, 0, 0, 1, 4, 2, 3, 0, 0]],

        [[3, 6, 1, 1, 4, 3, 1, 2, 4, 6],
         [0, 1, 0, 4, 4, 4, 3, 0, 0, 0],
         [0, 0, 2, 4, 1, 1, 5, 4, 0, 4],
         [0, 0, 0, 1, 2, 6, 1, 3, 1, 3],
         [0, 0, 0, 0, 1, 5, 5, 1, 3, 6]]],


       [[[1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1, 0, 1, 0, 1, 1]],

        [[1, 2, 2, 0, 2, 1, 2, 0, 2, 2],
         [0, 1, 1, 0, 2, 1, 1, 0, 1, 0],
         [0, 0, 2, 1, 2, 2, 0, 2, 0, 1],
         [0, 0, 0, 1, 1, 2, 2, 0, 2, 0],
         [0, 0, 0, 0, 1, 0, 2, 2, 0, 2]]],


       [[[2, 1, 1, 1, 1, 0, 1, 4, 1, 1],
         [0, 4, 3, 3, 3, 2, 3, 1, 4, 3],
         [0, 0, 3, 1, 1, 3, 3, 2, 0, 3],
         [0, 0, 0, 1, 1, 3, 3, 4, 1, 1],
         [0, 0, 0, 0, 4, 0, 4, 2, 3, 4]],

        [[3, 1, 5, 1, 4, 5, 1, 5, 0, 1],
         [0, 5, 2, 4, 3, 6, 1, 6, 2, 1],
         [0, 0, 6, 4, 4, 6, 5, 0, 5, 1],
         [0, 0, 0, 4, 6, 1, 2, 3, 5, 2],
         [0, 0, 0, 0, 3, 3, 3, 6, 6, 0]]]], dtype=uint64)
```

## generate natural order

```python
>>> print(qmcseqcl.gdn_gen_natural.__doc__)
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
>>> time_perf,time_process = qmcseqcl.gdn_gen_natural(r,n,d,r_b,mmax,tmax,n_start,bases,C,xdig,**kwargs)
>>> xdig
array([[[[0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
         [1, 0, 2, 1, 1, 1, 1, 1, 0, 1]],

        [[1, 0, 0, 0, 1, 1, 1, 1, 1, 0],
         [0, 1, 0, 0, 1, 0, 2, 0, 2, 0]],

        [[0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
         [2, 1, 1, 2, 0, 2, 1, 2, 2, 2]],

        [[1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
         [1, 1, 2, 1, 2, 1, 0, 1, 2, 1]],

        [[0, 1, 1, 0, 0, 0, 1, 0, 1, 1],
         [0, 2, 0, 0, 2, 0, 1, 0, 1, 0]],

        [[1, 0, 1, 0, 1, 1, 1, 1, 0, 0],
         [2, 2, 1, 2, 1, 2, 0, 2, 1, 2]]],


       [[[1, 2, 3, 3, 1, 1, 4, 1, 1, 3],
         [6, 5, 2, 2, 1, 6, 2, 4, 1, 5]],

        [[4, 3, 2, 2, 4, 4, 1, 4, 4, 2],
         [2, 4, 3, 3, 5, 2, 3, 6, 5, 4]],

        [[2, 4, 1, 1, 2, 2, 3, 2, 2, 1],
         [5, 3, 4, 4, 2, 5, 4, 1, 2, 3]],

        [[0, 1, 2, 2, 4, 3, 2, 3, 3, 3],
         [1, 2, 5, 5, 6, 1, 5, 3, 6, 2]],

        [[3, 2, 1, 1, 2, 1, 4, 1, 1, 2],
         [4, 1, 6, 6, 3, 4, 6, 5, 3, 1]],

        [[1, 3, 0, 0, 0, 4, 1, 4, 4, 1],
         [0, 1, 0, 4, 4, 4, 3, 0, 0, 0]]],


       [[[0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
         [2, 1, 1, 0, 1, 2, 1, 0, 1, 1]],

        [[1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
         [0, 1, 1, 0, 2, 1, 1, 0, 1, 0]],

        [[0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 1, 2, 0, 0, 0, 2]],

        [[1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
         [2, 2, 2, 0, 0, 0, 2, 0, 2, 1]],

        [[0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
         [0, 2, 2, 0, 1, 2, 2, 0, 2, 0]],

        [[1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
         [1, 1, 1, 0, 0, 0, 1, 0, 1, 2]]],


       [[[4, 2, 2, 2, 2, 0, 2, 3, 2, 2],
         [6, 2, 3, 2, 1, 3, 2, 3, 0, 2]],

        [[1, 3, 3, 3, 3, 0, 3, 2, 3, 3],
         [2, 3, 1, 3, 5, 1, 3, 1, 0, 3]],

        [[3, 4, 4, 4, 4, 0, 4, 1, 4, 4],
         [5, 4, 6, 4, 2, 6, 4, 6, 0, 4]],

        [[0, 4, 3, 3, 3, 2, 3, 1, 4, 3],
         [1, 5, 4, 5, 6, 4, 5, 4, 0, 5]],

        [[2, 0, 4, 4, 4, 2, 4, 0, 0, 4],
         [4, 6, 2, 6, 3, 2, 6, 2, 0, 6]],

        [[4, 1, 0, 0, 0, 2, 0, 4, 1, 0],
         [0, 5, 2, 4, 3, 6, 1, 6, 2, 1]]]], dtype=uint64)
```

## digital shift 

```python
>>> print(qmcseqcl.gdn_digital_shift.__doc__)
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
>>> print(qmcseqcl.gdn_get_digital_shifts.__doc__)
Return digital shifts for gdn

Args: 
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications 
    d (np.uint64): dimension 
    tmax_new (np.uint64): number of bits in each shift 
    r_b (np.uint64): replications of bases 
    bases (np.ndarray of np.uint64): bases of size r_b*d
>>> shifts = qmcseqcl.gdn_get_digital_shifts(rng,r,d,tmax_new,r_b,bases)
>>> shifts
array([[[0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1],
        [1, 2, 0, 2, 2, 0, 0, 0, 1, 0, 0, 1]],

       [[4, 0, 0, 1, 2, 0, 0, 1, 0, 3, 2, 1],
        [1, 3, 6, 5, 0, 2, 5, 2, 1, 6, 6, 6]],

       [[1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 2, 0, 2, 2, 2, 0, 1, 1]],

       [[0, 4, 1, 0, 2, 2, 2, 2, 1, 1, 1, 1],
        [5, 5, 1, 3, 0, 4, 5, 4, 3, 0, 3, 2]]], dtype=uint64)
>>> xdig_new = np.empty((r,n,d,tmax_new),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.gdn_digital_shift(r,n,d,r_x,r_b,tmax,tmax_new,bases,shifts,xdig,xdig_new,**kwargs)
>>> xdig_new
array([[[[0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
         [2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 0, 1]],

        [[1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 1]],

        [[0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
         [0, 0, 1, 1, 2, 2, 1, 2, 0, 2, 0, 1]],

        [[1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
         [2, 0, 2, 0, 1, 1, 0, 1, 0, 1, 0, 1]],

        [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
         [1, 1, 0, 2, 1, 0, 1, 0, 2, 0, 0, 1]],

        [[1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
         [0, 1, 1, 1, 0, 2, 0, 2, 2, 2, 0, 1]]],


       [[[0, 2, 3, 4, 3, 1, 4, 2, 1, 1, 2, 1],
         [0, 1, 1, 0, 1, 1, 0, 6, 2, 4, 6, 6]],

        [[3, 3, 2, 3, 1, 4, 1, 0, 4, 0, 2, 1],
         [3, 0, 2, 1, 5, 4, 1, 1, 6, 3, 6, 6]],

        [[1, 4, 1, 2, 4, 2, 3, 3, 2, 4, 2, 1],
         [6, 6, 3, 2, 2, 0, 2, 3, 3, 2, 6, 6]],

        [[4, 1, 2, 3, 1, 3, 2, 4, 3, 1, 2, 1],
         [2, 5, 4, 3, 6, 3, 3, 5, 0, 1, 6, 6]],

        [[2, 2, 1, 2, 4, 1, 4, 2, 1, 0, 2, 1],
         [5, 4, 5, 4, 3, 6, 4, 0, 4, 0, 6, 6]],

        [[0, 3, 0, 1, 2, 4, 1, 0, 4, 4, 2, 1],
         [1, 4, 6, 2, 4, 6, 1, 2, 1, 6, 6, 6]]],


       [[[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
         [0, 1, 2, 0, 0, 2, 0, 2, 0, 1, 1, 1]],

        [[0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
         [1, 1, 2, 0, 1, 1, 0, 2, 0, 0, 1, 1]],

        [[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
         [2, 0, 1, 0, 0, 2, 2, 2, 2, 2, 1, 1]],

        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 2, 0, 0, 2, 0, 1, 2, 1, 1, 1, 1]],

        [[1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
         [1, 2, 0, 0, 0, 2, 1, 2, 1, 0, 1, 1]],

        [[0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
         [2, 1, 2, 0, 2, 0, 0, 2, 0, 2, 1, 1]]],


       [[[4, 1, 3, 2, 4, 2, 4, 0, 3, 3, 1, 1],
         [4, 0, 4, 5, 1, 0, 0, 0, 3, 2, 3, 2]],

        [[1, 2, 4, 3, 0, 2, 0, 4, 4, 4, 1, 1],
         [0, 1, 2, 6, 5, 5, 1, 5, 3, 3, 3, 2]],

        [[3, 3, 0, 4, 1, 2, 1, 3, 0, 0, 1, 1],
         [3, 2, 0, 0, 2, 3, 2, 3, 3, 4, 3, 2]],

        [[0, 3, 4, 3, 0, 4, 0, 3, 0, 4, 1, 1],
         [6, 3, 5, 1, 6, 1, 3, 1, 3, 5, 3, 2]],

        [[2, 4, 0, 4, 1, 4, 1, 2, 1, 0, 1, 1],
         [2, 4, 3, 2, 3, 6, 4, 6, 3, 6, 3, 2]],

        [[4, 0, 1, 0, 2, 4, 2, 1, 2, 1, 1, 1],
         [5, 3, 3, 0, 3, 3, 6, 3, 5, 1, 3, 2]]]], dtype=uint64)
```

## digital permutation 

```python
>>> print(qmcseqcl.gdn_digital_permutation.__doc__)
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
>>> print(qmcseqcl.gdn_get_permutations.__doc__)
Return permutations for gdn

Args: 
    rng (np.random._generator.Generator): random number generator
    r (np.uint64): replications 
    d (np.uint64): dimension 
    tmax_new (np.uint64): number of bits in each shift 
    r_b (np.uint64): replications of bases 
    bases (np.ndarray of np.uint64): bases of size r_b*d
>>> perms = qmcseqcl.gdn_get_permutations(rng,r,d,tmax_new,r_b,bases)
>>> perms
array([[[[1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0]],

        [[0, 1, 2, 0, 0, 0, 0],
         [1, 0, 2, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0],
         [2, 1, 0, 0, 0, 0, 0],
         [0, 2, 1, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0, 0],
         [1, 0, 2, 0, 0, 0, 0],
         [1, 2, 0, 0, 0, 0, 0],
         [2, 1, 0, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0, 0],
         [2, 1, 0, 0, 0, 0, 0]]],


       [[[3, 4, 0, 2, 1, 0, 0],
         [3, 0, 1, 4, 2, 0, 0],
         [1, 3, 2, 0, 4, 0, 0],
         [1, 0, 3, 4, 2, 0, 0],
         [3, 2, 1, 4, 0, 0, 0],
         [1, 0, 3, 4, 2, 0, 0],
         [1, 3, 0, 4, 2, 0, 0],
         [0, 4, 1, 2, 3, 0, 0],
         [2, 4, 3, 1, 0, 0, 0],
         [2, 1, 4, 0, 3, 0, 0],
         [2, 0, 4, 1, 3, 0, 0],
         [1, 4, 2, 3, 0, 0, 0]],

        [[5, 0, 1, 3, 2, 6, 4],
         [4, 2, 3, 1, 6, 0, 5],
         [6, 2, 0, 5, 1, 4, 3],
         [2, 0, 3, 5, 6, 4, 1],
         [0, 6, 2, 1, 4, 3, 5],
         [3, 2, 5, 4, 6, 0, 1],
         [1, 4, 3, 6, 5, 0, 2],
         [5, 6, 4, 2, 3, 0, 1],
         [3, 0, 1, 4, 2, 5, 6],
         [3, 2, 6, 1, 0, 4, 5],
         [1, 4, 3, 2, 5, 0, 6],
         [6, 0, 4, 1, 5, 3, 2]]],


       [[[1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0]],

        [[1, 2, 0, 0, 0, 0, 0],
         [1, 2, 0, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0, 0],
         [1, 0, 2, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0, 0],
         [2, 0, 1, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0],
         [0, 2, 1, 0, 0, 0, 0],
         [2, 1, 0, 0, 0, 0, 0],
         [1, 2, 0, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0],
         [0, 1, 2, 0, 0, 0, 0]]],


       [[[1, 3, 0, 2, 4, 0, 0],
         [4, 3, 2, 1, 0, 0, 0],
         [0, 4, 1, 3, 2, 0, 0],
         [1, 4, 2, 3, 0, 0, 0],
         [1, 2, 0, 4, 3, 0, 0],
         [1, 0, 2, 4, 3, 0, 0],
         [3, 2, 0, 4, 1, 0, 0],
         [2, 4, 0, 1, 3, 0, 0],
         [2, 3, 1, 4, 0, 0, 0],
         [4, 0, 2, 1, 3, 0, 0],
         [0, 1, 3, 4, 2, 0, 0],
         [2, 0, 1, 3, 4, 0, 0]],

        [[1, 0, 4, 5, 6, 2, 3],
         [4, 5, 0, 3, 1, 6, 2],
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
>>> time_perf,time_process = qmcseqcl.gdn_digital_permutation(r,n,d,r_x,r_b,tmax,tmax_new,bmax,perms,xdig,xdig_new,**kwargs)
>>> xdig_new
array([[[[1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1],
         [1, 1, 2, 1, 2, 0, 0, 0, 1, 1, 2, 2]],

        [[0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
         [0, 0, 0, 2, 2, 2, 1, 1, 0, 2, 2, 2]],

        [[1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
         [2, 0, 1, 0, 0, 1, 0, 2, 0, 0, 2, 2]],

        [[0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
         [1, 0, 2, 1, 1, 0, 2, 0, 0, 1, 2, 2]],

        [[1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
         [0, 2, 0, 2, 1, 2, 0, 1, 2, 2, 2, 2]],

        [[0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
         [2, 2, 1, 0, 2, 1, 2, 2, 2, 0, 2, 2]]],


       [[[4, 1, 0, 4, 2, 0, 2, 4, 4, 0, 2, 1],
         [4, 0, 0, 3, 6, 1, 3, 3, 0, 4, 1, 6]],

        [[1, 4, 2, 3, 0, 2, 3, 3, 0, 4, 2, 1],
         [1, 6, 5, 5, 3, 5, 6, 1, 5, 0, 1, 6]],

        [[0, 2, 3, 0, 1, 3, 4, 1, 3, 1, 2, 1],
         [6, 1, 1, 6, 2, 0, 5, 6, 1, 1, 1, 6]],

        [[3, 0, 2, 3, 0, 4, 0, 2, 1, 0, 2, 1],
         [0, 3, 4, 4, 5, 2, 0, 2, 6, 6, 1, 6]],

        [[2, 1, 3, 0, 1, 0, 2, 4, 4, 4, 2, 1],
         [2, 2, 3, 1, 1, 6, 2, 0, 4, 2, 1, 6]],

        [[4, 4, 1, 1, 3, 2, 3, 3, 0, 1, 2, 1],
         [5, 2, 6, 6, 4, 6, 6, 5, 3, 3, 1, 6]]],


       [[[1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1],
         [0, 2, 0, 1, 0, 1, 1, 0, 1, 2, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1],
         [1, 2, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0]],

        [[1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
         [2, 1, 2, 1, 0, 1, 0, 0, 2, 0, 0, 0]],

        [[0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
         [0, 0, 1, 1, 2, 2, 2, 0, 0, 2, 0, 0]],

        [[1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
         [1, 0, 1, 1, 0, 1, 2, 0, 0, 1, 0, 0]],

        [[0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
         [2, 2, 0, 1, 2, 2, 1, 0, 1, 0, 0, 0]]],


       [[[4, 2, 1, 2, 0, 1, 0, 1, 1, 2, 0, 2],
         [3, 0, 5, 4, 1, 6, 4, 4, 4, 6, 3, 2]],

        [[3, 1, 3, 3, 4, 1, 4, 0, 4, 1, 0, 2],
         [4, 3, 2, 5, 4, 0, 2, 1, 4, 5, 3, 2]],

        [[2, 0, 2, 0, 3, 1, 1, 4, 0, 3, 0, 2],
         [2, 1, 6, 6, 3, 3, 1, 2, 4, 0, 3, 2]],

        [[1, 0, 3, 3, 4, 2, 4, 4, 0, 1, 0, 2],
         [0, 6, 3, 0, 5, 1, 3, 0, 4, 4, 3, 2]],

        [[0, 4, 2, 0, 3, 2, 1, 2, 2, 3, 0, 2],
         [6, 2, 1, 2, 6, 2, 5, 3, 4, 1, 3, 2]],

        [[4, 3, 0, 1, 1, 2, 3, 3, 3, 4, 0, 2],
         [1, 6, 1, 6, 6, 3, 6, 2, 0, 2, 3, 2]]]], dtype=uint64)
```

## convert digits to doubles 

```python 
>>> print(qmcseqcl.gdn_integer_to_float.__doc__)
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
>>> time_perf,time_process = qmcseqcl.gdn_integer_to_float(r,n,d,r_b,tmax_new,bases,xdig_new,x,**kwargs)
>>> x
array([[[0.97290039, 0.53917744],
        [0.20629883, 0.03632388],
        [0.60864258, 0.70539533],
        [0.33618164, 0.4248148 ],
        [0.84985352, 0.25407524],
        [0.0793457 , 0.93686411]],

       [[0.84707793, 0.57304772],
        [0.38097453, 0.28219443],
        [0.10456744, 0.88309157],
        [0.62106168, 0.07486728],
        [0.46435834, 0.33580649],
        [0.97073423, 0.77539095]],

       [[0.88549805, 0.23648157],
        [0.02124023, 0.57254145],
        [0.70288086, 0.86567088],
        [0.32885742, 0.06130502],
        [0.76147461, 0.38501922],
        [0.14526367, 0.91271656]],

       [[0.89126728, 0.44493083],
        [0.67019736, 0.64080715],
        [0.41704736, 0.32631978],
        [0.23026955, 0.13150509],
        [0.17710726, 0.90208831],
        [0.92209603, 0.27111067]]])
```

## nested uniform scramble 

```python 
>>> r = np.uint64(1)
>>> n_start = np.uint64(0) 
>>> n = np.uint64(9)
>>> bases = np.array([[2,3,5]],dtype=np.uint64)
>>> r_b = np.uint64(bases.shape[0])
>>> d = np.uint64(bases.shape[1])
>>> mmax = np.uint64(3)
>>> tmax = mmax
>>> C = np.tile(np.eye(mmax,dtype=np.uint64)[None,None,:,:],(r,d,1,1))
>>> xdig = np.empty((r,n,d,tmax),dtype=np.uint64)
>>> time_perf,time_process = qmcseqcl.gdn_gen_natural(r,n,d,r_b,mmax,tmax,n_start,bases,C,xdig)
>>> r_x = r 
>>> r = np.uint64(2) 
>>> tmax_new = np.uint64(8)
>>> base_seed_seq = np.random.SeedSequence(7)
>>> seeds = base_seed_seq.spawn(r*d)
>>> rngs = np.array([np.random.Generator(np.random.SFC64(seed)) for seed in seeds]).reshape(r,d)
>>> root_nodes = np.array([qmcseqcl.NUSNode_gdn() for i in range(r*d)]).reshape(r,d)
>>> xrdig = np.zeros((r,n,d,tmax_new),dtype=np.uint64)
>>> print(qmcseqcl.gdn_nested_uniform_scramble.__doc__)
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
>>> time_perf,time_process = qmcseqcl.gdn_nested_uniform_scramble(r,n,d,r_x,r_b,tmax,tmax_new,rngs,root_nodes,bases,xdig,xrdig)
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
>>> time_perf,time_process = qmcseqcl.gdn_integer_to_float(r,n,d,r_b,tmax_new,bases,xrdig,xr)
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

## digital interlacing 

```python
>>> print(qmcseqcl.gdn_interlace.__doc__)
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
>>> time_perf,time_process = qmcseqcl.gdn_interlace(r,d_alpha,mmax,d,tmax,tmax_alpha,alpha,C,C_alpha)
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

## undo digital interlacing 

```python 
>>> print(qmcseqcl.gdn_undo_interlace.__doc__)
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
>>> time_perf,time_process = qmcseqcl.gdn_undo_interlace(r,d,mmax,d_alpha,tmax,tmax_alpha,alpha,C_alpha,C_cp) 
>>> print((C==C_cp).all())
True
```

## generate natural order with the same base 

```python
>>> print(qmcseqcl.gdn_gen_natural_same_base.__doc__)
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
>>> time_perf,time_process = qmcseqcl.gdn_gen_natural_same_base(r,n,d,mmax,tmax,n_start,b,C,xdig)
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

# Issues 

- When using natural order for lattices or digital nets in base 2, it is required that `n_start` and `n_start+n` are either 0 or powers of 2