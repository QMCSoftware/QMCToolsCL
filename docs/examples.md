<h1>Common use Cases for QMCseqCL</h1>

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