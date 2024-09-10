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

# Documentation

[Examples](./examples.md) and [doctests](./doctests.md) are available 