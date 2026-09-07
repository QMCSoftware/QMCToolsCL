#include "qmctoolscl.h"

/* On Windows, setuptools builds a .pyd and the linker requires a
   PyInit_<name> symbol to be exported.  On POSIX the shared object is
   loaded with ctypes.CDLL, never imported as a module, so no CPython
   symbols -- and therefore no Python.h / python3-dev -- are needed. */
#ifdef _WIN32
#include <Python.h>
PyMODINIT_FUNC PyInit_c_lib(void) { return NULL; }
#endif