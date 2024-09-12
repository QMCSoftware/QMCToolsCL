#include <stdlib.h>
#include <math.h>

#if defined(_MSC_VER)
//  Microsoft
#define EXPORT __declspec(dllexport)
#define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
//  GCC
#define EXPORT __attribute__((visibility("default")))
#define IMPORT
#else
//  do nothing and hope for the best?
#define EXPORT
#define IMPORT
#pragma warning Unknown dynamic link import / export semantics.
#endif

// in Windows, you must define an initialization function for your extension
// because setuptools will build a .pyd file, not a DLL
// https://stackoverflow.com/questions/34689210/error-exporting-symbol-when-building-python-c-extension-in-windows

#include <stdio.h>
#include <Python.h>

PyMODINIT_FUNC PyInit_c_lib(void)
{
    // do stuff...
    printf("");
}