import setuptools
from setuptools import Extension
import glob 
import os 

cl_files = glob.glob(os.path.dirname(os.path.abspath(__file__))+"/qmcgencl/*.cl")
for cl_file in cl_files: 
    with open(cl_file,"r") as f:
        cl_content = f.read()
    c_content = '#include "qmcgencl.h"\n\n'+cl_content 
    c_content = c_content.replace("__kernel void","void")
    c_content = c_content.replace("__global ","")
    c_content = c_content.replace("ulong","unsigned long long")
    c_content = c_content.replace("get_global_id(0)","0")
    c_content = c_content.replace("get_global_id(1)","0")
    c_content = c_content.replace("get_global_id(2)","0")
    with open(cl_file[:-1],"w") as f:
        f.write(c_content)

setuptools.setup(
    name="qmcgencl",
    version="1.0",
    install_requires=[
        'numpy >= 1.17.0',
        'scipy >= 1.0.0'
    ],
    python_requires=">=3.5",
    include_package_data=True,
    packages=[
        'qmcgencl',
    ],
    ext_modules=[
        Extension(
            name='qmcgencl.clib',
            sources=glob.glob(os.path.dirname(os.path.abspath(__file__))+"/qmcgencl/*.c")
        )
    ],
)
