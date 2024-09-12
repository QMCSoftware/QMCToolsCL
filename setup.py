import setuptools
from setuptools import Extension

cl_file = "./qmcpytoolscl/qmcpytoolscl.cl"
with open(cl_file,"r") as f:
    cl_content = f.read()
c_content = '#include "qmcpytoolscl.h"\n\n'+cl_content 
c_content = c_content.replace("__kernel void","EXPORT void")
c_content = c_content.replace("__global ","")
c_content = c_content.replace("ulong","unsigned long long")
c_content = c_content.replace("get_global_id(0)","0")
c_content = c_content.replace("get_global_id(1)","0")
c_content = c_content.replace("get_global_id(2)","0")
c_content = c_content.replace("barrier(CLK_LOCAL_MEM_FENCE);","")
c_content = c_content.replace("barrier(CLK_GLOBAL_MEM_FENCE);","")
c_content = c_content.replace("barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);","")
with open(cl_file[:-1],"w") as f:
    f.write(c_content)

setuptools.setup(
    name = "qmcpytoolscl",
    version = "1.0",
    install_requires = [
        'numpy >= 1.17.0',
    ],
    python_requires = ">=3.5",
    include_package_data=True,
    packages = [
        'qmcpytoolscl',
    ],
    ext_modules = [
        Extension(
            name = 'qmcpytoolscl.c_lib',
            sources = ["./qmcpytoolscl/qmcpytoolscl.c"]
        )
    ],
)
