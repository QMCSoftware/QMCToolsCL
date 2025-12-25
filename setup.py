import setuptools
from setuptools import Extension
import os 
import re 

cl_files = [file[:-3] for file in os.listdir("./qmctoolscl/cl_kernels/") if file[-3:]==".cl"]

# for cl_file in cl_files:
#     with open("./qmctoolscl/cl_kernels/%s.cl"%cl_file,"r") as f:
#         cl_content = f.read()
#     c_content = '#include "qmctoolscl.h"\n\n'+cl_content 
#     c_content = c_content.replace("__kernel void","EXPORT void")
#     c_content = c_content.replace("__global ","")
#     c_content = c_content.replace("ulong","unsigned long long")
#     c_content = c_content.replace("get_global_id(0)","0")
#     c_content = c_content.replace("get_global_id(1)","0")
#     c_content = c_content.replace("get_global_id(2)","0")
#     c_content = c_content.replace("barrier(CLK_LOCAL_MEM_FENCE);","")
#     c_content = c_content.replace("barrier(CLK_GLOBAL_MEM_FENCE);","")
#     c_content = c_content.replace("barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);","")
#     with open("./qmctoolscl/c_funcs/%s.c"%cl_file,"w") as f:
#         f.write(c_content)

setuptools.setup(
    name = "qmctoolscl",
    version = "1.1.6",
    author="Aleksei G Sorokin",
    author_email="asorokin@hawk.iit.edu",
    install_requires = [
        'numpy >= 1.17.0',
    ],
    python_requires = ">=3.5",
    description = "Quasi-Monte Carlo Tools in PyOpenCL and C",
    long_description="Python interface to QMC tools with C and OpenCL backends. See https://qmcsoftware.github.io/QMCToolsCL/",
    long_description_content_type="text/markdown",
    url="https://qmcsoftware.github.io/QMCToolsCL/",
    include_package_data=True,
    packages = [
        'qmctoolscl',
    ],
    ext_modules = [
        Extension(
            name = 'qmctoolscl.c_lib',
            sources = ["./qmctoolscl/c_funcs/python_compat.c"]+["./qmctoolscl/c_funcs/%s.c"%cl_file for cl_file in cl_files]
        )
    ],
)

c_to_ctypes_map = {
    "ulong": "uint64",
    "double": "double",
    "char": "uint8",
}

THISDIR = os.path.dirname(os.path.realpath(__file__))

str_c = "import ctypes\nimport numpy as np\nfrom .util import c_lib\n\n"
str_wf = "from .util import _opencl_c_func\nfrom .c_funcs import *\n\n"
str_init = "from .rand_funcs import *\nfrom .wrapped_funcs import ("
for cl_file in cl_files:
    with open("%s/qmctoolscl/cl_kernels/%s.cl"%(THISDIR,cl_file),"r") as f:
        code = f.read() 
    blocks = re.findall(r'(?<=void\s).*?(?=\s?\))',code,re.DOTALL)
    for block in blocks:
        lines = block.replace("(","").splitlines()
        name = lines[0]
        desc = [] 
        si = 1
        while lines[si].strip()[:2]=="//":
            desc += [lines[si].split("// ")[1].strip()]
            si += 1
        desc = "\n".join(desc)
        args = []
        doc_args = []
        for i in range(si,len(lines)):
            var_input,var_desc = lines[i].split(" // ")
            var_type,var = var_input.replace(",","").split(" ")[-2:]
            if var_type not in c_to_ctypes_map:
                    raise Exception("var_type %s not found in map"%var_type)
            c_var_type = c_to_ctypes_map[var_type]
            if var[0]!="*":
                doc_args += ["%s (np.%s): %s"%(var,c_var_type,var_desc)]
                args += ["ctypes.c_%s"%c_var_type]
            else:
                doc_args += ["%s (np.ndarray of np.%s): %s"%(var[1:],c_var_type,var_desc)]
                args += ["np.ctypeslib.ndpointer(ctypes.c_%s,flags='C_CONTIGUOUS')"%c_var_type]
        doc_args = doc_args[:3]+doc_args[6:] # skip batch size args
        str_c += "%s_c = c_lib.%s\n"%(name,name)
        str_c += "%s_c.argtypes = [\n\t%s\n]\n\n"%(name,',\n\t'.join(args))
        str_wf += '@_opencl_c_func\ndef %s():\n    """%s\n\nArgs:\n    %s"""\n    pass\n\n'%(name,desc.strip(),"\n    ".join(doc_args))
        str_init += "\n\t%s,"%name

with open("%s/qmctoolscl/c_funcs.py"%THISDIR,"w") as f: f.write(str_c)
with open("%s/qmctoolscl/wrapped_funcs.py"%THISDIR,"w") as f: f.write(str_wf)
with open("%s/qmctoolscl/__init__.py"%THISDIR,"w") as f: f.write(str_init+"\n)")

# str_tex = str_tex.replace("np.double","floats")
# str_tex = str_tex.replace("np.ndarray","array")
# str_tex = str_tex.replace("of np.uint64","of ints")
# str_tex = str_tex.replace("np.uint64","ints")
# with open("api.tex","w") as f: f.write(str_tex+"\\end{itemize}")


