import ctypes
import glob
import os

qmcgencl_clib = ctypes.CDLL(glob.glob(os.path.dirname(os.path.abspath(__file__))+"/clib*")[0], mode=ctypes.RTLD_GLOBAL)

try: import pyopencl as cl
except: print("install pyopencl to access HPC capabilities")

def add_program_to_context(cl_file:str, context):
    FILEDIR = os.path.dirname(os.path.realpath(__file__))
    with open(FILEDIR+"/qmcseqcl/"+cl_file,"r") as kernel_file:
        kernelsource = kernel_file.read()
    program = cl.Program(context,kernelsource).build()
    return program

def lattice_linear_cl(context=None, program=None, queue=None, global_sizes=None, local_sizes=None):
    if context is None: context = cl.create_some_context()
    if program is None: program = add_program_to_context("lattice.cl",context) 
    if queue is None: queue = cl.CommandQueue(context,properties=cl.command_queue_properties.PROFILING_ENABLE)




event_lattice_linear = lattice_linear(queue,(global_size_r_x,global_size_n,global_size_d),None,np.uint64(r_x),np.uint64(n),np.uint64(d),np.uint64(batch_size_r_x),np.uint64(batch_size_n),np.uint64(batch_size_d),g_d,x_d)
