import numpy as np
import time
import os 

def get_qmcseqcl_program_from_context(context):
    import pyopencl as cl
    FILEDIR = os.path.dirname(os.path.realpath(__file__))
    with open(FILEDIR+"/qmcseqcl.cl","r") as kernel_file:
        kernelsource = kernel_file.read()
    program = cl.Program(context,kernelsource).build()
    return program

def print_opencl_device_info():
    """
    Copied from https://github.com/HandsOnOpenCL/Exercises-Solutions/blob/master/Exercises/Exercise01/Python/DeviceInfo.py
    """
    import pyopencl as cl
    platforms = cl.get_platforms()
    print("\nNumber of OpenCL platforms:", len(platforms))
    print("\n-------------------------")
    for p in platforms:
        print("Platform:", p.name)
        print("Vendor:", p.vendor)
        print("Version:", p.version)
        devices = p.get_devices()
        for d in devices:
            print("\t-------------------------")
            print("\t\tName:", d.name)
            print("\t\tName:", d.name)
            print("\t\tVersion:", d.opencl_c_version)
            print("\t\tMax. Compute Units:", d.max_compute_units)
            print("\t\tLocal Memory Size:", d.local_mem_size/1024, "KB")
            print("\t\tGlobal Memory Size:", d.global_mem_size/(1024*1024), "MB")
            print("\t\tMax Alloc Size:", d.max_mem_alloc_size/(1024*1024), "MB")
            print("\t\tMax Work-group Total Size:", d.max_work_group_size)
            dim = d.max_work_item_sizes
            print("\t\tMax Work-group Dims:(", dim[0], " ".join(map(str, dim[1:])), ")")
            print("\t-------------------------")
        print("\n-------------------------")