import pycuda.driver as cuda
from pycuda.tools import make_default_context


def cuda_function():
    voxels_out_d = cuda.mem_alloc(10000)
    voxels_out_d.free()
    return True

cuda.init()
ctx = make_default_context()
for i in range(100):
    print(i)
    cuda_function()
ctx.detach()
