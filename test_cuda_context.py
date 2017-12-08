import pycuda.driver as cuda
from pycuda.tools import make_default_context

def cuda_function():
    cuda.init()
    ctx = make_default_context()
    ctx.detach()
    return True

for i in range(100):
    print(i)
    cuda_function()
