from numba import *
from numba import cuda, vectorize
import numpy as np
from timeit import default_timer as timer

@cuda.jit(argtypes=[f4, f4, f4], device=True, inline=True)
def device_sum(a,b,d):
    return d*a+b

# @cuda.jit(argtypes=[f4[:,:,:], f4[:,:,:], f4[:,:,:], f4])
@vectorize(["float32(float32, float32, float32)"], target='cuda')
def sum_array(a,b,d):
    return device_sum(a,b,d)

def main():
    # size = 3000
    s1 = 1000
    s2 = 200
    s3 = 100

    griddim = s1, s2, s3
    blockdim = 32, 32, 1
    size = s1*s2*s3


    print(s1,s2)

    print("size", size/1000000, "M")

    # print(cuda.grid(1))

    # cuda_sum_array = sum_array.configure(griddim, blockdim)

    A = np.ones((s1, s2, s3), dtype=np.float32)
    B = np.ones((s1, s2, s3), dtype=np.float32)
    C = np.zeros((s1, s2, s3), dtype=np.float32)

    # cuda_sum_array(A,B,C, 4.0)
    C = sum_array(A,B, 4.0)

    # print(C[:5,:5])
    # print(C[-5:,-5:])
    print(C[:,:,-1])

if __name__ == '__main__':
    s = timer()
    main()
    print(timer()-s)
