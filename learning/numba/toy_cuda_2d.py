from numba import *
from numba import cuda
import numpy as np
from timeit import default_timer as timer

@cuda.jit(argtypes=[f4[:,:,:], f4[:,:,:], f4[:,:,:], f4])
def sum_array(a,b,c, d):
    # tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y
    # bx = cuda.blockIdx.x
    # by = cuda.blockIdx.y
    # bw = cuda.blockDim.x
    # bh = cuda.blockDim.y
    # i = tx + bx * bw
    # j = ty + by * bh
    i,j,k = cuda.grid(3)
    if i<c.shape[0] and j<c.shape[1] and k<c.shape[2]:
        c[i,j,k] = d*a[i,j,k]+b[i,j,k]

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

    cuda_sum_array = sum_array.configure(griddim, blockdim)

    A = np.ones((s1, s2, s3), dtype=np.float32)
    B = np.ones((s1, s2, s3), dtype=np.float32)
    C = np.zeros((s1, s2, s3), dtype=np.float32)

    cuda_sum_array(A,B,C, 4.0)

    # print(C[:5,:5])
    # print(C[-5:,-5:])
    print(C[:,:,-1])

if __name__ == '__main__':
    s = timer()
    main()
    print(timer()-s)
