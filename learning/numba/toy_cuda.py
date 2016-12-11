from numba import *
from numba import cuda
import numpy as np
from timeit import default_timer as timer

@cuda.jit(argtypes=[f4[:], f4[:], f4[:]])
def sum_array(a,b,c):
    i = cuda.grid(1)
    c[i] = a[i]+b[i]

def main():
    # size = 3000

    griddim = 100000, 1
    blockdim = 32, 1, 1
    size = griddim[0]*blockdim[0]

    s1 = 1000
    s2 = size // s1

    print(s1,s2)

    print("size", size/1000000, "M")

    # print(cuda.grid(1))

    cuda_sum_array = sum_array.configure(griddim, blockdim)

    A = np.ones((s1, s2), dtype=np.float32).flatten()
    B = np.ones((s1, s2), dtype=np.float32).flatten()
    C = np.zeros((s1, s2), dtype=np.float32).flatten()

    cuda_sum_array(A,B,C)

    print(C[:5])
    print(C[-5:])

if __name__ == '__main__':
    s = timer()
    main()
    print(timer()-s)
