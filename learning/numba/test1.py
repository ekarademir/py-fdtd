import numpy as np
from numba import jit, cuda
from timeit import default_timer as timer

@jit(nopython=True)
# @cuda.jit([float32(float32)])
def sum2d(arr):
    M,N = arr.shape
    result = 0.0

    for i in range(M):
        for j in range(N):
            result += arr[i,j]

    return result

def main():
    asize = 3000
    test = np.ones((asize, asize), ndtype=float32)

    start = timer()
    print(sum2d(test))
    end = timer() - start

    print(end)

if __name__ == '__main__':
    main()
