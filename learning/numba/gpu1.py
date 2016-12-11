import numpy as np
from numba import cuda, jit
from timeit import default_timer as timer

@cuda.jit
def increment_by_one(an_array):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x

    pos = tx + ty*bw

    if pos < an_array.size:
        an_array[pos] = an_array[pos] + 1

@jit
def increment_by_onec(an_array):
    for i in range(an_array.size):
        an_array[i] = an_array[i] + 1


def main():
    ts = timer()
    dim = 10000000000
    an_array = np.ones((dim,))
    threadsperblock = 32
    blockspergrid = (an_array.size + (threadsperblock - 1) )
    start = timer()
    # increment_by_one[blockspergrid, threadsperblock](an_array)
    increment_by_onec(an_array)

    print(an_array[:5])
    print(timer() - start, timer() - ts)


if __name__ == '__main__':
    main()
