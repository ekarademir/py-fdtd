import numpy as np
from timeit import default_timer as timer
from numba import vectorize

@vectorize(["float32(float32,float32)"], target='cpu')
# @vectorize(["float32(float32,float32)"], target='cuda')
def VectorAdd(a,b):
    return a + b

# def VectorAdd(a,b,c):
#     for i in range(a.size):
#         c[i] = a[i] + b[i]

def main():
    N = 32000000

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)

    start = timer()
    C = VectorAdd(A,B)
    # VectorAdd(A,B,C)
    vectoradd_time = timer() - start

    print("C[:5] = " + str(C[:5]))
    print("C[-5:] = " + str(C[-5:]))

    print(vectoradd_time)


if __name__ == '__main__':
    main()
