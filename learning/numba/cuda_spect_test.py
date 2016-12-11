from numba import cuda

def main():
    print(cuda.current_context(0))
    memory = cuda.current_context().get_memory_info()
    print(memory[0]/1024/1024)
    print(memory[1]/1024/1024)

if __name__ == '__main__':
    main()
