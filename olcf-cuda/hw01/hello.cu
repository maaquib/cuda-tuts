#include <stdio.h>

__global__ void hello() {
  printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}

int main() {
  dim3 blocks(2);
  dim3 threadsPerBlock(2);
  hello<<<blocks, threadsPerBlock>>>();
  cudaDeviceSynchronize();
}
