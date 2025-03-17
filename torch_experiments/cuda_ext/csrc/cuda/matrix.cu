#include "matrix.cuh"
#define BLOCK_SIZE 32

namespace cuda_ext {

template <typename scalar_t>
__global__ void add_kernel(
    const scalar_t* __restrict__ first,
    const scalar_t* __restrict__ second,
    scalar_t* result,
    int numel)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < numel; i += blockDim.x * gridDim.x) {
        result[i] = first[i] + second[i];
    }
}


template <typename scalar_t>
__global__ void mul_kernel(
    const scalar_t* __restrict__ first,
    const scalar_t* __restrict__ second,
    scalar_t* __restrict__ result,
    int M, int K, int N)
{
    // Shared memory
    __shared__ scalar_t firstTile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t secondTile[BLOCK_SIZE][BLOCK_SIZE];

    auto result_row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    auto result_col = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    scalar_t running_sum = scalar_t(0);
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1)/BLOCK_SIZE; ++tile) {
        if (result_row < M && tile * BLOCK_SIZE + threadIdx.x < K) {
            firstTile[threadIdx.y][threadIdx.x] = first[result_row * K + tile * BLOCK_SIZE + threadIdx.x];
        } else {
            firstTile[threadIdx.y][threadIdx.x] = scalar_t(0);
        }

        if (result_col < N && tile * BLOCK_SIZE + threadIdx.y < K) {
            secondTile[threadIdx.y][threadIdx.x] = second[(tile * BLOCK_SIZE + threadIdx.y) * N + result_col];
        } else {
            secondTile[threadIdx.y][threadIdx.x] = scalar_t(0);
        }
        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            // Won't work for __half types
            running_sum += firstTile[threadIdx.y][i] * secondTile[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (result_row < M && result_col < N) {
        result[result_row * N + result_col] = running_sum;
    }
}

template <>
__global__ void mul_kernel<__half>(
    const __half* __restrict__ first,
    const __half* __restrict__ second,
    __half* __restrict__ result,
    int M, int K, int N)
{
    // Shared memory
    __shared__ __half firstTile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __half secondTile[BLOCK_SIZE][BLOCK_SIZE];

    auto result_row = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    auto result_col = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    __half running_sum =  __float2half(0.0f);
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1)/BLOCK_SIZE; ++tile) {
        if (result_row < M && tile * BLOCK_SIZE + threadIdx.x < K) {
            firstTile[threadIdx.y][threadIdx.x] = first[result_row * K + tile * BLOCK_SIZE + threadIdx.x];
        } else {
            firstTile[threadIdx.y][threadIdx.x] =  __float2half(0.0f);
        }

        if (result_col < N && tile * BLOCK_SIZE + threadIdx.y < K) {
            secondTile[threadIdx.y][threadIdx.x] = second[(tile * BLOCK_SIZE + threadIdx.y) * N + result_col];
        } else {
            secondTile[threadIdx.y][threadIdx.x] =  __float2half(0.0f);
        }
        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            running_sum = __hfma(firstTile[threadIdx.y][i], secondTile[i][threadIdx.x], running_sum);
        }
        __syncthreads();
    }
    if (result_row < M && result_col < N) {
        result[result_row * N + result_col] = running_sum;
    }
}

} // namepace cuda_ext

torch::Tensor add(const torch::Tensor& first, const torch::Tensor& second) {
    TORCH_CHECK(first.sizes() == second.sizes(), "Tensor size mismatch: ", first.sizes(), " vs ", second.sizes());
    TORCH_CHECK(first.dtype() == second.dtype(), "Tensor dtype mismatch: ", first.dtype(), " vs ", second.dtype());
    TORCH_CHECK(first.device() == second.device(), "Tensor device mismatch: ", first.device(), " vs ", second.device());

    const int numel = first.numel();
    const int threadsPerBlock = 256;
    const int numBlocks = (numel + threadsPerBlock - 1) / threadsPerBlock;
    auto result = torch::zeros_like(first);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        first.scalar_type(), "add", ([&] {
            cuda_ext::add_kernel<scalar_t><<<numBlocks, threadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
                first.data_ptr<scalar_t>(),
                second.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                first.size(0)
            );
        })
    );
    cudaDeviceSynchronize();
    return result;
}

torch::Tensor mul(const torch::Tensor& first, const torch::Tensor& second) {
    TORCH_CHECK(first.size(1) == second.size(0), "Tensor inner dim mismatch: ", first.size(1), " vs ", second.size(0));
    TORCH_CHECK(first.dtype() == second.dtype(), "Tensor dtype mismatch: ", first.dtype(), " vs ", second.dtype());
    TORCH_CHECK(first.device() == second.device(), "Tensor device mismatch: ", first.device(), " vs ", second.device());

    const int M = first.size(0);
    const int K = first.size(1);
    const int N = second.size(1);
    const dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 numBlocks(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    auto result = torch::zeros({M, N}, first.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        first.scalar_type(), "mul", [&] {
            cuda_ext::mul_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
                first.data_ptr<scalar_t>(),
                second.data_ptr<scalar_t>(),
                result.data_ptr<scalar_t>(),
                M, K, N
            );
        }
    );
    cudaDeviceSynchronize();
    return result;
}