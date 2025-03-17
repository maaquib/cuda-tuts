#ifndef CUDA_ADDER_H
#define CUDA_ADDER_H

#include <ATen/Operators.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/library.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void add_kernel(
    const scalar_t* __restrict__ first,
    const scalar_t* __restrict__ second,
    scalar_t* result,
    int numel
);

template <typename scalar_t>
__global__ void mul_kernel(
    const scalar_t* __restrict__ first,
    const scalar_t* __restrict__ second,
    scalar_t* result,
    int N, int M, int K
);

torch::Tensor add(const torch::Tensor& first, const torch::Tensor& second);
torch::Tensor mul(const torch::Tensor& first, const torch::Tensor& second);

#endif // CUDA_ADDER_H