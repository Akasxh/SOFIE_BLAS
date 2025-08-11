# SOFIE_BLAS (Standalone BLAS for SOFIE.)

This repository contains GPU-accelerated implementations of fundamental mathematical and neural network operations using [ALPAKA](https://github.com/alpaka-group/alpaka).

The work is organized into **phases**, with operators grouped by type for development.

### Activations (GPU Kernels)

* [x] ReLU
* [x] Leaky ReLU
* [ ] PReLU
* [x] Tanh
* [x] ELU
* [x] GELU
* [ ] Swish
* [x] Sigmoid
* [ ] SELU

### Binary Operations (element-wise unless noted)

* [ ] Add (`A + B`)
* [ ] Subtract (`A - B`)
* [ ] Multiply (`A * B`)
* [ ] Divide (`A / B`)
* [ ] Max (`max(A, B)`)
* [ ] Min (`min(A, B)`)
* [ ] Power (`A ** B`)
* [ ] Dot Product (vector inner product)

### Unary Operations (element-wise)

* [ ] Negate (`-A`)
* [ ] Absolute (`|A|`)
* [ ] Square (`A^2`)
* [ ] Sqrt (`sqrt(A)`)
* [ ] Exp (`exp(A)`)
* [ ] Log (`log(A)`)
* [ ] Clip (min/max)
* [ ] Sign (−1/0/1)

### Normalization Layers

* [ ] Batch Normalization (train + inference)
* [ ] Layer Normalization

### Functions with BLAS Abstraction

* [ ] GEMM (matrix–matrix multiply)
* [ ] Softmax (axis-wise, stable implementation)

### Matrix Operations

* [ ] Transpose (2D)
* [ ] Batched Transpose (3D/ND)




