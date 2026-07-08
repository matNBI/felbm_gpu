#ifndef FELBM_GPU_COMMON_CUH
#define FELBM_GPU_COMMON_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

//=============================================================================
//  Small CUDA helpers: error checking, launch config, typed device alloc.
//=============================================================================

namespace felbm_gpu
{
  inline void gpu_check( cudaError_t err, char const * file, int line )
  {
    if( err != cudaSuccess )
    {
      std::fprintf( stderr, "[felbm_gpu] CUDA error %s at %s:%d -> %s\n",
                    cudaGetErrorName(err), file, line, cudaGetErrorString(err) );
      std::exit( EXIT_FAILURE );
    }
  }

  #define GPU_CHECK(call) ::felbm_gpu::gpu_check( (call), __FILE__, __LINE__ )
  #define GPU_CHECK_KERNEL() ::felbm_gpu::gpu_check( cudaGetLastError(), __FILE__, __LINE__ )

  /// Standard 1-D grid over n elements at a given block size.
  inline dim3 grid_1d( unsigned int n, unsigned int block )
  {
    return dim3( (n + block - 1u) / block );
  }

  static unsigned int const BLOCK = 256u;

  template<typename T>
  T * device_alloc( size_t n )
  {
    T * p = nullptr;
    GPU_CHECK( cudaMalloc( (void**)&p, n*sizeof(T) ) );
    return p;
  }

  template<typename T>
  void device_free( T * p )
  {
    if( p ) cudaFree( p );
  }

  template<typename T>
  void copy_h2d( T * d, T const * h, size_t n )
  {
    GPU_CHECK( cudaMemcpy( d, h, n*sizeof(T), cudaMemcpyHostToDevice ) );
  }

  template<typename T>
  void copy_d2h( T * h, T const * d, size_t n )
  {
    GPU_CHECK( cudaMemcpy( h, d, n*sizeof(T), cudaMemcpyDeviceToHost ) );
  }

} // namespace felbm_gpu

#endif // FELBM_GPU_COMMON_CUH
