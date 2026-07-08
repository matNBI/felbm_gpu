#ifndef FELBM_GPU_DEVICE_CSR_CUH
#define FELBM_GPU_DEVICE_CSR_CUH

#include "precision.h"
#include "gpu_common.cuh"

#include <vector>

//=============================================================================
//  Device-side CSR sparse operators + SpMV.
//
//  The CPU code precomputes every stencil (streaming, gradients, Laplacian) as a
//  CSR matrix and applies it as a matrix-vector product each step. We upload the
//  identical CSR arrays and run the same SpMV on the device, so the operators are
//  bit-for-bit the CPU ones (including all halfway-bounce-back near-wall terms).
//  This is the "naive but provably correct" port; a matrix-free rewrite is a
//  later optimisation (see docs/PORTING_SPEC.md, section "Optimisation roadmap").
//=============================================================================

namespace felbm_gpu
{
  // ---- single-value CSR (Laplacian, per-direction operators, streaming) -------
  struct DeviceCSR
  {
    int    n_rows = 0;
    int  * rows   = nullptr;   // size n_rows+1
    int  * cols   = nullptr;   // size nnz
    real_t * vals = nullptr;   // size nnz

    template<class IdxVec, class ValVec>
    void upload( IdxVec const & h_rows, IdxVec const & h_cols, ValVec const & h_vals )
    {
      n_rows = (int)h_rows.size() - 1;
      std::vector<int>    ri( h_rows.begin(), h_rows.end() );
      std::vector<int>    ci( h_cols.begin(), h_cols.end() );
      std::vector<real_t> vv( h_vals.begin(), h_vals.end() );
      rows = device_alloc<int>( ri.size() );    copy_h2d( rows, ri.data(), ri.size() );
      cols = device_alloc<int>( ci.size() );    copy_h2d( cols, ci.data(), ci.size() );
      vals = device_alloc<real_t>( vv.size() ); copy_h2d( vals, vv.data(), vv.size() );
    }
    void free() { device_free(rows); device_free(cols); device_free(vals); }
  };

  // ---- triple-value CSR (vector gradient: shared sparsity, three value sets) ---
  struct DeviceCSR3
  {
    int    n_rows = 0;
    int  * rows   = nullptr;
    int  * cols   = nullptr;
    real_t * vx = nullptr, * vy = nullptr, * vz = nullptr;

    template<class IdxVec, class ValVec>
    void upload( IdxVec const & h_rows, IdxVec const & h_cols,
                 ValVec const & h_vx, ValVec const & h_vy, ValVec const & h_vz )
    {
      n_rows = (int)h_rows.size() - 1;
      std::vector<int>    ri( h_rows.begin(), h_rows.end() );
      std::vector<int>    ci( h_cols.begin(), h_cols.end() );
      std::vector<real_t> x( h_vx.begin(), h_vx.end() ), y( h_vy.begin(), h_vy.end() ), z( h_vz.begin(), h_vz.end() );
      rows = device_alloc<int>( ri.size() );  copy_h2d( rows, ri.data(), ri.size() );
      cols = device_alloc<int>( ci.size() );  copy_h2d( cols, ci.data(), ci.size() );
      vx = device_alloc<real_t>( x.size() );  copy_h2d( vx, x.data(), x.size() );
      vy = device_alloc<real_t>( y.size() );  copy_h2d( vy, y.data(), y.size() );
      vz = device_alloc<real_t>( z.size() );  copy_h2d( vz, z.data(), z.size() );
    }
    void free() { device_free(rows); device_free(cols); device_free(vx); device_free(vy); device_free(vz); }
  };

#ifdef __CUDACC__
  // y = A x   (one row per thread)
  __global__ void spmv_kernel( int n_rows, int const* rows, int const* cols,
                               real_t const* vals, real_t const* x, real_t* y )
  {
    int r = blockIdx.x*blockDim.x + threadIdx.x;
    if( r >= n_rows ) return;
    real_t acc = 0;
    int a = rows[r], b = rows[r+1];
    for( int k=a; k<b; ++k ) acc += vals[k]*x[ cols[k] ];
    y[r] = acc;
  }

  // (gx,gy,gz) = A_{x,y,z} x   (shared sparsity)
  __global__ void spmv3_kernel( int n_rows, int const* rows, int const* cols,
                                real_t const* vx, real_t const* vy, real_t const* vz,
                                real_t const* x, real_t* gx, real_t* gy, real_t* gz )
  {
    int r = blockIdx.x*blockDim.x + threadIdx.x;
    if( r >= n_rows ) return;
    real_t ax=0, ay=0, az=0;
    int a = rows[r], b = rows[r+1];
    for( int k=a; k<b; ++k ){ real_t xv = x[ cols[k] ]; ax += vx[k]*xv; ay += vy[k]*xv; az += vz[k]*xv; }
    gx[r]=ax; gy[r]=ay; gz[r]=az;
  }

  inline void spmv( DeviceCSR const& A, real_t const* x, real_t* y )
  {
    spmv_kernel<<< grid_1d(A.n_rows,BLOCK), BLOCK >>>( A.n_rows, A.rows, A.cols, A.vals, x, y );
    GPU_CHECK_KERNEL();
  }
  inline void spmv3( DeviceCSR3 const& A, real_t const* x, real_t* gx, real_t* gy, real_t* gz )
  {
    spmv3_kernel<<< grid_1d(A.n_rows,BLOCK), BLOCK >>>( A.n_rows, A.rows, A.cols, A.vx, A.vy, A.vz, x, gx, gy, gz );
    GPU_CHECK_KERNEL();
  }
#endif

} // namespace felbm_gpu

#endif // FELBM_GPU_DEVICE_CSR_CUH
