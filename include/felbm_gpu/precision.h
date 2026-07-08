#ifndef FELBM_GPU_PRECISION_H
#define FELBM_GPU_PRECISION_H

//=============================================================================
//  Compile-time precision switch for the GPU engine.
//
//    default            -> real_t = double  (matches the CPU reference physics)
//    -DFELBM_SINGLE      -> real_t = float   (~2x memory / bandwidth, bigger
//                                             domains, but changes numerics and
//                                             can destabilise free-energy LBM)
//
//  Only the DEVICE fields/distributions use real_t. Host-side geometry, config
//  parsing and the initial condition are always built in double via the reused
//  felbm_local host code, then cast to real_t on upload.
//=============================================================================

namespace felbm_gpu
{
#ifdef FELBM_SINGLE
  typedef float  real_t;
  #define FELBM_REAL_IS_DOUBLE 0
#else
  typedef double real_t;
  #define FELBM_REAL_IS_DOUBLE 1
#endif
}

#endif // FELBM_GPU_PRECISION_H
