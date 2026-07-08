#ifndef FELBM_GPU_D3Q19_CUH
#define FELBM_GPU_D3Q19_CUH

//=============================================================================
//  D3Q19 velocity set constants, matching felbm_local/LBM/include/lbm_velocity_set.h
//  exactly (same ordering, weights, and opposite-direction pairing), so the GPU
//  engine reproduces the CPU streaming / equilibria bit-for-bit.
//
//  coords (index : vector):
//    0:( 0, 0, 0)
//    1:( 1, 0, 0)  2:(-1, 0, 0)
//    3:( 0, 1, 0)  4:( 0,-1, 0)
//    5:( 0, 0, 1)  6:( 0, 0,-1)
//    7:( 1, 1, 0)  8:(-1,-1, 0)
//    9:( 1,-1, 0) 10:(-1, 1, 0)
//   11:( 0, 1, 1) 12:( 0,-1,-1)
//   13:( 0, 1,-1) 14:( 0,-1, 1)
//   15:( 1, 0, 1) 16:(-1, 0,-1)
//   17:( 1, 0,-1) 18:(-1, 0, 1)
//
//  weights: w0 = 1/3; w[1..6] = 1/18; w[7..18] = 1/36.
//  cs^2 = 1/3.  In the CPU code:  temperature = cs^2,
//               inv_temperature (ALPHA)        = 1/cs^2      = 3,
//               inv_two_temperature (GAMMA)    = 1/(2 cs^2)  = 1.5,
//               inv_two_temperature_2 (BETA)   = 1/(2 cs^4)  = 4.5.
//=============================================================================

namespace felbm_gpu
{
  static int const Q = 19;

  // Host-side mirrors (used to fill __constant__ memory and on the host path).
  static const int H_CX[19] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1 };
  static const int H_CY[19] = { 0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 1,-1, 1,-1, 0, 0, 0, 0 };
  static const int H_CZ[19] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1 };
  static const int H_OPP[19]= { 0, 2, 1, 4, 3, 6, 5, 8, 7,10, 9,12,11,14,13,16,15,18,17 };

  // weights as exact rationals evaluated in double
  static inline double h_weight( int k )
  {
    if( k == 0 )        return 1.0/3.0;
    else if( k < 7 )    return 1.0/18.0;
    else                return 1.0/36.0;
  }

  // Thermodynamic constants of the equilibrium (see header note).
  static double const CS2   = 1.0/3.0;   // temperature
  static double const ALPHA = 3.0;       // 1/cs^2
  static double const GAMMA = 1.5;       // 1/(2 cs^2)
  static double const BETA  = 4.5;       // 1/(2 cs^4)

#ifdef __CUDACC__
  // Device constant-memory copies, initialised once by upload_d3q19_constants().
  __constant__ int    d_cx[19];
  __constant__ int    d_cy[19];
  __constant__ int    d_cz[19];
  __constant__ int    d_opp[19];
  __constant__ double d_w[19];

  inline void upload_d3q19_constants()
  {
    double w[19];
    for( int k=0;k<19;++k) w[k] = h_weight(k);
    cudaMemcpyToSymbol( d_cx,  H_CX,  sizeof(H_CX)  );
    cudaMemcpyToSymbol( d_cy,  H_CY,  sizeof(H_CY)  );
    cudaMemcpyToSymbol( d_cz,  H_CZ,  sizeof(H_CZ)  );
    cudaMemcpyToSymbol( d_opp, H_OPP, sizeof(H_OPP) );
    cudaMemcpyToSymbol( d_w,   w,     sizeof(w)     );
  }
#endif

} // namespace felbm_gpu

#endif // FELBM_GPU_D3Q19_CUH
