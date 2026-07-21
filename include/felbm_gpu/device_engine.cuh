#ifndef FELBM_GPU_DEVICE_ENGINE_CUH
#define FELBM_GPU_DEVICE_ENGINE_CUH

#include "precision.h"
#include "gpu_common.cuh"
#include "d3q19.cuh"
#include "device_csr.cuh"

#include <vector>

//=============================================================================
//  MultiPhaseEngineGPU  —  CUDA port of the Lee-Liu multiphase time step.
//
//  Correctness strategy: every STENCIL (streaming, gradients, Laplacian, the
//  per-direction operators) is the CPU FieldOperator / StreamingOperator CSR,
//  uploaded verbatim and applied by device SpMV, so those match the CPU bitwise.
//  The POINTWISE physics (moments, equilibria, force term, collision combine) is
//  ported below kernel-by-kernel from the CPU sources, with the source function
//  named above each kernel.  See docs/PORTING_SPEC.md for the exact mapping.
//
//  STATUS: v0, compiles/needs validation on a GPU box.  Not yet ported:
//    - MRT collision (use_mrt): only BGK path here.
//    - open inlet/outlet boundaries (use_open_bnd): body-force periodic only.
//    - order-parameter mass correction (correct_op_mass): host reduction TODO.
//    - particle tracking (host subsystem; run on CPU or port later).
//=============================================================================

namespace felbm_gpu
{
  // POD parameter block copied to the device by value into each kernel.
  struct DevParams
  {
    int    n;           // n_sites
    real_t rho0, rho1, drho;
    real_t kappa, beta, four_beta, bnd_coeff;
    real_t tau0, tau1;
    real_t cs2, alpha, gamma_c, beta_c;   // equilibrium constants (gamma_c=1/(2cs2), beta_c=1/(2cs4))
    real_t mobility, forcing_factor;
    real_t gx, gy, gz;      // gravity (settings.acceleration)
    real_t fx, fy, fz;      // forcing  (settings.forcing)
    real_t mrt_lambda;      // MRT magic parameter (settings.mrt_lambda); unused for BGK
    int    mrt_fast = 0;    // 1 = real_t MRT moment transform (mrt_fast_transform)
    int    stream_inplace = 0; // 1 = fused collision writes reversed slots; streaming is in-place
  };

#ifdef __CUDACC__

  // D3Q19 MRT transform matrices (M and M^-1), copied from the CPU
  // EquilibriumDistributionMRT statics at init. Kept in double for accuracy.
  __constant__ double d_M[19][19];
  __constant__ double d_Minv[19][19];
  // real_t copies for the fast transform (mrt_fast_transform): same matrices,
  // same summation order in the kernel -> bit-identical to the double path when
  // real_t==double; a pure-float transform under -DFELBM_SINGLE.
  __constant__ real_t d_Mf[19][19];
  __constant__ real_t d_Minvf[19][19];

  inline void upload_mrt_matrices( double const* M, double const* Minv )
  {
    cudaMemcpyToSymbol( d_M,    M,    19*19*sizeof(double) );
    cudaMemcpyToSymbol( d_Minv, Minv, 19*19*sizeof(double) );
    real_t Mf[19*19], If[19*19];
    for( int k=0;k<19*19;++k ){ Mf[k]=(real_t)M[k]; If[k]=(real_t)Minv[k]; }
    cudaMemcpyToSymbol( d_Mf,    Mf, sizeof(Mf) );
    cudaMemcpyToSymbol( d_Minvf, If, sizeof(If) );
  }

  __device__ __forceinline__ real_t dir_x(int k){ return (real_t)d_cx[k]; }
  __device__ __forceinline__ real_t dir_y(int k){ return (real_t)d_cy[k]; }
  __device__ __forceinline__ real_t dir_z(int k){ return (real_t)d_cz[k]; }

  // ---- single directional values from the matrix-free tables (for fusion) -----
  // Direction 0 (rest) has no derivative -> 0.  m in 1..Q-1 index (m-1)*n+i.
  __device__ __forceinline__ real_t cd_dir_val( int n, int const* cminus, int const* cplus,
                                                real_t const* f, int i, int m )
  {
    if( m==0 ) return real_t(0);
    int idx=(m-1)*n+i, cm=cminus[idx];
    return (cm<0) ? real_t(0) : real_t(0.5)*(f[cplus[idx]]-f[cm]);
  }
  __device__ __forceinline__ real_t bd_dir_val( int n, int const* bcase, int const* bc1, int const* bc2,
                                                real_t const* f, real_t f0, int i, int m )
  {
    if( m==0 ) return real_t(0);
    int idx=(m-1)*n+i, cs=bcase[idx];
    if     ( cs==0 ) return real_t(0);
    real_t f1=f[bc1[idx]];
    if     ( cs==1 ){ real_t f2=f[bc2[idx]]; return real_t(-1.5)*f0+real_t(2.0)*f1+real_t(-0.5)*f2; }
    else if( cs==2 ) return real_t(-2.0)*f0+real_t(2.0)*f1;
    else if( cs==3 ) return real_t(-1.5)*f0+real_t(1.5)*f1;
    else             return real_t(0.5)*f0+real_t(-0.5)*f1;
  }
  __device__ __forceinline__ real_t avg_dir_val( int n, int const* avgnext,
                                                 real_t const* x, real_t xi, int i, int m )
  {
    int nx=avgnext[m*n+i];
    return (nx<0) ? xi : real_t(0.5)*(xi+x[nx]);
  }

  //--- CPU: FieldManagerMultiPhase::compute_fields pass 1 (lines 75-130) --------
  __global__ void k_moments( DevParams P, real_t const* h, real_t const* g,
                             real_t* c_, real_t* p_, real_t* rho_, real_t* mu_,
                             real_t* ux_, real_t* uy_, real_t* uz_, real_t* relax )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=P.n ) return;
    int const n=P.n;
    real_t c=0,p=0,ux=0,uy=0,uz=0;
    for( int k=0;k<Q;++k ){ real_t hv=h[k*n+i], gv=g[k*n+i];
      c+=hv; p+=gv; ux+=dir_x(k)*gv; uy+=dir_y(k)*gv; uz+=dir_z(k)*gv; }
    real_t c_log=c*(real_t(1)-c);
    real_t rho=P.rho1+P.drho*c;
    real_t inv=real_t(1)/(P.cs2*rho);
    ux*=inv; uy*=inv; uz*=inv;
    real_t mu=P.four_beta*c_log*(real_t(0.5)-c);
    real_t cp = c<real_t(0)?real_t(0):(c>real_t(1)?real_t(1):c);
    real_t tau=real_t(1)/(cp/P.tau0+(real_t(1)-cp)/P.tau1);
    real_t rp=real_t(1)/(tau+real_t(0.5));
    c_[i]=c; p_[i]=p; rho_[i]=rho; mu_[i]=mu; ux_[i]=ux; uy_[i]=uy; uz_[i]=uz;
    relax[i]=rp;                       // d_relax is length n (uniform over k)
  }

  //--- CPU: grad_md = 0.5*(grad_cd + grad_bd)  (compute_fields lines 155-161) ----
  __global__ void k_grad_md( int n, real_t const* cd_x, real_t const* cd_y, real_t const* cd_z,
                             real_t const* bd_x, real_t const* bd_y, real_t const* bd_z,
                             real_t* md_x, real_t* md_y, real_t* md_z )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=n ) return;
    md_x[i]=real_t(0.5)*(cd_x[i]+bd_x[i]);
    md_y[i]=real_t(0.5)*(cd_y[i]+bd_y[i]);
    md_z[i]=real_t(0.5)*(cd_z[i]+bd_z[i]);
  }

  //--- CPU: mu += -kappa * laplacian(c)  (compute_fields line 173, axpy) ---------
  __global__ void k_mu_axpy( int n, real_t kappa, real_t const* lap_c, real_t* mu )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=n ) return;
    mu[i] += -kappa*lap_c[i];
  }

  //--- Pack the length-2n input the node-centred Laplacian expects:
  //---   xtnd[0..n-1]   = c
  //---   xtnd[n..2n-1]  = bnd_cond = boundary_coefficient * c*(1-c)   (wetting BC)
  //--- The Laplacian CSR has 2n columns; near-wall rows reference the second half.
  __global__ void k_pack_lap_c( DevParams P, real_t const* c, real_t* xtnd )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=P.n ) return;
    real_t cv = c[i];
    xtnd[i]       = cv;
    xtnd[P.n + i] = P.bnd_coeff*cv*(real_t(1)-cv);
  }

  //--- Pack [f, 0] for the Laplacian of a field with no boundary term (e.g. mu).
  __global__ void k_pack_lap_zero( int n, real_t const* f, real_t* xtnd )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=n ) return;
    xtnd[i]     = f[i];
    xtnd[n + i] = real_t(0);
  }

  //--- CPU: velocity + pressure correction (compute_fields lines 186-214) --------
  //--- (is_processed==false everywhere for the body-force / periodic case.)
  __global__ void k_vel_press_corr( DevParams P, real_t const* mu_, real_t const* rho_,
                                    real_t const* gcc_x, real_t const* gcc_y, real_t const* gcc_z,
                                    real_t* ux_, real_t* uy_, real_t* uz_, real_t* p_ )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=P.n ) return;
    real_t mu=mu_[i], rho=rho_[i];
    real_t gx=gcc_x[i], gy=gcc_y[i], gz=gcc_z[i];
    real_t mor=mu/rho;
    real_t ux=ux_[i]+real_t(0.5)*(P.gx+P.fx*P.forcing_factor/rho+mor*gx);
    real_t uy=uy_[i]+real_t(0.5)*(P.gy+P.fy*P.forcing_factor/rho+mor*gy);
    real_t uz=uz_[i]+real_t(0.5)*(P.gz+P.fz*P.forcing_factor/rho+mor*gz);
    ux_[i]=ux; uy_[i]=uy; uz_[i]=uz;
    real_t u_dot_drho=P.drho*(ux*gx+uy*gy+uz*gz);
    p_[i]+=real_t(0.5)*u_dot_drho*P.cs2;
  }

  //--- CPU: CollisionModelMultiPhase::compute_all_equilibria (lines 59-112) ------
  __global__ void k_equilibria( DevParams P, unsigned char const* is_solid, unsigned char const* is_streamed,
                                real_t const* c_, real_t const* rho_, real_t const* p_, real_t const* mu_,
                                real_t const* ux_, real_t const* uy_, real_t const* uz_,
                                real_t const* gpc_x, real_t const* gpc_y, real_t const* gpc_z,
                                real_t const* gcc_x, real_t const* gcc_y, real_t const* gcc_z,
                                real_t const* gc_dir, real_t const* gp_dir,
                                real_t* eq_h, real_t* eq_g )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=P.n ) return;
    int const n=P.n;
    if( is_solid[i] || !is_streamed[i] ){ for(int k=0;k<Q;++k){ eq_h[k*n+i]=0; eq_g[k*n+i]=0; } return; }
    real_t c=c_[i], rho=rho_[i], p=p_[i], mu=mu_[i];
    real_t ux=ux_[i], uy=uy_[i], uz=uz_[i];
    real_t dpx=gpc_x[i],dpy=gpc_y[i],dpz=gpc_z[i];
    real_t dcx=gcc_x[i],dcy=gcc_y[i],dcz=gcc_z[i];
    real_t drcx=P.drho*dcx*P.cs2, drcy=P.drho*dcy*P.cs2, drcz=P.drho*dcz*P.cs2; // drho_cd
    real_t bfx=rho*P.gx+P.fx*P.forcing_factor, bfy=rho*P.gy+P.fy*P.forcing_factor, bfz=rho*P.gz+P.fz*P.forcing_factor;
    real_t fgx=mu*dcx+bfx, fgy=mu*dcy+bfy, fgz=mu*dcz+bfz;                       // force_g
    real_t a_over_rho=P.alpha/rho;
    real_t fhx=dcx-c*a_over_rho*(dpx-fgx), fhy=dcy-c*a_over_rho*(dpy-fgy), fhz=dcz-c*a_over_rho*(dpz-fgz); // force_h
    real_t u_sq=ux*ux+uy*uy+uz*uz;
    for( int k=0;k<Q;++k ){
      real_t w=(real_t)d_w[k];
      real_t uk=ux*dir_x(k)+uy*dir_y(k)+uz*dir_z(k);
      real_t u_term=uk*(P.alpha+P.beta_c*uk)-P.gamma_c*u_sq;
      real_t G=w*(real_t(1)+u_term);
      real_t dc_dk=gc_dir[k*n+i], dp_dk=gp_dir[k*n+i];
      real_t drho_dk=P.drho*dc_dk*P.cs2;
      real_t f_proj=bfx*dir_x(k)+bfy*dir_y(k)+bfz*dir_z(k);
      real_t u_fh=ux*fhx+uy*fhy+uz*fhz;
      real_t u_drhocd=ux*drcx+uy*drcy+uz*drcz;
      real_t u_fg=ux*fgx+uy*fgy+uz*fgz;
      real_t eh = c*G;
      eh -= real_t(0.5)*G*(dc_dk-c*a_over_rho*(dp_dk-mu*dc_dk-f_proj));
      eh += real_t(0.5)*G*u_fh;
      eq_h[k*n+i]=eh;
      real_t eg = w*(p+rho*P.cs2*u_term);
      eg -= real_t(0.5)*(drho_dk*(G-w)+(mu*dc_dk+f_proj)*G);
      eg += real_t(0.5)*((G-w)*u_drhocd+G*u_fg);
      eq_g[k*n+i]=eg;
    }
  }

  //--- CPU: CollisionModelMultiPhase::compute_collision_term (lines 140-153) -----
  //--- coll_h = eq_h - h;  coll_g = relax .* (eq_g - g)
  //    relax is length n (one value per site); j = k*n + i -> site i = j % n.
  __global__ void k_collision_term( int n_var, int n, real_t const* eq_h, real_t const* eq_g,
                                    real_t const* h, real_t const* g, real_t const* relax,
                                    real_t* coll_h, real_t* coll_g )
  {
    int j = blockIdx.x*blockDim.x + threadIdx.x; if( j>=n_var ) return;
    coll_h[j]=eq_h[j]-h[j];
    coll_g[j]=relax[j%n]*(eq_g[j]-g[j]);
  }

  //--- CPU: ForceTermMultiPhase::update (lines 120-178) --------------------------
  __global__ void k_force_term( DevParams P, unsigned char const* is_streamed,
                                real_t const* c_, real_t const* rho_, real_t const* mu_,
                                real_t const* ux_, real_t const* uy_, real_t const* uz_,
                                real_t const* gpm_x, real_t const* gpm_y, real_t const* gpm_z,
                                real_t const* gcm_x, real_t const* gcm_y, real_t const* gcm_z,
                                real_t const* gc_cd_dir, real_t const* gc_bd_dir,
                                real_t const* gp_cd_dir, real_t const* gp_bd_dir,
                                real_t const* avg_lap_mu,
                                real_t* force_h, real_t* force_g )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=P.n ) return;
    int const n=P.n;
    if( !is_streamed[i] ){ for(int k=0;k<Q;++k){ force_h[k*n+i]=0; force_g[k*n+i]=0; } return; }
    real_t c=c_[i], rho=rho_[i], mu=mu_[i];
    real_t ux=ux_[i], uy=uy_[i], uz=uz_[i];
    real_t dpx=gpm_x[i],dpy=gpm_y[i],dpz=gpm_z[i];
    real_t dcx=gcm_x[i],dcy=gcm_y[i],dcz=gcm_z[i];
    real_t drho_cs2=P.drho*P.cs2;
    real_t drmx=drho_cs2*dcx, drmy=drho_cs2*dcy, drmz=drho_cs2*dcz;
    real_t bfx=rho*P.gx+P.fx*P.forcing_factor, bfy=rho*P.gy+P.fy*P.forcing_factor, bfz=rho*P.gz+P.fz*P.forcing_factor;
    real_t c_over=c/P.cs2/rho;
    real_t fgx=mu*dcx+bfx, fgy=mu*dcy+bfy, fgz=mu*dcz+bfz;
    real_t fhx=dcx-c_over*(dpx-fgx), fhy=dcy-c_over*(dpy-fgy), fhz=dcz-c_over*(dpz-fgz);
    real_t u_fh=ux*fhx+uy*fhy+uz*fhz;
    real_t u_drm=ux*drmx+uy*drmy+uz*drmz;
    real_t u_fg=ux*fgx+uy*fgy+uz*fgz;
    real_t u_sq=ux*ux+uy*uy+uz*uz;
    for( int k=0;k<Q;++k ){
      real_t w=(real_t)d_w[k];
      real_t uk=ux*dir_x(k)+uy*dir_y(k)+uz*dir_z(k);
      real_t u_term=uk*(P.alpha+P.beta_c*uk)-P.gamma_c*u_sq;
      real_t Gk=w*(real_t(1)+u_term);           // EquilibriumDistributionH with rho=1
      real_t dGk=Gk-w;
      real_t dc_dk=real_t(0.5)*(gc_cd_dir[k*n+i]+gc_bd_dir[k*n+i]);
      real_t dp_dk=real_t(0.5)*(gp_cd_dir[k*n+i]+gp_bd_dir[k*n+i]);
      real_t drho_dk=drho_cs2*dc_dk;
      real_t fproj=bfx*dir_x(k)+bfy*dir_y(k)+bfz*dir_z(k);
      real_t ff=mu*dc_dk+fproj;
      force_h[k*n+i]=Gk*(dc_dk-c_over*(dp_dk-ff)-u_fh+P.mobility*avg_lap_mu[k*n+i]);
      force_g[k*n+i]=(drho_dk-u_drm)*dGk+(ff-u_fg)*Gk;
    }
  }

  //--- CPU: CollisionOperator::transform  f += collision_term + force_term -------
  __global__ void k_collide_apply( int n_var, real_t const* coll_h, real_t const* coll_g,
                                   real_t const* force_h, real_t const* force_g,
                                   real_t* h, real_t* g )
  {
    int j = blockIdx.x*blockDim.x + threadIdx.x; if( j>=n_var ) return;
    h[j]+=coll_h[j]+force_h[j];
    g[j]+=coll_g[j]+force_g[j];
  }

  // ===========================================================================
  //  FUSED variants (cfg: fused=true, requires grad_matrix_free).  Identical to
  //  k_equilibria / k_equilibria_mrt / k_force_term above, except the per-direction
  //  directional derivatives (grad_*_cd_dir / _bd_dir) and the avg(lap mu) are
  //  recomputed in registers from the matrix-free tables instead of being read
  //  from precomputed Q*n temporaries. Eliminates 5 Q*n fields (gc_cd, gp_cd,
  //  gc_bd, gp_bd, avg) — never written, never re-read.
  // ===========================================================================
  __global__ void k_equilibria_fused( DevParams P, unsigned char const* is_solid, unsigned char const* is_streamed,
                                real_t const* c_, real_t const* rho_, real_t const* p_, real_t const* mu_,
                                real_t const* ux_, real_t const* uy_, real_t const* uz_,
                                real_t const* gpc_x, real_t const* gpc_y, real_t const* gpc_z,
                                real_t const* gcc_x, real_t const* gcc_y, real_t const* gcc_z,
                                int const* cdm, int const* cdp,
                                real_t* eq_h, real_t* eq_g )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=P.n ) return;
    int const n=P.n;
    if( is_solid[i] || !is_streamed[i] ){ for(int k=0;k<Q;++k){ eq_h[k*n+i]=0; eq_g[k*n+i]=0; } return; }
    real_t c=c_[i], rho=rho_[i], p=p_[i], mu=mu_[i];
    real_t ux=ux_[i], uy=uy_[i], uz=uz_[i];
    real_t dpx=gpc_x[i],dpy=gpc_y[i],dpz=gpc_z[i];
    real_t dcx=gcc_x[i],dcy=gcc_y[i],dcz=gcc_z[i];
    real_t drcx=P.drho*dcx*P.cs2, drcy=P.drho*dcy*P.cs2, drcz=P.drho*dcz*P.cs2;
    real_t bfx=rho*P.gx+P.fx*P.forcing_factor, bfy=rho*P.gy+P.fy*P.forcing_factor, bfz=rho*P.gz+P.fz*P.forcing_factor;
    real_t fgx=mu*dcx+bfx, fgy=mu*dcy+bfy, fgz=mu*dcz+bfz;
    real_t a_over_rho=P.alpha/rho;
    real_t fhx=dcx-c*a_over_rho*(dpx-fgx), fhy=dcy-c*a_over_rho*(dpy-fgy), fhz=dcz-c*a_over_rho*(dpz-fgz);
    real_t u_sq=ux*ux+uy*uy+uz*uz;
    for( int k=0;k<Q;++k ){
      real_t w=(real_t)d_w[k];
      real_t uk=ux*dir_x(k)+uy*dir_y(k)+uz*dir_z(k);
      real_t u_term=uk*(P.alpha+P.beta_c*uk)-P.gamma_c*u_sq;
      real_t G=w*(real_t(1)+u_term);
      real_t dc_dk=cd_dir_val(n,cdm,cdp,c_,i,k), dp_dk=cd_dir_val(n,cdm,cdp,p_,i,k);
      real_t drho_dk=P.drho*dc_dk*P.cs2;
      real_t f_proj=bfx*dir_x(k)+bfy*dir_y(k)+bfz*dir_z(k);
      real_t u_fh=ux*fhx+uy*fhy+uz*fhz;
      real_t u_drhocd=ux*drcx+uy*drcy+uz*drcz;
      real_t u_fg=ux*fgx+uy*fgy+uz*fgz;
      real_t eh = c*G;
      eh -= real_t(0.5)*G*(dc_dk-c*a_over_rho*(dp_dk-mu*dc_dk-f_proj));
      eh += real_t(0.5)*G*u_fh;
      eq_h[k*n+i]=eh;
      real_t eg = w*(p+rho*P.cs2*u_term);
      eg -= real_t(0.5)*(drho_dk*(G-w)+(mu*dc_dk+f_proj)*G);
      eg += real_t(0.5)*((G-w)*u_drhocd+G*u_fg);
      eq_g[k*n+i]=eg;
    }
  }

  __global__ void k_equilibria_mrt_fused( DevParams P, unsigned char const* is_streamed,
                                    real_t const* c_, real_t const* rho_, real_t const* p_, real_t const* mu_,
                                    real_t const* ux_, real_t const* uy_, real_t const* uz_,
                                    real_t const* gpc_x, real_t const* gpc_y, real_t const* gpc_z,
                                    real_t const* gcc_x, real_t const* gcc_y, real_t const* gcc_z,
                                    int const* cdm, int const* cdp,
                                    real_t* eq_h, real_t* eq_g )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=P.n ) return;
    int const n=P.n;
    if( !is_streamed[i] ){ for(int k=0;k<Q;++k){ eq_h[k*n+i]=0; eq_g[k*n+i]=0; } return; }
    real_t c=c_[i], rho=rho_[i], p=p_[i], mu=mu_[i];
    real_t ux=ux_[i], uy=uy_[i], uz=uz_[i];
    real_t dpx=gpc_x[i],dpy=gpc_y[i],dpz=gpc_z[i];
    real_t dcx=gcc_x[i],dcy=gcc_y[i],dcz=gcc_z[i];
    real_t drho_cs2=P.drho*P.cs2;
    real_t drcx=drho_cs2*dcx, drcy=drho_cs2*dcy, drcz=drho_cs2*dcz;
    real_t bfx=rho*P.gx+P.fx*P.forcing_factor, bfy=rho*P.gy+P.fy*P.forcing_factor, bfz=rho*P.gz+P.fz*P.forcing_factor;
    real_t rho_cs2=rho*P.cs2;
    real_t c_over=c/rho_cs2;
    real_t fgx=mu*dcx+bfx, fgy=mu*dcy+bfy, fgz=mu*dcz+bfz;
    real_t fhx=dcx-c_over*(dpx-fgx), fhy=dcy-c_over*(dpy-fgy), fhz=dcz-c_over*(dpz-fgz);
    real_t gamma_u_sq=P.gamma_c*(ux*ux+uy*uy+uz*uz);
    real_t u_term_h=ux*fhx+uy*fhy+uz*fhz;
    real_t u_force_g=ux*fgx+uy*fgy+uz*fgz;
    real_t u_drho=ux*drcx+uy*drcy+uz*drcz;
    for( int k=0;k<Q;++k ){
      real_t w=(real_t)d_w[k];
      real_t uk=ux*dir_x(k)+uy*dir_y(k)+uz*dir_z(k);
      real_t w_u_term=w*(uk*(P.alpha+P.beta_c*uk)-gamma_u_sq);
      real_t G=w+w_u_term;
      real_t dc_dk=cd_dir_val(n,cdm,cdp,c_,i,k), dp_dk=cd_dir_val(n,cdm,cdp,p_,i,k);
      real_t drho_dk=drho_cs2*dc_dk;
      real_t f_proj=bfx*dir_x(k)+bfy*dir_y(k)+bfz*dir_z(k);
      eq_h[k*n+i] = c*G + real_t(0.5)*G*(u_term_h - dc_dk + c_over*(dp_dk - mu*dc_dk - f_proj));
      eq_g[k*n+i] = w*p + rho_cs2*w_u_term
                  + real_t(0.5)*((u_drho - drho_dk)*w_u_term + (u_force_g + mu*dc_dk + f_proj)*G);
    }
  }

  __global__ void k_force_term_fused( DevParams P, unsigned char const* is_streamed,
                                real_t const* c_, real_t const* rho_, real_t const* p_, real_t const* mu_,
                                real_t const* ux_, real_t const* uy_, real_t const* uz_,
                                real_t const* gpm_x, real_t const* gpm_y, real_t const* gpm_z,
                                real_t const* gcm_x, real_t const* gcm_y, real_t const* gcm_z,
                                real_t const* lapmu_,
                                int const* cdm, int const* cdp,
                                int const* bcase, int const* bc1, int const* bc2,
                                int const* avgnext,
                                real_t* force_h, real_t* force_g )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=P.n ) return;
    int const n=P.n;
    if( !is_streamed[i] ){ for(int k=0;k<Q;++k){ force_h[k*n+i]=0; force_g[k*n+i]=0; } return; }
    real_t c=c_[i], rho=rho_[i], mu=mu_[i], p=p_[i], lmu=lapmu_[i];
    real_t ux=ux_[i], uy=uy_[i], uz=uz_[i];
    real_t dpx=gpm_x[i],dpy=gpm_y[i],dpz=gpm_z[i];
    real_t dcx=gcm_x[i],dcy=gcm_y[i],dcz=gcm_z[i];
    real_t drho_cs2=P.drho*P.cs2;
    real_t drmx=drho_cs2*dcx, drmy=drho_cs2*dcy, drmz=drho_cs2*dcz;
    real_t bfx=rho*P.gx+P.fx*P.forcing_factor, bfy=rho*P.gy+P.fy*P.forcing_factor, bfz=rho*P.gz+P.fz*P.forcing_factor;
    real_t c_over=c/P.cs2/rho;
    real_t fgx=mu*dcx+bfx, fgy=mu*dcy+bfy, fgz=mu*dcz+bfz;
    real_t fhx=dcx-c_over*(dpx-fgx), fhy=dcy-c_over*(dpy-fgy), fhz=dcz-c_over*(dpz-fgz);
    real_t u_fh=ux*fhx+uy*fhy+uz*fhz;
    real_t u_drm=ux*drmx+uy*drmy+uz*drmz;
    real_t u_fg=ux*fgx+uy*fgy+uz*fgz;
    real_t u_sq=ux*ux+uy*uy+uz*uz;
    for( int k=0;k<Q;++k ){
      real_t w=(real_t)d_w[k];
      real_t uk=ux*dir_x(k)+uy*dir_y(k)+uz*dir_z(k);
      real_t u_term=uk*(P.alpha+P.beta_c*uk)-P.gamma_c*u_sq;
      real_t Gk=w*(real_t(1)+u_term);
      real_t dGk=Gk-w;
      real_t dc_dk=real_t(0.5)*(cd_dir_val(n,cdm,cdp,c_,i,k)+bd_dir_val(n,bcase,bc1,bc2,c_,c,i,k));
      real_t dp_dk=real_t(0.5)*(cd_dir_val(n,cdm,cdp,p_,i,k)+bd_dir_val(n,bcase,bc1,bc2,p_,p,i,k));
      real_t drho_dk=drho_cs2*dc_dk;
      real_t fproj=bfx*dir_x(k)+bfy*dir_y(k)+bfz*dir_z(k);
      real_t ff=mu*dc_dk+fproj;
      real_t avgk=avg_dir_val(n,avgnext,lapmu_,lmu,i,k);
      force_h[k*n+i]=Gk*(dc_dk-c_over*(dp_dk-ff)-u_fh+P.mobility*avgk);
      force_g[k*n+i]=(drho_dk-u_drm)*dGk+(ff-u_fg)*Gk;
    }
  }

  // ===========================================================================
  //  STEP 2 FUSION (cfg: fuse_collision=true).  One per-site kernel that fuses
  //  equilibria + force + collision_term + collide_apply, so eq_h/eq_g/force_h/
  //  force_g/coll_h/coll_g (6 Q*n temporaries) are NEVER materialised. The h
  //  update collapses: h_new = h + (eq_h - h) + force_h = eq_h + force_h.
  // ===========================================================================
  __global__ void k_collide_fused_bgk( DevParams P, unsigned char const* is_solid, unsigned char const* is_streamed,
        real_t const* c_, real_t const* rho_, real_t const* p_, real_t const* mu_,
        real_t const* ux_, real_t const* uy_, real_t const* uz_,
        real_t const* gpc_x, real_t const* gpc_y, real_t const* gpc_z,   // grad_p_cd (eq)
        real_t const* gcc_x, real_t const* gcc_y, real_t const* gcc_z,   // grad_c_cd (eq)
        real_t const* gpm_x, real_t const* gpm_y, real_t const* gpm_z,   // grad_p_md (force)
        real_t const* gcm_x, real_t const* gcm_y, real_t const* gcm_z,   // grad_c_md (force)
        real_t const* lapmu_, real_t const* relax_,
        int const* cdm, int const* cdp, int const* bcase, int const* bc1, int const* bc2, int const* avgnext,
        real_t* h, real_t* g )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=P.n ) return;
    int const n=P.n;
    bool ge = ( is_solid[i] || !is_streamed[i] );   // equilibria guard -> eq=0
    bool gf = ( !is_streamed[i] );                  // force guard      -> force=0
    real_t c=c_[i], rho=rho_[i], p=p_[i], mu=mu_[i];
    real_t ux=ux_[i], uy=uy_[i], uz=uz_[i], relax=relax_[i], lmu=lapmu_[i];
    // equilibria site scalars (cd gradients)
    real_t e_dpx=gpc_x[i],e_dpy=gpc_y[i],e_dpz=gpc_z[i];
    real_t e_dcx=gcc_x[i],e_dcy=gcc_y[i],e_dcz=gcc_z[i];
    real_t drcx=P.drho*e_dcx*P.cs2, drcy=P.drho*e_dcy*P.cs2, drcz=P.drho*e_dcz*P.cs2;
    real_t bfx=rho*P.gx+P.fx*P.forcing_factor, bfy=rho*P.gy+P.fy*P.forcing_factor, bfz=rho*P.gz+P.fz*P.forcing_factor;
    real_t e_fgx=mu*e_dcx+bfx, e_fgy=mu*e_dcy+bfy, e_fgz=mu*e_dcz+bfz;
    real_t a_over_rho=P.alpha/rho;
    real_t e_fhx=e_dcx-c*a_over_rho*(e_dpx-e_fgx), e_fhy=e_dcy-c*a_over_rho*(e_dpy-e_fgy), e_fhz=e_dcz-c*a_over_rho*(e_dpz-e_fgz);
    real_t u_sq=ux*ux+uy*uy+uz*uz;
    real_t u_fh_e=ux*e_fhx+uy*e_fhy+uz*e_fhz;
    real_t u_drhocd=ux*drcx+uy*drcy+uz*drcz;
    real_t u_fg_e=ux*e_fgx+uy*e_fgy+uz*e_fgz;
    // force site scalars (md gradients)
    real_t f_dpx=gpm_x[i],f_dpy=gpm_y[i],f_dpz=gpm_z[i];
    real_t f_dcx=gcm_x[i],f_dcy=gcm_y[i],f_dcz=gcm_z[i];
    real_t drho_cs2=P.drho*P.cs2;
    real_t drmx=drho_cs2*f_dcx, drmy=drho_cs2*f_dcy, drmz=drho_cs2*f_dcz;
    real_t c_over=c/P.cs2/rho;
    real_t f_fgx=mu*f_dcx+bfx, f_fgy=mu*f_dcy+bfy, f_fgz=mu*f_dcz+bfz;
    real_t f_fhx=f_dcx-c_over*(f_dpx-f_fgx), f_fhy=f_dcy-c_over*(f_dpy-f_fgy), f_fhz=f_dcz-c_over*(f_dpz-f_fgz);
    real_t u_fh_f=ux*f_fhx+uy*f_fhy+uz*f_fhz;
    real_t u_drm=ux*drmx+uy*drmy+uz*drmz;
    real_t u_fg_f=ux*f_fgx+uy*f_fgy+uz*f_fgz;
    real_t gout[19];                 // deferred g writes (stream_inplace only)
    for( int k=0;k<Q;++k ){
      real_t w=(real_t)d_w[k];
      real_t uk=ux*dir_x(k)+uy*dir_y(k)+uz*dir_z(k);
      real_t u_term=uk*(P.alpha+P.beta_c*uk)-P.gamma_c*u_sq;
      real_t G=w*(real_t(1)+u_term);
      real_t f_proj=bfx*dir_x(k)+bfy*dir_y(k)+bfz*dir_z(k);
      real_t cd_c=cd_dir_val(n,cdm,cdp,c_,i,k), cd_p=cd_dir_val(n,cdm,cdp,p_,i,k);
      real_t eh=0, eg=0;
      if( !ge ){
        real_t drho_dk=P.drho*cd_c*P.cs2;
        eh = c*G;
        eh -= real_t(0.5)*G*(cd_c-c*a_over_rho*(cd_p-mu*cd_c-f_proj));
        eh += real_t(0.5)*G*u_fh_e;
        eg = w*(p+rho*P.cs2*u_term);
        eg -= real_t(0.5)*(drho_dk*(G-w)+(mu*cd_c+f_proj)*G);
        eg += real_t(0.5)*((G-w)*u_drhocd+G*u_fg_e);
      }
      real_t fh=0, fg=0;
      if( !gf ){
        real_t dGk=G-w;
        real_t bd_c=bd_dir_val(n,bcase,bc1,bc2,c_,c,i,k), bd_p=bd_dir_val(n,bcase,bc1,bc2,p_,p,i,k);
        real_t dc_dk=real_t(0.5)*(cd_c+bd_c), dp_dk=real_t(0.5)*(cd_p+bd_p);
        real_t drho_dk2=drho_cs2*dc_dk;
        real_t ff=mu*dc_dk+f_proj;
        real_t avgk=avg_dir_val(n,avgnext,lapmu_,lmu,i,k);
        fh=G*(dc_dk-c_over*(dp_dk-ff)-u_fh_f+P.mobility*avgk);
        fg=(drho_dk2-u_drm)*dGk+(ff-u_fg_f)*G;
      }
      real_t gk=g[k*n+i];
      real_t gnew=gk+relax*(eg-gk)+fg;
      if( P.stream_inplace ){
        // reversed-slot writes for in-place streaming: h is write-only (safe in
        // the loop); g is read at slot k in later iterations, so defer.
        h[d_opp[k]*n+i]=eh+fh;
        gout[k]=gnew;
      } else {
        h[k*n+i]=eh+fh;
        g[k*n+i]=gnew;
      }
    }
    if( P.stream_inplace )
      for( int k=0;k<Q;++k ) g[d_opp[k]*n+i]=gout[k];
  }

  __global__ void k_collide_fused_mrt( DevParams P, unsigned char const* is_streamed,
        real_t const* c_, real_t const* rho_, real_t const* p_, real_t const* mu_,
        real_t const* ux_, real_t const* uy_, real_t const* uz_,
        real_t const* gpc_x, real_t const* gpc_y, real_t const* gpc_z,
        real_t const* gcc_x, real_t const* gcc_y, real_t const* gcc_z,
        real_t const* gpm_x, real_t const* gpm_y, real_t const* gpm_z,
        real_t const* gcm_x, real_t const* gcm_y, real_t const* gcm_z,
        real_t const* lapmu_, real_t const* relax_,
        int const* cdm, int const* cdp, int const* bcase, int const* bc1, int const* bc2, int const* avgnext,
        real_t* h, real_t* g )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=P.n ) return;
    int const n=P.n;
    bool ns = ( !is_streamed[i] );                  // eq=0 AND force=0 when true
    real_t c=c_[i], rho=rho_[i], p=p_[i], mu=mu_[i];
    real_t ux=ux_[i], uy=uy_[i], uz=uz_[i], lmu=lapmu_[i];
    real_t drho_cs2=P.drho*P.cs2, rho_cs2=rho*P.cs2, c_over=c/rho_cs2;
    real_t bfx=rho*P.gx+P.fx*P.forcing_factor, bfy=rho*P.gy+P.fy*P.forcing_factor, bfz=rho*P.gz+P.fz*P.forcing_factor;
    // equilibria site scalars (cd gradients)
    real_t e_dpx=gpc_x[i],e_dpy=gpc_y[i],e_dpz=gpc_z[i];
    real_t e_dcx=gcc_x[i],e_dcy=gcc_y[i],e_dcz=gcc_z[i];
    real_t drcx=drho_cs2*e_dcx, drcy=drho_cs2*e_dcy, drcz=drho_cs2*e_dcz;
    real_t e_fgx=mu*e_dcx+bfx, e_fgy=mu*e_dcy+bfy, e_fgz=mu*e_dcz+bfz;
    real_t e_fhx=e_dcx-c_over*(e_dpx-e_fgx), e_fhy=e_dcy-c_over*(e_dpy-e_fgy), e_fhz=e_dcz-c_over*(e_dpz-e_fgz);
    real_t gamma_u_sq=P.gamma_c*(ux*ux+uy*uy+uz*uz);
    real_t u_term_h=ux*e_fhx+uy*e_fhy+uz*e_fhz;
    real_t u_force_g=ux*e_fgx+uy*e_fgy+uz*e_fgz;
    real_t u_drho=ux*drcx+uy*drcy+uz*drcz;
    // force site scalars (md gradients)
    real_t f_dpx=gpm_x[i],f_dpy=gpm_y[i],f_dpz=gpm_z[i];
    real_t f_dcx=gcm_x[i],f_dcy=gcm_y[i],f_dcz=gcm_z[i];
    real_t drmx=drho_cs2*f_dcx, drmy=drho_cs2*f_dcy, drmz=drho_cs2*f_dcz;
    real_t f_fgx=mu*f_dcx+bfx, f_fgy=mu*f_dcy+bfy, f_fgz=mu*f_dcz+bfz;
    real_t f_fhx=f_dcx-c_over*(f_dpx-f_fgx), f_fhy=f_dcy-c_over*(f_dpy-f_fgy), f_fhz=f_dcz-c_over*(f_dpz-f_fgz);
    real_t u_fh_f=ux*f_fhx+uy*f_fhy+uz*f_fhz;
    real_t u_drm=ux*drmx+uy*drmy+uz*drmz;
    real_t u_fg_f=ux*f_fgx+uy*f_fgy+uz*f_fgz;
    real_t cg[19], fgs[19], go[19];
    for( int k=0;k<Q;++k ){
      real_t w=(real_t)d_w[k];
      real_t uk=ux*dir_x(k)+uy*dir_y(k)+uz*dir_z(k);
      real_t u_term=uk*(P.alpha+P.beta_c*uk)-gamma_u_sq;
      real_t G=w*(real_t(1)+u_term);
      real_t w_u_term=G-w;
      real_t f_proj=bfx*dir_x(k)+bfy*dir_y(k)+bfz*dir_z(k);
      real_t cd_c=cd_dir_val(n,cdm,cdp,c_,i,k), cd_p=cd_dir_val(n,cdm,cdp,p_,i,k);
      real_t eh=0, eg=0;
      if( !ns ){
        real_t drho_dk=drho_cs2*cd_c;
        eh = c*G + real_t(0.5)*G*(u_term_h - cd_c + c_over*(cd_p - mu*cd_c - f_proj));
        eg = w*p + rho_cs2*w_u_term
           + real_t(0.5)*((u_drho - drho_dk)*w_u_term + (u_force_g + mu*cd_c + f_proj)*G);
      }
      real_t fh=0, fg=0;
      if( !ns ){
        real_t dGk=G-w;
        real_t bd_c=bd_dir_val(n,bcase,bc1,bc2,c_,c,i,k), bd_p=bd_dir_val(n,bcase,bc1,bc2,p_,p,i,k);
        real_t dc_dk=real_t(0.5)*(cd_c+bd_c), dp_dk=real_t(0.5)*(cd_p+bd_p);
        real_t drho_dk2=drho_cs2*dc_dk;
        real_t ff=mu*dc_dk+f_proj;
        real_t avgk=avg_dir_val(n,avgnext,lapmu_,lmu,i,k);
        fh=G*(dc_dk-c_over*(dp_dk-ff)-u_fh_f+P.mobility*avgk);
        fg=(drho_dk2-u_drm)*dGk+(ff-u_fg_f)*G;
      }
      real_t gk=g[k*n+i];
      int const wk = P.stream_inplace ? d_opp[k] : k;   // reversed slot when in-place
      h[wk*n+i]=eh+fh;                // h not MRT-relaxed
      cg[k]=eg-gk; fgs[k]=fg; go[k]=gk;
    }
    // MRT moment relaxation of the raw g collision term:  cg <- M^-1 S M cg
    if( P.mrt_fast )
    {
      // real_t transform (mrt_fast_transform): identical algebra and summation
      // order as the double path below, in real_t. With real_t==double this is
      // bit-identical; under -DFELBM_SINGLE it avoids the FP64 throughput cliff
      // and the double-register spill of df/xx.
      real_t s0=relax_[i]; real_t tau=real_t(1)/s0-real_t(0.5);
      real_t s1=real_t(2)*tau/(real_t(2)*P.mrt_lambda+tau);
      real_t s[19]={0,s0,s0,0,s1,0,s1,0,s1,s0,s0,s0,s0,s0,s0,s0,s1,s1,s1};
      real_t xx[19];
      #pragma unroll
      for(int k=0;k<19;++k) xx[k]=0;
      #pragma unroll
      for(int nn=0;nn<19;++nn){ real_t y=cg[nn];
        #pragma unroll
        for(int m=1;m<19;++m) xx[m]+=s[m]*d_Mf[m][nn]*y; }
      #pragma unroll
      for(int m=0;m<19;++m){ real_t acc=0;
        #pragma unroll
        for(int nn=0;nn<19;++nn) acc+=d_Minvf[m][nn]*xx[nn];
        cg[m]=acc; }
    }
    else
    {
      double s0=(double)relax_[i]; double tau=1.0/s0-0.5;
      double s1=2.0*tau/(2.0*(double)P.mrt_lambda+tau);
      double s[19]={0.0,s0,s0,0.0,s1,0.0,s1,0.0,s1,s0,s0,s0,s0,s0,s0,s0,s1,s1,s1};
      double df[19], xx[19];
      for(int k=0;k<19;++k){ df[k]=(double)cg[k]; xx[k]=0.0; }
      for(int nn=0;nn<19;++nn){ double y=df[nn]; for(int m=1;m<19;++m) xx[m]+=s[m]*d_M[m][nn]*y; }
      for(int m=0;m<19;++m){ double acc=0.0; for(int nn=0;nn<19;++nn) acc+=d_Minv[m][nn]*xx[nn]; cg[m]=(real_t)acc; }
    }
    if( P.stream_inplace ) for(int k=0;k<Q;++k) g[d_opp[k]*n+i]=go[k]+cg[k]+fgs[k];
    else                   for(int k=0;k<Q;++k) g[k*n+i]      =go[k]+cg[k]+fgs[k];
  }

  // --- flow statistics (monitoring) ------------------------------------------
  // Reduces the 6 timeseries observables on the GPU so a log event downloads
  // 48 bytes instead of six full fields: out[0..4] = sum ux, sum uy, sum uz,
  // sum u^2, sum c (double atomicAdd, cc>=6.0); out[5] = max u^2 via atomicMax
  // on the raw bits (valid because u^2 >= 0, where the double ordering matches
  // the unsigned-integer ordering of the bit patterns). Sum order differs from
  // the old serial host loop at the usual ~1e-12 monitoring-only level.
  __global__ void k_flow_stats( int n, real_t const* ux, real_t const* uy,
                                real_t const* uz, real_t const* c, double* out )
  {
    __shared__ double sh[256];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    double vx=0,vy=0,vz=0,v2=0,cc=0;
    if( i<n ){ vx=(double)ux[i]; vy=(double)uy[i]; vz=(double)uz[i];
               v2=vx*vx+vy*vy+vz*vz; cc=(double)c[i]; }
    double vals[5]={vx,vy,vz,v2,cc};
    for( int q=0;q<5;++q ){
      sh[threadIdx.x]=vals[q]; __syncthreads();
      for( int s2=blockDim.x/2;s2>0;s2>>=1 ){
        if( threadIdx.x<s2 ) sh[threadIdx.x]+=sh[threadIdx.x+s2];
        __syncthreads(); }
      if( threadIdx.x==0 ) atomicAdd( &out[q], sh[0] );
      __syncthreads();
    }
    sh[threadIdx.x]=v2; __syncthreads();
    for( int s2=blockDim.x/2;s2>0;s2>>=1 ){
      if( threadIdx.x<s2 && sh[threadIdx.x+s2]>sh[threadIdx.x] ) sh[threadIdx.x]=sh[threadIdx.x+s2];
      __syncthreads(); }
    if( threadIdx.x==0 )
      atomicMax( (unsigned long long*)&out[5], (unsigned long long)__double_as_longlong(sh[0]) );
  }

  // --- in-place streaming ----------------------------------------------------
  // Requires the fused collision to have written every direction k into slot
  // opp(k) ("reversed layout"). Streaming then reduces to a set of DISJOINT ops
  // built at init from the src table (see multiphase_gpu.cuh):
  //   t=0 SWAP(a,b): bulk pull pair    -- swap the two slots
  //   t=1 AVG (a,b): corner 0.5/0.5    -- both slots get (v_a+v_b)*0.5
  //   t=2 ZERO(a)  : empty row         -- slot = 0
  // Halfway bounce-back and the rest population need no op: the reversed write
  // already left their value in the correct slot. Ops are disjoint (each slot
  // appears in exactly one op or none), so one thread per op is race-free, and
  // the layout is standard again after this kernel. Applied to h and g together.
  __global__ void k_stream_inplace( int nops, int const* oa, int const* ob,
                                    unsigned char const* ot, real_t* h, real_t* g )
  {
    int q = blockIdx.x*blockDim.x + threadIdx.x; if( q>=nops ) return;
    int a=oa[q], b=ob[q]; unsigned char t=ot[q];
    if( t==0 ){
      real_t th=h[a]; h[a]=h[b]; h[b]=th;
      real_t tg=g[a]; g[a]=g[b]; g[b]=tg;
    } else if( t==1 ){
      real_t mh=real_t(0.5)*(h[a]+h[b]); h[a]=mh; h[b]=mh;
      real_t mg=real_t(0.5)*(g[a]+g[b]); g[a]=mg; g[b]=mg;
    } else {
      h[a]=0; g[a]=0;
    }
  }

  // --- matrix-free streaming (drop-in for spmv(A_stream)) --------------------
  // src[j] encodes the streaming source for distribution index j = k*n + i:
  //   src >= 0 : h2[j] = h[src]                 (bulk gather / bounce-back; one Vn source)
  //   src == -1: h2[j] = 0.5*(h[j] + h[opp])    (corner: 0.5/0.5 self + opp-self average)
  //   src == -2: h2[j] = 0                       (empty / solid row)
  // This reproduces the streaming CSR exactly, at ~76 B/site (one int) instead of
  // storing rows+cols+vals (~300 B/site).
  __global__ void k_stream_gather( int Vn, int n, int const* src, real_t const* h, real_t* h2 )
  {
    int j = blockIdx.x*blockDim.x + threadIdx.x; if( j>=Vn ) return;
    int s = src[j];
    if( s >= 0 ) h2[j] = h[s];
    else if( s == -1 ){ int k=j/n, i=j-k*n; int o=d_opp[k]; h2[j]=real_t(0.5)*(h[j]+h[o*n+i]); }
    else h2[j] = real_t(0);
  }

  // --- matrix-free central-difference vector gradient (drop-in for spmv3 A_grad_cd)
  // Per (site i, direction m) a column pair (cminus,cplus) encodes the stencil:
  //   grad(f)[i] = sum_m inv_T*0.5*w_m*e_m*(f[cplus]-f[cminus])   (cminus<0 => skip)
  // The pair reproduces the CPU near-wall cases exactly (central / backward / forward
  // / none); the constant weights are recomputed here instead of stored (~5x less
  // memory than the CSR, and no stored value arrays to stream).
  __global__ void k_grad_cd_mf( int n, real_t alpha,
                                int const* cminus, int const* cplus, real_t const* f,
                                real_t* gx, real_t* gy, real_t* gz )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=n ) return;
    real_t sx=0, sy=0, sz=0;
    for( int m=1;m<Q;++m ){
      int idx=(m-1)*n+i;
      int cm=cminus[idx];
      if( cm<0 ) continue;
      int cp=cplus[idx];
      real_t d = f[cp]-f[cm];
      real_t coef = alpha*real_t(0.5)*(real_t)d_w[m];
      sx += coef*dir_x(m)*d; sy += coef*dir_y(m)*d; sz += coef*dir_z(m)*d;
    }
    gx[i]=sx; gy[i]=sy; gz[i]=sz;
  }

  // --- matrix-free biased vector gradient (drop-in for spmv3 A_grad_bd) --------
  // Per (site,direction): case code + up to two neighbour columns (c1,c2); c0=self.
  //   grad(f)[i] += inv_T*w_m*e_m * (case coefficients . f), reproducing the CPU
  //   create_gradient_matrix_bd cases exactly:
  //     1: 3-point one-sided  (-1.5 f0 + 2.0 f1 - 0.5 f2)   [forward OR backward]
  //     2: 2-point fwd non-BB (-2.0 f0 + 2.0 f1)
  //     3: 2-point fwd  BB    (-1.5 f0 + 1.5 f1)
  //     4: 2-point bwd  BB    ( 0.5 f0 - 0.5 f1)
  //     0: none
  __global__ void k_grad_bd_mf( int n, real_t alpha,
                                int const* bcase, int const* bc1, int const* bc2,
                                real_t const* f, real_t* gx, real_t* gy, real_t* gz )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=n ) return;
    real_t sx=0, sy=0, sz=0;
    real_t f0 = f[i];
    for( int m=1;m<Q;++m ){
      int idx=(m-1)*n+i;
      int cs=bcase[idx];
      if( cs==0 ) continue;
      real_t f1=f[bc1[idx]];
      real_t val;
      if     ( cs==1 ){ real_t f2=f[bc2[idx]]; val = real_t(-1.5)*f0 + real_t(2.0)*f1 + real_t(-0.5)*f2; }
      else if( cs==2 ){ val = real_t(-2.0)*f0 + real_t(2.0)*f1; }
      else if( cs==3 ){ val = real_t(-1.5)*f0 + real_t(1.5)*f1; }
      else            { val = real_t( 0.5)*f0 + real_t(-0.5)*f1; }   // cs==4
      real_t base = alpha*(real_t)d_w[m];
      sx += base*dir_x(m)*val; sy += base*dir_y(m)*val; sz += base*dir_z(m)*val;
    }
    gx[i]=sx; gy[i]=sy; gz[i]=sz;
  }

  // --- matrix-free per-direction derivatives (drop-in for spmv A_cd_dir / A_bd_dir)
  // Same column pairs / case tables as the vector gradients, but the raw per-direction
  // value (no inv_T, no w_m*e_m). Output is the Q*n buffer, direction 0 = 0.
  __global__ void k_cd_dir_mf( int n, int const* cminus, int const* cplus,
                               real_t const* f, real_t* out )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=n ) return;
    out[i] = 0;                                   // rest direction
    for( int m=1;m<Q;++m ){
      int idx=(m-1)*n+i, cm=cminus[idx];
      out[m*n+i] = (cm<0) ? real_t(0) : real_t(0.5)*(f[cplus[idx]]-f[cm]);
    }
  }

  __global__ void k_bd_dir_mf( int n, int const* bcase, int const* bc1, int const* bc2,
                               real_t const* f, real_t* out )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=n ) return;
    out[i] = 0;
    real_t f0=f[i];
    for( int m=1;m<Q;++m ){
      int idx=(m-1)*n+i, cs=bcase[idx];
      real_t v;
      if     ( cs==0 ) v=0;
      else if( cs==1 ){ real_t f1=f[bc1[idx]], f2=f[bc2[idx]]; v=real_t(-1.5)*f0+real_t(2.0)*f1+real_t(-0.5)*f2; }
      else if( cs==2 ){ real_t f1=f[bc1[idx]]; v=real_t(-2.0)*f0+real_t(2.0)*f1; }
      else if( cs==3 ){ real_t f1=f[bc1[idx]]; v=real_t(-1.5)*f0+real_t(1.5)*f1; }
      else            { real_t f1=f[bc1[idx]]; v=real_t(0.5)*f0+real_t(-0.5)*f1; }
      out[m*n+i]=v;
    }
  }

  // --- matrix-free directional average (drop-in for spmv A_avg_dir) -----------
  // out[m*n+i] = 0.5*(x[i]+x[next])  if avgnext>=0  else  x[i]   (m = 0..Q-1)
  __global__ void k_avg_dir_mf( int n, int const* avgnext, real_t const* x, real_t* out )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=n ) return;
    real_t xi=x[i];
    for( int m=0;m<Q;++m ){
      int nx=avgnext[m*n+i];
      out[m*n+i] = (nx<0) ? xi : real_t(0.5)*(xi + x[nx]);
    }
  }

  // --- matrix-free node-centred Laplacian (drop-in for spmv A_lap) ------------
  // Input is the length-2n buffer xtnd = [field(n), bnd(n)]. Per (site,direction):
  //   case 1 central : inv_T*w_m*(f[c1] - 2 f0 + f[c2])
  //   case 2 (non-BB): inv_T*w_m*(2 f[c1] - 2 f0 - 2 bnd)
  //   case 3 (BB)    : inv_T*w_m*(  f[c1] -   f0 -   bnd)
  __global__ void k_lap_mf( int n, real_t alpha, real_t const* xtnd,
                            int const* lcase, int const* lc1, int const* lc2, real_t* out )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=n ) return;
    real_t f0=xtnd[i], bnd=xtnd[n+i], s=0;
    for( int m=1;m<Q;++m ){
      int idx=(m-1)*n+i, cs=lcase[idx];
      if( cs==0 ) continue;
      real_t w=alpha*(real_t)d_w[m];
      if     ( cs==1 ){ real_t f1=xtnd[lc1[idx]], f2=xtnd[lc2[idx]]; s += w*(f1 - real_t(2)*f0 + f2); }
      else if( cs==2 ){ real_t f1=xtnd[lc1[idx]]; s += w*(real_t(2)*f1 - real_t(2)*f0 - real_t(2)*bnd); }
      else            { real_t f1=xtnd[lc1[idx]]; s += w*(f1 - f0 - bnd); }
    }
    out[i]=s;
  }

  // ======================= MRT collision path ================================

  //--- CPU: CollisionModelMRTMultiPhase::compute_all_equilibria (lines 61-123) ---
  //--- eq_h matches BGK; eq_g differs (sign of mu*dc_dk+f_proj), so a separate kernel.
  __global__ void k_equilibria_mrt( DevParams P, unsigned char const* is_streamed,
                                    real_t const* c_, real_t const* rho_, real_t const* p_, real_t const* mu_,
                                    real_t const* ux_, real_t const* uy_, real_t const* uz_,
                                    real_t const* gpc_x, real_t const* gpc_y, real_t const* gpc_z,
                                    real_t const* gcc_x, real_t const* gcc_y, real_t const* gcc_z,
                                    real_t const* gc_dir, real_t const* gp_dir,
                                    real_t* eq_h, real_t* eq_g )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=P.n ) return;
    int const n=P.n;
    if( !is_streamed[i] ){ for(int k=0;k<Q;++k){ eq_h[k*n+i]=0; eq_g[k*n+i]=0; } return; }
    real_t c=c_[i], rho=rho_[i], p=p_[i], mu=mu_[i];
    real_t ux=ux_[i], uy=uy_[i], uz=uz_[i];
    real_t dpx=gpc_x[i],dpy=gpc_y[i],dpz=gpc_z[i];
    real_t dcx=gcc_x[i],dcy=gcc_y[i],dcz=gcc_z[i];
    real_t drho_cs2=P.drho*P.cs2;
    real_t drcx=drho_cs2*dcx, drcy=drho_cs2*dcy, drcz=drho_cs2*dcz;   // drho_cd
    real_t bfx=rho*P.gx+P.fx*P.forcing_factor, bfy=rho*P.gy+P.fy*P.forcing_factor, bfz=rho*P.gz+P.fz*P.forcing_factor;
    real_t rho_cs2=rho*P.cs2;
    real_t c_over=c/rho_cs2;
    real_t fgx=mu*dcx+bfx, fgy=mu*dcy+bfy, fgz=mu*dcz+bfz;            // force_g
    real_t fhx=dcx-c_over*(dpx-fgx), fhy=dcy-c_over*(dpy-fgy), fhz=dcz-c_over*(dpz-fgz); // force_h
    real_t gamma_u_sq=P.gamma_c*(ux*ux+uy*uy+uz*uz);
    real_t u_term_h=ux*fhx+uy*fhy+uz*fhz;
    real_t u_force_g=ux*fgx+uy*fgy+uz*fgz;
    real_t u_drho=ux*drcx+uy*drcy+uz*drcz;
    for( int k=0;k<Q;++k ){
      real_t w=(real_t)d_w[k];
      real_t uk=ux*dir_x(k)+uy*dir_y(k)+uz*dir_z(k);
      real_t w_u_term=w*(uk*(P.alpha+P.beta_c*uk)-gamma_u_sq);
      real_t G=w+w_u_term;
      real_t dc_dk=gc_dir[k*n+i], dp_dk=gp_dir[k*n+i];
      real_t drho_dk=drho_cs2*dc_dk;
      real_t f_proj=bfx*dir_x(k)+bfy*dir_y(k)+bfz*dir_z(k);
      eq_h[k*n+i] = c*G + real_t(0.5)*G*(u_term_h - dc_dk + c_over*(dp_dk - mu*dc_dk - f_proj));
      eq_g[k*n+i] = w*p + rho_cs2*w_u_term
                  + real_t(0.5)*((u_drho - drho_dk)*w_u_term + (u_force_g + mu*dc_dk + f_proj)*G);
    }
  }

  //--- coll_h = eq_h - h ;  coll_g = eq_g - g   (raw; MRT relaxes g afterwards) ---
  __global__ void k_collision_term_raw( int n_var, real_t const* eq_h, real_t const* eq_g,
                                        real_t const* h, real_t const* g,
                                        real_t* coll_h, real_t* coll_g )
  {
    int j = blockIdx.x*blockDim.x + threadIdx.x; if( j>=n_var ) return;
    coll_h[j]=eq_h[j]-h[j];
    coll_g[j]=eq_g[j]-g[j];
  }

  //--- CPU MRT moment relaxation of the g collision term (lines 168-233):
  //---   coll_g <- M^-1 * S * M * coll_g,  per site.  S built from relax[i] + lambda.
  __global__ void k_mrt_relax_g( DevParams P, real_t const* relax, real_t* coll_g )
  {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=P.n ) return;
    int const n=P.n;
    if( P.mrt_fast )
    {
      real_t s0=relax[i]; real_t tau=real_t(1)/s0-real_t(0.5);
      real_t s1=real_t(2)*tau/(real_t(2)*P.mrt_lambda+tau);
      real_t s[19]={0,s0,s0,0,s1,0,s1,0,s1,s0,s0,s0,s0,s0,s0,s0,s1,s1,s1};
      real_t df[19], x[19];
      #pragma unroll
      for(int k=0;k<19;++k){ df[k]=coll_g[k*n+i]; x[k]=0; }
      #pragma unroll
      for(int nn=0;nn<19;++nn){ real_t y=df[nn];
        #pragma unroll
        for(int m=1;m<19;++m) x[m]+=s[m]*d_Mf[m][nn]*y; }
      #pragma unroll
      for(int m=0;m<19;++m){ real_t acc=0;
        #pragma unroll
        for(int nn=0;nn<19;++nn) acc+=d_Minvf[m][nn]*x[nn];
        coll_g[m*n+i]=acc; }
      return;
    }
    double s0  = (double)relax[i];              // = 1/(tau+0.5)
    double tau = 1.0/s0 - 0.5;
    double s1  = 2.0*tau/(2.0*(double)P.mrt_lambda + tau);
    double s[19] = { 0.0, s0, s0, 0.0, s1, 0.0, s1, 0.0, s1, s0, s0, s0, s0, s0, s0, s0, s1, s1, s1 };
    double df[19], x[19];
    for(int k=0;k<19;++k){ df[k]=(double)coll_g[k*n+i]; x[k]=0.0; }
    // x = S * M * df   (s[0]=0, so row 0 stays 0)
    for(int nn=0;nn<19;++nn){ double y=df[nn]; for(int m=1;m<19;++m) x[m]+=s[m]*d_M[m][nn]*y; }
    // coll_g = M^-1 * x
    for(int m=0;m<19;++m){ double acc=0.0; for(int nn=0;nn<19;++nn) acc+=d_Minv[m][nn]*x[nn]; coll_g[m*n+i]=(real_t)acc; }
  }

  // ============== order-parameter mass-conservation corrector ================
  // CPU: MassConservationCorrector. c_i = sum_k h_k[i]; phi_i = c_i(1-c_i)>=0.
  // Reduce M = sum c_i and W = sum phi_i (block reduction -> atomicAdd per block),
  // then inject delta_c_i = -lambda*phi_i (lambda = (M-M0)/W) as h_k += w_k*delta_c
  // (changes c by delta_c, adds no momentum since sum_k w_k e_k = 0).

  __global__ void k_mass_weight( DevParams P, unsigned char const* is_streamed,
                                 real_t const* h, double* out /*[2] = {M,W}*/ )
  {
    __shared__ double sM[BLOCK];
    __shared__ double sW[BLOCK];
    int const tid = threadIdx.x;
    int const i   = blockIdx.x*blockDim.x + tid;
    double c=0.0, phi=0.0;
    if( i<P.n && is_streamed[i] ){
      int const n=P.n;
      for(int k=0;k<Q;++k) c += (double)h[k*n+i];
      phi = c*(1.0-c); if(phi<0.0) phi=0.0;
    }
    sM[tid]=c; sW[tid]=phi;
    __syncthreads();
    for(int s=blockDim.x/2; s>0; s>>=1){ if(tid<s){ sM[tid]+=sM[tid+s]; sW[tid]+=sW[tid+s]; } __syncthreads(); }
    if(tid==0){ atomicAdd(&out[0], sM[0]); atomicAdd(&out[1], sW[0]); }
  }

  __global__ void k_inject_mass( DevParams P, unsigned char const* is_streamed,
                                 double lambda, real_t* h )
  {
    int const i = blockIdx.x*blockDim.x + threadIdx.x; if( i>=P.n ) return;
    if( !is_streamed[i] ) return;
    int const n=P.n;
    double c=0.0; for(int k=0;k<Q;++k) c += (double)h[k*n+i];
    double phi = c*(1.0-c);
    if( phi<=0.0 ) return;
    double const delta_c = -lambda*phi;
    for(int k=0;k<Q;++k) h[k*n+i] += (real_t)((double)d_w[k]*delta_c);
  }

#endif // __CUDACC__

} // namespace felbm_gpu

#endif // FELBM_GPU_DEVICE_ENGINE_CUH
