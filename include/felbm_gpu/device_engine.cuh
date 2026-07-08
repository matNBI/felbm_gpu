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
  };

#ifdef __CUDACC__

  __device__ __forceinline__ real_t dir_x(int k){ return (real_t)d_cx[k]; }
  __device__ __forceinline__ real_t dir_y(int k){ return (real_t)d_cy[k]; }
  __device__ __forceinline__ real_t dir_z(int k){ return (real_t)d_cz[k]; }

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
    for( int k=0;k<Q;++k) relax[k*n+i]=rp;
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
  __global__ void k_collision_term( int n_var, real_t const* eq_h, real_t const* eq_g,
                                    real_t const* h, real_t const* g, real_t const* relax,
                                    real_t* coll_h, real_t* coll_g )
  {
    int j = blockIdx.x*blockDim.x + threadIdx.x; if( j>=n_var ) return;
    coll_h[j]=eq_h[j]-h[j];
    coll_g[j]=relax[j]*(eq_g[j]-g[j]);
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

#endif // __CUDACC__

} // namespace felbm_gpu

#endif // FELBM_GPU_DEVICE_ENGINE_CUH
