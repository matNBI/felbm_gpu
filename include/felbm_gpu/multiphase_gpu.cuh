#ifndef FELBM_GPU_MULTIPHASE_GPU_CUH
#define FELBM_GPU_MULTIPHASE_GPU_CUH

//=============================================================================
//  MultiPhaseGPU — owns the device state + operators and runs one time step.
//
//  Factored out of the driver so BOTH the app (felbm_gpu_main.cu) and the
//  CPU-vs-GPU comparison harness (compare_cpu_gpu.cu) drive the *identical* GPU
//  engine — the test therefore validates the real code path, not a copy.
//
//  Include AFTER <lbm.h> (needs lbm::SubDomain / VelocitySet / Settings /
//  ParametersMultiPhase).
//=============================================================================

#include "device_engine.cuh"
#include "device_csr.cuh"
#include "operator_access.h"

#include <vector>

namespace felbm_gpu
{
  struct MultiPhaseGPU
  {
    int n=0, V=0, Vn=0;
    bool use_mrt=false;
    DevParams P;

    DeviceCSR  A_stream, A_lap, A_cd_dir, A_bd_dir, A_avg_dir;
    DeviceCSR3 A_grad_cd, A_grad_bd;

    real_t *d_h=0,*d_g=0,*d_h2=0,*d_g2=0;
    real_t *d_c=0,*d_p=0,*d_rho=0,*d_mu=0,*d_ux=0,*d_uy=0,*d_uz=0,*d_lapc=0,*d_lapmu=0;
    real_t *d_gcc_x=0,*d_gcc_y=0,*d_gcc_z=0,*d_gcb_x=0,*d_gcb_y=0,*d_gcb_z=0,*d_gcm_x=0,*d_gcm_y=0,*d_gcm_z=0;
    real_t *d_gpc_x=0,*d_gpc_y=0,*d_gpc_z=0,*d_gpb_x=0,*d_gpb_y=0,*d_gpb_z=0,*d_gpm_x=0,*d_gpm_y=0,*d_gpm_z=0;
    real_t *d_relax=0,*d_gc_cd=0,*d_gp_cd=0,*d_gc_bd=0,*d_gp_bd=0,*d_avg=0;
    real_t *d_eqh=0,*d_eqg=0,*d_collh=0,*d_collg=0,*d_fh=0,*d_fg=0;
    real_t *d_xtnd=0;                    // length-2n input for the node-centred Laplacian
    unsigned char *d_solid=0,*d_stream=0;

    // ---- setup: build+upload operators, allocate state, fill parameters -------
    void init( lbm::SubDomain const & sd, lbm::VelocitySet const & vs,
               lbm::Settings const & s, lbm::ParametersMultiPhase const & param )
    {
      n  = (int)sd.size_sites();
      V  = (int)vs.size();
      Vn = V*n;

      FieldOperatorGPU     fop( sd, vs, s );
      StreamingOperatorGPU stream_op( sd, vs, s );
      upload_d3q19_constants();

      A_stream.upload(  stream_op.rows(), stream_op.cols(), stream_op.values() );
      A_grad_cd.upload( fop.rows_cd(), fop.cols_cd(), fop.grad_cd_x(), fop.grad_cd_y(), fop.grad_cd_z() );
      A_grad_bd.upload( fop.rows_bd(), fop.cols_bd(), fop.grad_bd_x(), fop.grad_bd_y(), fop.grad_bd_z() );
      A_lap.upload(     fop.rows_lap(), fop.cols_lap(), fop.laplacian() );
      A_cd_dir.upload(  fop.rows_cd_dir(),  fop.cols_cd_dir(),  fop.grad_cd_dir() );
      A_bd_dir.upload(  fop.rows_bd_dir(),  fop.cols_bd_dir(),  fop.grad_bd_dir() );
      A_avg_dir.upload( fop.rows_avg_dir(), fop.cols_avg_dir(), fop.values_avg_dir() );

      auto A=[&](int m){ return device_alloc<real_t>((size_t)m); };
      d_h=A(Vn);d_g=A(Vn);d_h2=A(Vn);d_g2=A(Vn);
      d_c=A(n);d_p=A(n);d_rho=A(n);d_mu=A(n);d_ux=A(n);d_uy=A(n);d_uz=A(n);d_lapc=A(n);d_lapmu=A(n);
      d_gcc_x=A(n);d_gcc_y=A(n);d_gcc_z=A(n);d_gcb_x=A(n);d_gcb_y=A(n);d_gcb_z=A(n);d_gcm_x=A(n);d_gcm_y=A(n);d_gcm_z=A(n);
      d_gpc_x=A(n);d_gpc_y=A(n);d_gpc_z=A(n);d_gpb_x=A(n);d_gpb_y=A(n);d_gpb_z=A(n);d_gpm_x=A(n);d_gpm_y=A(n);d_gpm_z=A(n);
      d_relax=A(Vn);d_gc_cd=A(Vn);d_gp_cd=A(Vn);d_gc_bd=A(Vn);d_gp_bd=A(Vn);d_avg=A(Vn);
      d_eqh=A(Vn);d_eqg=A(Vn);d_collh=A(Vn);d_collg=A(Vn);d_fh=A(Vn);d_fg=A(Vn);
      d_xtnd=A(2*n);
      d_solid=device_alloc<unsigned char>(n); d_stream=device_alloc<unsigned char>(n);

      { std::vector<unsigned char> sol(n),st(n);
        for(int i=0;i<n;++i){ sol[i]=sd.is_solid(i)?1:0; st[i]=sd.is_streamed(i)?1:0; }
        copy_h2d(d_solid,sol.data(),n); copy_h2d(d_stream,st.data(),n); }

      P.n=n;
      P.rho0=(real_t)param.phase_density(0u); P.rho1=(real_t)param.phase_density(1u); P.drho=P.rho0-P.rho1;
      P.kappa=(real_t)param.kappa(); P.beta=(real_t)param.beta(); P.four_beta=(real_t)(4.0*param.beta());
      P.bnd_coeff=(real_t)param.boundary_coefficient();
      P.tau0=(real_t)param.relaxation_time(0u); P.tau1=(real_t)param.relaxation_time(1u);
      P.cs2=(real_t)CS2; P.alpha=(real_t)ALPHA; P.gamma_c=(real_t)GAMMA; P.beta_c=(real_t)BETA;
      P.mobility=(real_t)param.mobility(); P.forcing_factor=(real_t)param.forcing_factor();
      lbm::Vector3d gv=s.acceleration(), fv=s.forcing();
      P.gx=(real_t)gv[0u]; P.gy=(real_t)gv[1u]; P.gz=(real_t)gv[2u];
      P.fx=(real_t)fv[0u]; P.fy=(real_t)fv[1u]; P.fz=(real_t)fv[2u];
      P.mrt_lambda=(real_t)s.mrt_lambda();

      use_mrt = s.use_mrt();
      // Upload the D3Q19 MRT transform matrices from the CPU statics (cheap; safe
      // even for BGK runs, which simply never use them).
      upload_mrt_matrices( &lbm::EquilibriumDistributionMRT::m_mass_matrix[0][0],
                           &lbm::EquilibriumDistributionMRT::m_mass_matrix_inv[0][0] );
    }

    // ---- upload an initial condition (host double distributions, Vn each) -----
    void upload_state( double const * h, double const * g )
    {
      std::vector<real_t> t(Vn);
      for(int j=0;j<Vn;++j) t[j]=(real_t)h[j]; copy_h2d(d_h,t.data(),Vn);
      for(int j=0;j<Vn;++j) t[j]=(real_t)g[j]; copy_h2d(d_g,t.data(),Vn);
    }

    // ---- one time step (mirrors TimeStepperMultiPhase::do_time_step) ----------
    void step()
    {
      dim3 gN=grid_1d(n,BLOCK), gVn=grid_1d(Vn,BLOCK);
      size_t const Vb=(size_t)Vn*sizeof(real_t);

      k_moments<<<gN,BLOCK>>>( P, d_h,d_g, d_c,d_p,d_rho,d_mu, d_ux,d_uy,d_uz, d_relax ); GPU_CHECK_KERNEL();
      spmv3( A_grad_cd, d_c, d_gcc_x,d_gcc_y,d_gcc_z );
      spmv3( A_grad_bd, d_c, d_gcb_x,d_gcb_y,d_gcb_z );
      k_grad_md<<<gN,BLOCK>>>( n, d_gcc_x,d_gcc_y,d_gcc_z, d_gcb_x,d_gcb_y,d_gcb_z, d_gcm_x,d_gcm_y,d_gcm_z ); GPU_CHECK_KERNEL();
      // Laplacian(c): pack [c, bnd_cond] (2n) then SpMV (node-centred, wetting BC)
      k_pack_lap_c<<<gN,BLOCK>>>( P, d_c, d_xtnd ); GPU_CHECK_KERNEL();
      spmv( A_lap, d_xtnd, d_lapc );
      k_mu_axpy<<<gN,BLOCK>>>( n, P.kappa, d_lapc, d_mu ); GPU_CHECK_KERNEL();
      // Laplacian(mu): boundary term is zero -> pack [mu, 0]
      k_pack_lap_zero<<<gN,BLOCK>>>( n, d_mu, d_xtnd ); GPU_CHECK_KERNEL();
      spmv( A_lap, d_xtnd, d_lapmu );
      k_vel_press_corr<<<gN,BLOCK>>>( P, d_mu,d_rho, d_gcc_x,d_gcc_y,d_gcc_z, d_ux,d_uy,d_uz, d_p ); GPU_CHECK_KERNEL();
      spmv3( A_grad_cd, d_p, d_gpc_x,d_gpc_y,d_gpc_z );
      spmv3( A_grad_bd, d_p, d_gpb_x,d_gpb_y,d_gpb_z );
      k_grad_md<<<gN,BLOCK>>>( n, d_gpc_x,d_gpc_y,d_gpc_z, d_gpb_x,d_gpb_y,d_gpb_z, d_gpm_x,d_gpm_y,d_gpm_z ); GPU_CHECK_KERNEL();
      GPU_CHECK( cudaMemset(d_gc_cd,0,Vb) ); spmv( A_cd_dir, d_c, d_gc_cd+n );
      GPU_CHECK( cudaMemset(d_gp_cd,0,Vb) ); spmv( A_cd_dir, d_p, d_gp_cd+n );

      GPU_CHECK( cudaMemset(d_gc_bd,0,Vb) ); spmv( A_bd_dir, d_c, d_gc_bd+n );
      GPU_CHECK( cudaMemset(d_gp_bd,0,Vb) ); spmv( A_bd_dir, d_p, d_gp_bd+n );
      spmv( A_avg_dir, d_lapmu, d_avg );
      k_force_term<<<gN,BLOCK>>>( P, d_stream, d_c,d_rho,d_mu, d_ux,d_uy,d_uz,
                                  d_gpm_x,d_gpm_y,d_gpm_z, d_gcm_x,d_gcm_y,d_gcm_z,
                                  d_gc_cd,d_gc_bd, d_gp_cd,d_gp_bd, d_avg, d_fh,d_fg ); GPU_CHECK_KERNEL();

      if( use_mrt )
      {
        k_equilibria_mrt<<<gN,BLOCK>>>( P, d_stream, d_c,d_rho,d_p,d_mu, d_ux,d_uy,d_uz,
                                        d_gpc_x,d_gpc_y,d_gpc_z, d_gcc_x,d_gcc_y,d_gcc_z,
                                        d_gc_cd,d_gp_cd, d_eqh,d_eqg ); GPU_CHECK_KERNEL();
        k_collision_term_raw<<<gVn,BLOCK>>>( Vn, d_eqh,d_eqg, d_h,d_g, d_collh,d_collg ); GPU_CHECK_KERNEL();
        k_mrt_relax_g<<<gN,BLOCK>>>( P, d_relax, d_collg ); GPU_CHECK_KERNEL();
      }
      else
      {
        k_equilibria<<<gN,BLOCK>>>( P, d_solid,d_stream, d_c,d_rho,d_p,d_mu, d_ux,d_uy,d_uz,
                                    d_gpc_x,d_gpc_y,d_gpc_z, d_gcc_x,d_gcc_y,d_gcc_z,
                                    d_gc_cd,d_gp_cd, d_eqh,d_eqg ); GPU_CHECK_KERNEL();
        k_collision_term<<<gVn,BLOCK>>>( Vn, d_eqh,d_eqg, d_h,d_g, d_relax, d_collh,d_collg ); GPU_CHECK_KERNEL();
      }
      k_collide_apply<<<gVn,BLOCK>>>( Vn, d_collh,d_collg, d_fh,d_fg, d_h,d_g ); GPU_CHECK_KERNEL();

      spmv( A_stream, d_h, d_h2 );
      spmv( A_stream, d_g, d_g2 );
      std::swap(d_h,d_h2); std::swap(d_g,d_g2);
    }

    // ---- recompute + download the macroscopic fields (post-step) --------------
    // (runs the fields pass so c/rho/u/p match the current h,g exactly)
    void download( std::vector<double>& c, std::vector<double>& rho, std::vector<double>& ux,
                   std::vector<double>& uy, std::vector<double>& uz, std::vector<double>& p )
    {
      dim3 gN=grid_1d(n,BLOCK);
      // refresh the primary fields from the current distributions (moments +
      // the same velocity/pressure correction the step applies)
      k_moments<<<gN,BLOCK>>>( P, d_h,d_g, d_c,d_p,d_rho,d_mu, d_ux,d_uy,d_uz, d_relax ); GPU_CHECK_KERNEL();
      spmv3( A_grad_cd, d_c, d_gcc_x,d_gcc_y,d_gcc_z );
      k_pack_lap_c<<<gN,BLOCK>>>( P, d_c, d_xtnd ); GPU_CHECK_KERNEL();
      spmv( A_lap, d_xtnd, d_lapc );
      k_mu_axpy<<<gN,BLOCK>>>( n, P.kappa, d_lapc, d_mu ); GPU_CHECK_KERNEL();
      k_vel_press_corr<<<gN,BLOCK>>>( P, d_mu,d_rho, d_gcc_x,d_gcc_y,d_gcc_z, d_ux,d_uy,d_uz, d_p ); GPU_CHECK_KERNEL();

      c.resize(n);rho.resize(n);ux.resize(n);uy.resize(n);uz.resize(n);p.resize(n);
      std::vector<real_t> t(n);
      auto grab=[&](real_t* dp, std::vector<double>& o){ copy_d2h(t.data(),dp,n); for(int i=0;i<n;++i)o[i]=(double)t[i]; };
      grab(d_c,c); grab(d_rho,rho); grab(d_ux,ux); grab(d_uy,uy); grab(d_uz,uz); grab(d_p,p);
    }

    void free()
    {
      A_stream.free();A_lap.free();A_cd_dir.free();A_bd_dir.free();A_avg_dir.free();A_grad_cd.free();A_grad_bd.free();
      real_t* arr[]={d_h,d_g,d_h2,d_g2,d_c,d_p,d_rho,d_mu,d_ux,d_uy,d_uz,d_lapc,d_lapmu,
        d_gcc_x,d_gcc_y,d_gcc_z,d_gcb_x,d_gcb_y,d_gcb_z,d_gcm_x,d_gcm_y,d_gcm_z,
        d_gpc_x,d_gpc_y,d_gpc_z,d_gpb_x,d_gpb_y,d_gpb_z,d_gpm_x,d_gpm_y,d_gpm_z,
        d_relax,d_gc_cd,d_gp_cd,d_gc_bd,d_gp_bd,d_avg,d_eqh,d_eqg,d_collh,d_collg,d_fh,d_fg};
      for(real_t* p2:arr) device_free(p2);
      device_free(d_xtnd);
      device_free(d_solid); device_free(d_stream);
    }
  };

} // namespace felbm_gpu

#endif // FELBM_GPU_MULTIPHASE_GPU_CUH
