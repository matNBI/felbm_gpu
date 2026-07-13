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
    bool correct_mass=false;
    bool mf_stream=false;      // matrix-free streaming instead of stored CSR SpMV
    bool mf_grad=false;        // matrix-free central-difference vector gradient (grad_cd)
    bool fused=false;          // fold cd_dir/bd_dir/avg into equilibria+force (implies mf_grad)
    bool fuse_coll=false;      // fuse equilibria+force+collision+apply into one kernel (implies fused)
    double target_mass=0.0;
    double* d_reduce=0;        // [2] device scratch for {M, W}
    int*    d_src=0;           // matrix-free streaming source-code table (Vn ints)
    int*    d_cdm=0;           // grad_cd column pair: minus / plus  ((Q-1)*n ints each)
    int*    d_cdp=0;
    int*    d_bcase=0;         // grad_bd: case code + two neighbour columns
    int*    d_bc1=0, *d_bc2=0;
    int*    d_avgnext=0;       // avg_dir: per-direction average neighbour (Q*n)
    int*    d_lcase=0;         // lap: case code + two neighbour columns ((Q-1)*n)
    int*    d_lc1=0, *d_lc2=0;
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
    //  mf = matrix-free streaming (source table instead of the stored CSR SpMV).
    void init( lbm::SubDomain const & sd, lbm::VelocitySet const & vs,
               lbm::Settings const & s, lbm::ParametersMultiPhase const & param,
               bool mf = false, bool mfg = false, bool fu = false, bool fc = false )
    {
      n  = (int)sd.size_sites();
      V  = (int)vs.size();
      Vn = V*n;
      mf_stream = mf;
      fuse_coll = fc;
      fused     = fu || fc;      // collision-fusion implies dir-fusion
      mf_grad   = mfg || fused;  // fusion recomputes the dir stencils from the mf tables

      // NB: the stored CSR field operators (FieldOperatorGPU) are the dominant host-
      // memory cost at large N (~100 GB at 300^3). They are only needed for the CSR
      // path, so they are built lazily inside the `!mf_grad` block below — NOT here.
      StreamingOperatorGPU stream_op( sd, vs, s );  // streaming src table needs its rows/cols
      upload_d3q19_constants();

      if( mf_grad )
      {
        // Build the grad_cd column-pair table (host, one-time) reproducing the CPU
        // create_gradient_matrix_cd near-wall case selection exactly.
        int const Q1n = (V-1)*n;
        std::vector<int> cm( Q1n ), cp( Q1n );
        int const Nx=(int)sd.size_x(), Ny=(int)sd.size_y(), Nz=(int)sd.size_z();
        bool const hbb = s.use_halfway_bb();
      #pragma omp parallel for schedule(static)
        for( int i=0;i<n;++i ){
          unsigned ux,uy,uz; sd.idx_to_coords( (unsigned)i, ux,uy,uz );
          int x=(int)ux,y=(int)uy,z=(int)uz;
          for( int m=1;m<V;++m ){
            int ex=H_CX[m], ey=H_CY[m], ez=H_CZ[m], mo=H_OPP[m];
            unsigned inx = sd.coords_to_idx( (unsigned)(((x+ex)%Nx+Nx)%Nx),
                                             (unsigned)(((y+ey)%Ny+Ny)%Ny),
                                             (unsigned)(((z+ez)%Nz+Nz)%Nz) );
            unsigned ipx = sd.coords_to_idx( (unsigned)(((x-ex)%Nx+Nx)%Nx),
                                             (unsigned)(((y-ey)%Ny+Ny)%Ny),
                                             (unsigned)(((z-ez)%Nz+Nz)%Nz) );
            bool Km = sd.is_known( (unsigned)i, (unsigned)m );
            bool Ko = sd.is_known( (unsigned)i, (unsigned)mo );
            bool Nok = !sd.is_solid( inx );
            bool Pok = !sd.is_solid( ipx );
            int a=-1, b=-1;
            if     ( Km && Ko && Pok && Nok ){ a=(int)ipx; b=(int)inx; }  // central
            else if( hbb && Km && Pok )      { a=(int)ipx; b=i;        }  // backward
            else if( hbb && Ko && Nok )      { a=i;        b=(int)inx; }  // forward
            cm[(m-1)*n+i]=a; cp[(m-1)*n+i]=b;
          }
        }
        d_cdm=device_alloc<int>((size_t)Q1n); copy_h2d(d_cdm, cm.data(), (size_t)Q1n);
        d_cdp=device_alloc<int>((size_t)Q1n); copy_h2d(d_cdp, cp.data(), (size_t)Q1n);

        // grad_bd case tables (reproduce create_gradient_matrix_bd exactly)
        std::vector<int> bcs( Q1n ), bc1( Q1n ), bc2( Q1n );
      #pragma omp parallel for schedule(static)
        for( int i=0;i<n;++i ){
          unsigned ux,uy,uz; sd.idx_to_coords( (unsigned)i, ux,uy,uz );
          int x=(int)ux,y=(int)uy,z=(int)uz;
          for( int m=1;m<V;++m ){
            int ex=H_CX[m], ey=H_CY[m], ez=H_CZ[m], mo=H_OPP[m];
            auto CI=[&]( int dx,int dy,int dz )->int{
              return (int)sd.coords_to_idx( (unsigned)(((x+dx)%Nx+Nx)%Nx),
                                            (unsigned)(((y+dy)%Ny+Ny)%Ny),
                                            (unsigned)(((z+dz)%Nz+Nz)%Nz) ); };
            int inx=CI(ex,ey,ez), in2=CI(2*ex,2*ey,2*ez), ipx=CI(-ex,-ey,-ez), ip2=CI(-2*ex,-2*ey,-2*ez);
            bool Km=sd.is_known((unsigned)i,(unsigned)m), Ko=sd.is_known((unsigned)i,(unsigned)mo);
            int cs=0,c1=-1,c2=-1;
            if( Ko && !sd.is_solid((unsigned)inx) ){                                   // forward
              if( sd.is_known((unsigned)inx,(unsigned)mo) && !sd.is_solid((unsigned)in2) ){ cs=1; c1=inx; c2=in2; }
              else if( !hbb ){ cs=2; c1=inx; }
              else           { cs=3; c1=inx; }
            } else {                                                                    // backward
              if( !hbb ){
                if( Km && !sd.is_solid((unsigned)ipx) && sd.is_known((unsigned)ipx,(unsigned)m) && !sd.is_solid((unsigned)ip2) ){ cs=1; c1=ipx; c2=ip2; }
              } else {
                if( Km && !sd.is_solid((unsigned)ipx) ){ cs=4; c1=ipx; }
              }
            }
            bcs[(m-1)*n+i]=cs; bc1[(m-1)*n+i]=c1; bc2[(m-1)*n+i]=c2;
          }
        }
        d_bcase=device_alloc<int>((size_t)Q1n); copy_h2d(d_bcase, bcs.data(), (size_t)Q1n);
        d_bc1  =device_alloc<int>((size_t)Q1n); copy_h2d(d_bc1,   bc1.data(), (size_t)Q1n);
        d_bc2  =device_alloc<int>((size_t)Q1n); copy_h2d(d_bc2,   bc2.data(), (size_t)Q1n);

        // avg_dir table (create_averaging_matrix_dir): per-direction neighbour to
        // average with, or -1 for the self-only case. m = 0..Q-1 -> Q*n.
        std::vector<int> avn( (size_t)Vn );
        // lap table (create_laplacian_matrix): case + up to two field columns.
        std::vector<int> lcs( Q1n ), l1( Q1n ), l2( Q1n );
      #pragma omp parallel for schedule(static)
        for( int i=0;i<n;++i ){
          unsigned ux,uy,uz; sd.idx_to_coords( (unsigned)i, ux,uy,uz );
          int x=(int)ux,y=(int)uy,z=(int)uz;
          auto CI=[&]( int dx,int dy,int dz )->int{
            return (int)sd.coords_to_idx( (unsigned)(((x+dx)%Nx+Nx)%Nx),
                                          (unsigned)(((y+dy)%Ny+Ny)%Ny),
                                          (unsigned)(((z+dz)%Nz+Nz)%Nz) ); };
          for( int m=0;m<V;++m ){                       // avg_dir: m = 0..Q-1
            int ex=H_CX[m],ey=H_CY[m],ez=H_CZ[m],mo=H_OPP[m];
            int nx = CI(ex,ey,ez);
            avn[(size_t)m*n+i] = ( sd.is_known((unsigned)i,(unsigned)mo) && !sd.is_solid((unsigned)nx) ) ? nx : -1;
          }
          for( int m=1;m<V;++m ){                       // lap: m = 1..Q-1
            int ex=H_CX[m],ey=H_CY[m],ez=H_CZ[m],mo=H_OPP[m];
            int inx=CI(ex,ey,ez), ipx=CI(-ex,-ey,-ez);
            bool Km=sd.is_known((unsigned)i,(unsigned)m), Ko=sd.is_known((unsigned)i,(unsigned)mo);
            bool Pok=!sd.is_solid((unsigned)ipx), Nok=!sd.is_solid((unsigned)inx);
            int cs=0,c1=-1,c2=-1;
            if     ( Km && Ko && Pok && Nok ){ cs=1; c1=ipx; c2=inx; }       // central
            else if( Km && Pok )             { cs=hbb?3:2; c1=ipx; }         // backward
            else if( Ko && Nok )             { cs=hbb?3:2; c1=inx; }         // forward
            lcs[(m-1)*n+i]=cs; l1[(m-1)*n+i]=c1; l2[(m-1)*n+i]=c2;
          }
        }
        d_avgnext=device_alloc<int>((size_t)Vn);  copy_h2d(d_avgnext, avn.data(), (size_t)Vn);
        d_lcase  =device_alloc<int>((size_t)Q1n); copy_h2d(d_lcase, lcs.data(), (size_t)Q1n);
        d_lc1    =device_alloc<int>((size_t)Q1n); copy_h2d(d_lc1,   l1.data(),  (size_t)Q1n);
        d_lc2    =device_alloc<int>((size_t)Q1n); copy_h2d(d_lc2,   l2.data(),  (size_t)Q1n);
      }

      if( mf_stream )
      {
        // Build the compact streaming source table from the CSR (host), upload it,
        // and skip the stored streaming operator entirely (~4x less streaming memory).
        auto const & R = stream_op.rows();
        auto const & C = stream_op.cols();
        std::vector<int> src( Vn );
        for( int j=0;j<Vn;++j ){
          unsigned a=R[j], b=R[j+1]; int nnz=(int)(b-a);
          if     ( nnz==1 ) src[j] = (int)C[a];   // single Vn source (bulk / bounce-back)
          else if( nnz==2 ) src[j] = -1;          // 0.5/0.5 self + opp-self average
          else              src[j] = -2;          // empty
        }
        d_src = device_alloc<int>( (size_t)Vn );
        copy_h2d( d_src, src.data(), (size_t)Vn );
      }
      else
      {
        A_stream.upload(  stream_op.rows(), stream_op.cols(), stream_op.values() );
      }
      if( !mf_grad ){   // stored CSR path: build the host operators only now, use, free
        FieldOperatorGPU fop( sd, vs, s );
        A_grad_cd.upload( fop.rows_cd(), fop.cols_cd(), fop.grad_cd_x(), fop.grad_cd_y(), fop.grad_cd_z() );
        A_grad_bd.upload( fop.rows_bd(), fop.cols_bd(), fop.grad_bd_x(), fop.grad_bd_y(), fop.grad_bd_z() );
        A_lap.upload(     fop.rows_lap(), fop.cols_lap(), fop.laplacian() );
        A_cd_dir.upload(  fop.rows_cd_dir(),  fop.cols_cd_dir(),  fop.grad_cd_dir() );
        A_bd_dir.upload(  fop.rows_bd_dir(),  fop.cols_bd_dir(),  fop.grad_bd_dir() );
        A_avg_dir.upload( fop.rows_avg_dir(), fop.cols_avg_dir(), fop.values_avg_dir() );
      }

      auto A=[&](int m){ return device_alloc<real_t>((size_t)m); };
      d_h=A(Vn);d_g=A(Vn);d_h2=A(Vn);d_g2=A(Vn);
      d_c=A(n);d_p=A(n);d_rho=A(n);d_mu=A(n);d_ux=A(n);d_uy=A(n);d_uz=A(n);d_lapc=A(n);d_lapmu=A(n);
      d_gcc_x=A(n);d_gcc_y=A(n);d_gcc_z=A(n);d_gcb_x=A(n);d_gcb_y=A(n);d_gcb_z=A(n);d_gcm_x=A(n);d_gcm_y=A(n);d_gcm_z=A(n);
      d_gpc_x=A(n);d_gpc_y=A(n);d_gpc_z=A(n);d_gpb_x=A(n);d_gpb_y=A(n);d_gpb_z=A(n);d_gpm_x=A(n);d_gpm_y=A(n);d_gpm_z=A(n);
      d_relax=A(Vn);
      // dir-derivative temporaries: not needed when fused (recomputed in registers)
      if( !fused ){ d_gc_cd=A(Vn);d_gp_cd=A(Vn);d_gc_bd=A(Vn);d_gp_bd=A(Vn);d_avg=A(Vn); }
      // eq/force/collision temporaries: not needed when the whole collision is fused
      if( !fuse_coll ){ d_eqh=A(Vn);d_eqg=A(Vn);d_collh=A(Vn);d_collg=A(Vn);d_fh=A(Vn);d_fg=A(Vn); }
      d_xtnd=A(2*n);
      d_reduce=device_alloc<double>(2);
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

      use_mrt      = s.use_mrt();
      correct_mass = s.correct_op_mass();
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

    // central-difference vector gradient: matrix-free or stored-CSR SpMV
    void grad_cd( real_t const * f, real_t * gx, real_t * gy, real_t * gz )
    {
      if( mf_grad ){
        k_grad_cd_mf<<<grid_1d(n,BLOCK),BLOCK>>>( n, P.alpha, d_cdm, d_cdp, f, gx,gy,gz );
        GPU_CHECK_KERNEL();
      } else {
        spmv3( A_grad_cd, f, gx,gy,gz );
      }
    }

    // biased vector gradient: matrix-free or stored-CSR SpMV
    void grad_bd( real_t const * f, real_t * gx, real_t * gy, real_t * gz )
    {
      if( mf_grad ){
        k_grad_bd_mf<<<grid_1d(n,BLOCK),BLOCK>>>( n, P.alpha, d_bcase, d_bc1, d_bc2, f, gx,gy,gz );
        GPU_CHECK_KERNEL();
      } else {
        spmv3( A_grad_bd, f, gx,gy,gz );
      }
    }

    // per-direction central/biased derivatives -> Q*n buffer (dir 0 = 0)
    void dir_cd( real_t const * f, real_t * out )
    {
      if( mf_grad ){
        k_cd_dir_mf<<<grid_1d(n,BLOCK),BLOCK>>>( n, d_cdm, d_cdp, f, out ); GPU_CHECK_KERNEL();
      } else {
        GPU_CHECK( cudaMemset( out, 0, (size_t)Vn*sizeof(real_t) ) );
        spmv( A_cd_dir, f, out+n );
      }
    }
    void dir_bd( real_t const * f, real_t * out )
    {
      if( mf_grad ){
        k_bd_dir_mf<<<grid_1d(n,BLOCK),BLOCK>>>( n, d_bcase, d_bc1, d_bc2, f, out ); GPU_CHECK_KERNEL();
      } else {
        GPU_CHECK( cudaMemset( out, 0, (size_t)Vn*sizeof(real_t) ) );
        spmv( A_bd_dir, f, out+n );
      }
    }

    // node-centred Laplacian (2n input) and directional average: mf or CSR
    void laplacian( real_t const * xtnd, real_t * out )
    {
      if( mf_grad ){
        k_lap_mf<<<grid_1d(n,BLOCK),BLOCK>>>( n, P.alpha, xtnd, d_lcase, d_lc1, d_lc2, out ); GPU_CHECK_KERNEL();
      } else {
        spmv( A_lap, xtnd, out );
      }
    }
    void avg_dir( real_t const * x, real_t * out )
    {
      if( mf_grad ){
        k_avg_dir_mf<<<grid_1d(n,BLOCK),BLOCK>>>( n, d_avgnext, x, out ); GPU_CHECK_KERNEL();
      } else {
        spmv( A_avg_dir, x, out );
      }
    }

    // ---- one time step (mirrors TimeStepperMultiPhase::do_time_step) ----------
    void step()
    {
      dim3 gN=grid_1d(n,BLOCK), gVn=grid_1d(Vn,BLOCK);

      k_moments<<<gN,BLOCK>>>( P, d_h,d_g, d_c,d_p,d_rho,d_mu, d_ux,d_uy,d_uz, d_relax ); GPU_CHECK_KERNEL();
      grad_cd( d_c, d_gcc_x,d_gcc_y,d_gcc_z );
      grad_bd( d_c, d_gcb_x,d_gcb_y,d_gcb_z );
      k_grad_md<<<gN,BLOCK>>>( n, d_gcc_x,d_gcc_y,d_gcc_z, d_gcb_x,d_gcb_y,d_gcb_z, d_gcm_x,d_gcm_y,d_gcm_z ); GPU_CHECK_KERNEL();
      // Laplacian(c): pack [c, bnd_cond] (2n) then SpMV (node-centred, wetting BC)
      k_pack_lap_c<<<gN,BLOCK>>>( P, d_c, d_xtnd ); GPU_CHECK_KERNEL();
      laplacian( d_xtnd, d_lapc );
      k_mu_axpy<<<gN,BLOCK>>>( n, P.kappa, d_lapc, d_mu ); GPU_CHECK_KERNEL();
      // Laplacian(mu): boundary term is zero -> pack [mu, 0]
      k_pack_lap_zero<<<gN,BLOCK>>>( n, d_mu, d_xtnd ); GPU_CHECK_KERNEL();
      laplacian( d_xtnd, d_lapmu );
      k_vel_press_corr<<<gN,BLOCK>>>( P, d_mu,d_rho, d_gcc_x,d_gcc_y,d_gcc_z, d_ux,d_uy,d_uz, d_p ); GPU_CHECK_KERNEL();
      grad_cd( d_p, d_gpc_x,d_gpc_y,d_gpc_z );
      grad_bd( d_p, d_gpb_x,d_gpb_y,d_gpb_z );
      k_grad_md<<<gN,BLOCK>>>( n, d_gpc_x,d_gpc_y,d_gpc_z, d_gpb_x,d_gpb_y,d_gpb_z, d_gpm_x,d_gpm_y,d_gpm_z ); GPU_CHECK_KERNEL();

      if( fuse_coll )
      {
        // one per-site kernel: equilibria + force + collision + apply (no Q*n temporaries)
        if( use_mrt )
          k_collide_fused_mrt<<<gN,BLOCK>>>( P, d_stream, d_c,d_rho,d_p,d_mu, d_ux,d_uy,d_uz,
              d_gpc_x,d_gpc_y,d_gpc_z, d_gcc_x,d_gcc_y,d_gcc_z,
              d_gpm_x,d_gpm_y,d_gpm_z, d_gcm_x,d_gcm_y,d_gcm_z,
              d_lapmu, d_relax, d_cdm,d_cdp, d_bcase,d_bc1,d_bc2, d_avgnext, d_h,d_g );
        else
          k_collide_fused_bgk<<<gN,BLOCK>>>( P, d_solid,d_stream, d_c,d_rho,d_p,d_mu, d_ux,d_uy,d_uz,
              d_gpc_x,d_gpc_y,d_gpc_z, d_gcc_x,d_gcc_y,d_gcc_z,
              d_gpm_x,d_gpm_y,d_gpm_z, d_gcm_x,d_gcm_y,d_gcm_z,
              d_lapmu, d_relax, d_cdm,d_cdp, d_bcase,d_bc1,d_bc2, d_avgnext, d_h,d_g );
        GPU_CHECK_KERNEL();
      }
      else {
      if( !fused )
      {
        dir_cd( d_c, d_gc_cd );
        dir_cd( d_p, d_gp_cd );
        dir_bd( d_c, d_gc_bd );
        dir_bd( d_p, d_gp_bd );
        avg_dir( d_lapmu, d_avg );
        k_force_term<<<gN,BLOCK>>>( P, d_stream, d_c,d_rho,d_mu, d_ux,d_uy,d_uz,
                                    d_gpm_x,d_gpm_y,d_gpm_z, d_gcm_x,d_gcm_y,d_gcm_z,
                                    d_gc_cd,d_gc_bd, d_gp_cd,d_gp_bd, d_avg, d_fh,d_fg ); GPU_CHECK_KERNEL();
      }
      else
      {
        k_force_term_fused<<<gN,BLOCK>>>( P, d_stream, d_c,d_rho,d_p,d_mu, d_ux,d_uy,d_uz,
                                    d_gpm_x,d_gpm_y,d_gpm_z, d_gcm_x,d_gcm_y,d_gcm_z,
                                    d_lapmu, d_cdm,d_cdp, d_bcase,d_bc1,d_bc2, d_avgnext,
                                    d_fh,d_fg ); GPU_CHECK_KERNEL();
      }

      if( use_mrt )
      {
        if( !fused )
          k_equilibria_mrt<<<gN,BLOCK>>>( P, d_stream, d_c,d_rho,d_p,d_mu, d_ux,d_uy,d_uz,
                                          d_gpc_x,d_gpc_y,d_gpc_z, d_gcc_x,d_gcc_y,d_gcc_z,
                                          d_gc_cd,d_gp_cd, d_eqh,d_eqg );
        else
          k_equilibria_mrt_fused<<<gN,BLOCK>>>( P, d_stream, d_c,d_rho,d_p,d_mu, d_ux,d_uy,d_uz,
                                          d_gpc_x,d_gpc_y,d_gpc_z, d_gcc_x,d_gcc_y,d_gcc_z,
                                          d_cdm,d_cdp, d_eqh,d_eqg );
        GPU_CHECK_KERNEL();
        k_collision_term_raw<<<gVn,BLOCK>>>( Vn, d_eqh,d_eqg, d_h,d_g, d_collh,d_collg ); GPU_CHECK_KERNEL();
        k_mrt_relax_g<<<gN,BLOCK>>>( P, d_relax, d_collg ); GPU_CHECK_KERNEL();
      }
      else
      {
        if( !fused )
          k_equilibria<<<gN,BLOCK>>>( P, d_solid,d_stream, d_c,d_rho,d_p,d_mu, d_ux,d_uy,d_uz,
                                      d_gpc_x,d_gpc_y,d_gpc_z, d_gcc_x,d_gcc_y,d_gcc_z,
                                      d_gc_cd,d_gp_cd, d_eqh,d_eqg );
        else
          k_equilibria_fused<<<gN,BLOCK>>>( P, d_solid,d_stream, d_c,d_rho,d_p,d_mu, d_ux,d_uy,d_uz,
                                      d_gpc_x,d_gpc_y,d_gpc_z, d_gcc_x,d_gcc_y,d_gcc_z,
                                      d_cdm,d_cdp, d_eqh,d_eqg );
        GPU_CHECK_KERNEL();
        k_collision_term<<<gVn,BLOCK>>>( Vn, d_eqh,d_eqg, d_h,d_g, d_relax, d_collh,d_collg ); GPU_CHECK_KERNEL();
      }
      k_collide_apply<<<gVn,BLOCK>>>( Vn, d_collh,d_collg, d_fh,d_fg, d_h,d_g ); GPU_CHECK_KERNEL();
      } // end else (non-fused-collision path)

      if( mf_stream )
      {
        k_stream_gather<<<gVn,BLOCK>>>( Vn, n, d_src, d_h, d_h2 ); GPU_CHECK_KERNEL();
        k_stream_gather<<<gVn,BLOCK>>>( Vn, n, d_src, d_g, d_g2 ); GPU_CHECK_KERNEL();
      }
      else
      {
        spmv( A_stream, d_h, d_h2 );
        spmv( A_stream, d_g, d_g2 );
      }
      std::swap(d_h,d_h2); std::swap(d_g,d_g2);
    }

    // ---- order-parameter mass conservation (CPU MassConservationCorrector) -----
    void reduce_mass_weight( double & M, double & W )
    {
      double zero[2] = {0.0,0.0};
      copy_h2d( d_reduce, zero, 2 );
      k_mass_weight<<< grid_1d(n,BLOCK), BLOCK >>>( P, d_stream, d_h, d_reduce ); GPU_CHECK_KERNEL();
      double h2[2]; copy_d2h( h2, d_reduce, 2 ); M=h2[0]; W=h2[1];
    }
    /// record M0 = current total mass (call once, after upload_state)
    void record_target_mass(){ double M,W; reduce_mass_weight(M,W); target_mass=M; }
    /// current total order-parameter mass (diagnostic; independent of correct_mass)
    double current_mass(){ double M,W; reduce_mass_weight(M,W); return M; }
    /// remove the accumulated drift (once per step); returns the drift that was removed
    double apply_mass_correction()
    {
      if( !correct_mass ) return 0.0;
      double M,W; reduce_mass_weight(M,W);
      double const drift = M - target_mass;
      if( W>0.0 && drift!=0.0 )
      {
        double const lambda = drift / W;
        k_inject_mass<<< grid_1d(n,BLOCK), BLOCK >>>( P, d_stream, lambda, d_h ); GPU_CHECK_KERNEL();
      }
      return drift;
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
      grad_cd( d_c, d_gcc_x,d_gcc_y,d_gcc_z );
      k_pack_lap_c<<<gN,BLOCK>>>( P, d_c, d_xtnd ); GPU_CHECK_KERNEL();
      laplacian( d_xtnd, d_lapc );
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
      device_free(d_reduce);
      device_free(d_src);
      device_free(d_cdm); device_free(d_cdp);
      device_free(d_bcase); device_free(d_bc1); device_free(d_bc2);
      device_free(d_avgnext); device_free(d_lcase); device_free(d_lc1); device_free(d_lc2);
      device_free(d_solid); device_free(d_stream);
    }
  };

} // namespace felbm_gpu

#endif // FELBM_GPU_MULTIPHASE_GPU_CUH
