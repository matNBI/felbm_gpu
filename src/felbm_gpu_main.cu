//=============================================================================
//  felbm_gpu — CUDA multiphase (Lee-Liu) solver driver.
//
//  Reuses the felbm_local HOST code for config parsing, geometry/TIFF init, the
//  multiphase initial condition, and the sparse stencil operators; replaces the
//  compute engine with the CUDA kernels in device_engine.cuh.  Single GPU, one
//  subdomain (num_subdomains forced to 1).
//
//  Run:   ./felbm_gpu settings.cfg     (model = multi_phase, use_open_bnd = false)
//=============================================================================

// ---- reused felbm_local host code (compiled as host by nvcc) ----------------
#include <lbm.h>
#include <util.h>

// ---- felbm_gpu device code --------------------------------------------------
#include <felbm_gpu/device_engine.cuh>
#include <felbm_gpu/operator_access.h>

#include <H5Cpp.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

using namespace lbm;
using namespace felbm_gpu;

// ---------- minimal HDF5 dump of the compressed macroscopic fields -----------
static void write_fields_h5( std::string const & path, int n,
                             std::vector<double> const& rho, std::vector<double> const& c,
                             std::vector<double> const& ux,  std::vector<double> const& uy,
                             std::vector<double> const& uz,  std::vector<double> const& p )
{
  H5::H5File f( path, H5F_ACC_TRUNC );
  hsize_t d[1] = { (hsize_t)n };
  auto put = [&]( char const* name, std::vector<double> const& v ){
    H5::DataSpace sp( 1, d );
    H5::DataSet ds = f.createDataSet( name, H5::PredType::NATIVE_DOUBLE, sp );
    ds.write( v.data(), H5::PredType::NATIVE_DOUBLE );
  };
  put("density",rho); put("concentration",c);
  put("u_x",ux); put("u_y",uy); put("u_z",uz); put("pressure",p);
}

int main( int argc, char** argv )
{
  util::ConfigFile cfg;
  cfg.load( argc>=2 ? argv[1] : "./settings.cfg" );

  Settings settings = make_settings( cfg );
  settings.num_subdomains() = 1u;                    // single GPU: one subdomain

  util::LogInfo::on()       = settings.logging();
  util::LogInfo::console()  = settings.console();
  util::LogInfo::filename() = settings.output_dir() + settings.log_filename();

  if( settings.model().find("single_phase") != std::string::npos ){
    std::cerr << "felbm_gpu: this build is the MULTI-phase GPU solver; model=single_phase not supported.\n";
    return 1;
  }
  if( settings.use_open_bnd() )
    std::cerr << "felbm_gpu: WARNING — open boundaries are not ported yet; running body-force/periodic only.\n";
  if( settings.use_mrt() )
    std::cerr << "felbm_gpu: WARNING — MRT not ported yet; using BGK collision.\n";

  VelocitySetD3Q19 vs;
  Domain domain = make_domain( settings, vs );

  // geometry (duct / 3d_image / cylinders_repulsive / spheres_repulsive / ...)
  {
    DomainInitializer * di = NULL;
    std::string const & geom = settings.domain_geometry();
    if(      geom.find("3d_image")            != std::string::npos ) di = new DomainInitializer_3DImage(            settings, vs );
    else if( geom.find("cylinders_repulsive") != std::string::npos ) di = new DomainInitializer_CylindersRepulsive( settings, vs );
    else if( geom.find("spheres_repulsive")   != std::string::npos ) di = new DomainInitializer_SpheresRepulsive(   settings, vs );
    else if( geom.find("duct")                != std::string::npos ) di = new DomainInitializer_Duct(               settings, vs );
    else                                                             di = new DomainInitializer_Empty(              settings, vs );
    di->initialize( domain );
    delete di;
  }

  DomainManager dm = make_subdomains( domain, settings );
  SubDomain const & sd = dm.subdomain( 0u );

  ParametersMultiPhase param = make_parameters<ParametersMultiPhase>( cfg );

  int const n  = (int)sd.size_sites();
  int const V  = (int)vs.size();
  int const Vn = V*n;

  // ---- initial condition on the host, then upload -----------------------------
  DiscreteDistribution h( sd.size_sites(), vs, 0. );
  DiscreteDistribution g( sd.size_sites(), vs, 0. );
  {
    InitializerMultiPhase * in = NULL;
    std::string const & f = settings.fluid_initializer();
    if(      f.find("uniform")          != std::string::npos ) in = new InitializerMultiPhase_Uniform(         param, vs, sd, settings );
    else if( f.find("random")           != std::string::npos ) in = new InitializerMultiPhase_Random(          param, vs, sd, settings );
    else if( f.find("single_interface") != std::string::npos ) in = new InitializerMultiPhase_SingleInterface( param, vs, sd, settings );
    else if( f.find("sinestripe")       != std::string::npos ) in = new InitializerMultiPhase_SineStripe(      param, vs, sd, settings );
    else if( f.find("spherical_droplet")!= std::string::npos ) in = new InitializerMultiPhase_SphericalDroplet(param, vs, sd, settings );
    else if( f.find("droplet_pipe")     != std::string::npos ) in = new InitializerMultiPhase_DropletPipe(     param, vs, sd, settings );
    else                                                       in = new InitializerMultiPhase_Uniform(         param, vs, sd, settings );
    in->initialize( h, g );
    delete in;
  }

  // ---- build the CPU sparse operators, upload as device CSR -------------------
  //--- (subclasses that expose the protected CSR members; no felbm_local edit needed)
  StreamingOperatorGPU stream_op( sd, vs, settings );
  FieldOperatorGPU     fop( sd, vs, settings );

  upload_d3q19_constants();

  DeviceCSR  A_stream, A_lap, A_cd_dir, A_bd_dir, A_avg_dir;
  DeviceCSR3 A_grad_cd, A_grad_bd;
  A_stream.upload(  stream_op.rows(), stream_op.cols(), stream_op.values() );
  A_grad_cd.upload( fop.rows_cd(), fop.cols_cd(), fop.grad_cd_x(), fop.grad_cd_y(), fop.grad_cd_z() );
  A_grad_bd.upload( fop.rows_bd(), fop.cols_bd(), fop.grad_bd_x(), fop.grad_bd_y(), fop.grad_bd_z() );
  A_lap.upload(     fop.rows_lap(), fop.cols_lap(), fop.laplacian() );
  A_cd_dir.upload(  fop.rows_cd_dir(),  fop.cols_cd_dir(),  fop.grad_cd_dir() );
  A_bd_dir.upload(  fop.rows_bd_dir(),  fop.cols_bd_dir(),  fop.grad_bd_dir() );
  A_avg_dir.upload( fop.rows_avg_dir(), fop.cols_avg_dir(), fop.values_avg_dir() );

  // ---- device state ----------------------------------------------------------
  auto dalloc = [&](int m){ return device_alloc<real_t>( (size_t)m ); };
  real_t *d_h=dalloc(Vn), *d_g=dalloc(Vn), *d_h2=dalloc(Vn), *d_g2=dalloc(Vn);
  real_t *d_c=dalloc(n),*d_p=dalloc(n),*d_rho=dalloc(n),*d_mu=dalloc(n);
  real_t *d_ux=dalloc(n),*d_uy=dalloc(n),*d_uz=dalloc(n),*d_lapc=dalloc(n),*d_lapmu=dalloc(n);
  real_t *d_gcc_x=dalloc(n),*d_gcc_y=dalloc(n),*d_gcc_z=dalloc(n);
  real_t *d_gcb_x=dalloc(n),*d_gcb_y=dalloc(n),*d_gcb_z=dalloc(n);
  real_t *d_gcm_x=dalloc(n),*d_gcm_y=dalloc(n),*d_gcm_z=dalloc(n);
  real_t *d_gpc_x=dalloc(n),*d_gpc_y=dalloc(n),*d_gpc_z=dalloc(n);
  real_t *d_gpb_x=dalloc(n),*d_gpb_y=dalloc(n),*d_gpb_z=dalloc(n);
  real_t *d_gpm_x=dalloc(n),*d_gpm_y=dalloc(n),*d_gpm_z=dalloc(n);
  real_t *d_relax=dalloc(Vn),*d_gc_cd=dalloc(Vn),*d_gp_cd=dalloc(Vn),*d_gc_bd=dalloc(Vn),*d_gp_bd=dalloc(Vn);
  real_t *d_avg=dalloc(Vn),*d_eqh=dalloc(Vn),*d_eqg=dalloc(Vn),*d_collh=dalloc(Vn),*d_collg=dalloc(Vn),*d_fh=dalloc(Vn),*d_fg=dalloc(Vn);
  unsigned char *d_solid=device_alloc<unsigned char>(n), *d_stream=device_alloc<unsigned char>(n);

  // upload distributions + masks (cast double -> real_t)
  { std::vector<real_t> tmp(Vn);
    for(int j=0;j<Vn;++j) tmp[j]=(real_t)h.data()[j]; copy_h2d(d_h,tmp.data(),Vn);
    for(int j=0;j<Vn;++j) tmp[j]=(real_t)g.data()[j]; copy_h2d(d_g,tmp.data(),Vn); }
  { std::vector<unsigned char> s(n),st(n);
    for(int i=0;i<n;++i){ s[i]=sd.is_solid(i)?1:0; st[i]=sd.is_streamed(i)?1:0; }
    copy_h2d(d_solid,s.data(),n); copy_h2d(d_stream,st.data(),n); }

  // ---- parameters block ------------------------------------------------------
  DevParams P;
  P.n=n;
  P.rho0=(real_t)param.phase_density(0u); P.rho1=(real_t)param.phase_density(1u); P.drho=P.rho0-P.rho1;
  P.kappa=(real_t)param.kappa(); P.beta=(real_t)param.beta(); P.four_beta=(real_t)(4.0*param.beta());
  P.bnd_coeff=(real_t)param.boundary_coefficient();
  P.tau0=(real_t)param.relaxation_time(0u); P.tau1=(real_t)param.relaxation_time(1u);
  P.cs2=(real_t)CS2; P.alpha=(real_t)ALPHA; P.gamma_c=(real_t)GAMMA; P.beta_c=(real_t)BETA;
  P.mobility=(real_t)param.mobility(); P.forcing_factor=(real_t)param.forcing_factor();
  Vector3d gv=settings.acceleration(), fv=settings.forcing();
  P.gx=(real_t)gv[0u]; P.gy=(real_t)gv[1u]; P.gz=(real_t)gv[2u];
  P.fx=(real_t)fv[0u]; P.fy=(real_t)fv[1u]; P.fz=(real_t)fv[2u];

  dim3 gN=grid_1d(n,BLOCK), gVn=grid_1d(Vn,BLOCK);

  // ---- time loop -------------------------------------------------------------
  unsigned int const steps = settings.max_iterations();
  unsigned int const fskip = settings.file_skip()?settings.file_skip():1u;

  std::cout << "felbm_gpu: n_sites="<<n<<"  Q="<<V<<"  steps="<<steps
            << "  precision="<<(FELBM_REAL_IS_DOUBLE?"double":"float")<<"\n";

  for( unsigned int t=0; t<=steps; ++t )
  {
    // --- compute hydrodynamic fields ---
    k_moments<<<gN,BLOCK>>>( P, d_h,d_g, d_c,d_p,d_rho,d_mu, d_ux,d_uy,d_uz, d_relax ); GPU_CHECK_KERNEL();
    spmv3( A_grad_cd, d_c, d_gcc_x,d_gcc_y,d_gcc_z );
    spmv3( A_grad_bd, d_c, d_gcb_x,d_gcb_y,d_gcb_z );
    k_grad_md<<<gN,BLOCK>>>( n, d_gcc_x,d_gcc_y,d_gcc_z, d_gcb_x,d_gcb_y,d_gcb_z, d_gcm_x,d_gcm_y,d_gcm_z ); GPU_CHECK_KERNEL();
    spmv( A_lap, d_c, d_lapc );
    k_mu_axpy<<<gN,BLOCK>>>( n, P.kappa, d_lapc, d_mu ); GPU_CHECK_KERNEL();
    spmv( A_lap, d_mu, d_lapmu );
    k_vel_press_corr<<<gN,BLOCK>>>( P, d_mu,d_rho, d_gcc_x,d_gcc_y,d_gcc_z, d_ux,d_uy,d_uz, d_p ); GPU_CHECK_KERNEL();
    spmv3( A_grad_cd, d_p, d_gpc_x,d_gpc_y,d_gpc_z );
    spmv3( A_grad_bd, d_p, d_gpb_x,d_gpb_y,d_gpb_z );
    k_grad_md<<<gN,BLOCK>>>( n, d_gpc_x,d_gpc_y,d_gpc_z, d_gpb_x,d_gpb_y,d_gpb_z, d_gpm_x,d_gpm_y,d_gpm_z ); GPU_CHECK_KERNEL();
    GPU_CHECK( cudaMemset(d_gc_cd,0,(size_t)Vn*sizeof(real_t)) ); spmv( A_cd_dir, d_c, d_gc_cd+n );
    GPU_CHECK( cudaMemset(d_gp_cd,0,(size_t)Vn*sizeof(real_t)) ); spmv( A_cd_dir, d_p, d_gp_cd+n );

    // --- body forces ---
    GPU_CHECK( cudaMemset(d_gc_bd,0,(size_t)Vn*sizeof(real_t)) ); spmv( A_bd_dir, d_c, d_gc_bd+n );
    GPU_CHECK( cudaMemset(d_gp_bd,0,(size_t)Vn*sizeof(real_t)) ); spmv( A_bd_dir, d_p, d_gp_bd+n );
    spmv( A_avg_dir, d_lapmu, d_avg );
    k_force_term<<<gN,BLOCK>>>( P, d_stream, d_c,d_rho,d_mu, d_ux,d_uy,d_uz,
                                d_gpm_x,d_gpm_y,d_gpm_z, d_gcm_x,d_gcm_y,d_gcm_z,
                                d_gc_cd,d_gc_bd, d_gp_cd,d_gp_bd, d_avg, d_fh,d_fg ); GPU_CHECK_KERNEL();

    // --- collide ---
    k_equilibria<<<gN,BLOCK>>>( P, d_solid,d_stream, d_c,d_rho,d_p,d_mu, d_ux,d_uy,d_uz,
                                d_gpc_x,d_gpc_y,d_gpc_z, d_gcc_x,d_gcc_y,d_gcc_z,
                                d_gc_cd,d_gp_cd, d_eqh,d_eqg ); GPU_CHECK_KERNEL();
    k_collision_term<<<gVn,BLOCK>>>( Vn, d_eqh,d_eqg, d_h,d_g, d_relax, d_collh,d_collg ); GPU_CHECK_KERNEL();
    k_collide_apply<<<gVn,BLOCK>>>( Vn, d_collh,d_collg, d_fh,d_fg, d_h,d_g ); GPU_CHECK_KERNEL();

    // --- stream ---
    spmv( A_stream, d_h, d_h2 );
    spmv( A_stream, d_g, d_g2 );
    std::swap(d_h,d_h2); std::swap(d_g,d_g2);

    // --- output ---
    if( t % fskip == 0u )
    {
      std::vector<double> hc(n),hrho(n),hux(n),huy(n),huz(n),hp(n);
      std::vector<real_t> tc(n);
      auto grab=[&](real_t* dptr, std::vector<double>& out){ copy_d2h(tc.data(),dptr,n); for(int i=0;i<n;++i) out[i]=(double)tc[i]; };
      grab(d_c,hc); grab(d_rho,hrho); grab(d_ux,hux); grab(d_uy,huy); grab(d_uz,huz); grab(d_p,hp);
      std::ostringstream fn; fn<<settings.output_dir()<<settings.output_name()<<"_"<<t<<".h5";
      write_fields_h5( fn.str(), n, hrho,hc,hux,huy,huz,hp );
      std::cout << "  step "<<t<<"  wrote "<<fn.str()<<"\n";
    }
  }

  A_stream.free(); A_lap.free(); A_cd_dir.free(); A_bd_dir.free(); A_avg_dir.free();
  A_grad_cd.free(); A_grad_bd.free();
  std::cout << "felbm_gpu: done.\n";
  return 0;
}
