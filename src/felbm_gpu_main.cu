//=============================================================================
//  felbm_gpu — CUDA multiphase (Lee-Liu) solver driver.
//
//  Reuses the felbm_local HOST code for config parsing, geometry/TIFF init and
//  the multiphase initial condition; the compute engine is MultiPhaseGPU
//  (include/felbm_gpu/multiphase_gpu.cuh), shared with the validation harness.
//  Single GPU, one subdomain (num_subdomains forced to 1).
//
//  Run:   ./felbm_gpu settings.cfg     (model = multi_phase, use_open_bnd = false)
//=============================================================================

#include <lbm.h>
#include <util.h>

#include <felbm_gpu/multiphase_gpu.cuh>

#include <lbm_particle_manager.h>

#include <H5Cpp.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <cctype>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <chrono>

using namespace lbm;
using namespace felbm_gpu;

// ---------- HDF5 dump of the compressed macroscopic fields -------------------
//  deflate = gzip level (0 = off, 1..9), f32 = write float32 instead of float64.
//  Both are transparent to readers (ParaView / h5py handle gzip + dtype). float32
//  halves the raw size; gzip then shrinks the smooth fields a further ~2-3x.
static void write_fields_h5( std::string const & path, int n,
                             std::vector<double> const& rho, std::vector<double> const& c,
                             std::vector<double> const& ux,  std::vector<double> const& uy,
                             std::vector<double> const& uz,  std::vector<double> const& p,
                             int deflate, bool f32 )
{
  H5::H5File f( path, H5F_ACC_TRUNC );
  hsize_t d[1] = { (hsize_t)n };

  H5::DSetCreatPropList plist;
  if( deflate > 0 && n > 0 )
  {
    hsize_t chunk[1] = { (hsize_t)n };   // one chunk per dataset (gzip needs chunking)
    plist.setChunk( 1, chunk );
    plist.setDeflate( (unsigned)deflate );
  }

  std::vector<float> tmp;
  if( f32 ) tmp.resize( (size_t)n );

  auto put = [&]( char const* name, std::vector<double> const& v ){
    H5::DataSpace sp( 1, d );
    if( f32 ){
      for( int i=0;i<n;++i ) tmp[i] = (float)v[i];
      H5::DataSet ds = f.createDataSet( name, H5::PredType::NATIVE_FLOAT, sp, plist );
      ds.write( tmp.data(), H5::PredType::NATIVE_FLOAT );
    } else {
      H5::DataSet ds = f.createDataSet( name, H5::PredType::NATIVE_DOUBLE, sp, plist );
      ds.write( v.data(), H5::PredType::NATIVE_DOUBLE );
    }
  };
  put("density",rho); put("concentration",c);
  put("u_x",ux); put("u_y",uy); put("u_z",uz); put("pressure",p);
}

// ---------- one-time geometry: fluid-site grid coordinates (for XDMF) ----------
// The field dumps are compressed fluid-only 1D arrays with no spatial info. This
// writes coords[i] = (x,y,z) of compressed site i (same index order as the fields)
// plus the domain bounding size, so make_xdmf.py can build a ParaView point cloud.
static void write_geometry_h5( std::string const & path, SubDomain const & sd, int deflate )
{
  int const n = (int)sd.size_sites();
  std::vector<int> coords( 3*n );
  int mx=0,my=0,mz=0;
  for( int i=0;i<n;++i ){
    unsigned int x,y,z; sd.idx_to_coords( (unsigned int)i, x,y,z );
    coords[3*i]=(int)x; coords[3*i+1]=(int)y; coords[3*i+2]=(int)z;
    if((int)x>mx)mx=(int)x; if((int)y>my)my=(int)y; if((int)z>mz)mz=(int)z;
  }
  H5::H5File f( path, H5F_ACC_TRUNC );
  { hsize_t d[2]={(hsize_t)n,3}; H5::DataSpace sp(2,d);
    H5::DSetCreatPropList plist;
    if( deflate>0 && n>0 ){ hsize_t chunk[2]={(hsize_t)n,3}; plist.setChunk(2,chunk); plist.setDeflate((unsigned)deflate); }
    H5::DataSet ds=f.createDataSet("coords", H5::PredType::NATIVE_INT, sp, plist);
    ds.write( coords.data(), H5::PredType::NATIVE_INT ); }
  { int dims[3]={mx+1,my+1,mz+1}; hsize_t d[1]={3}; H5::DataSpace sp(1,d);
    H5::DataSet ds=f.createDataSet("size", H5::PredType::NATIVE_INT, sp);
    ds.write( dims, H5::PredType::NATIVE_INT ); }
}

int main( int argc, char** argv )
{
  util::ConfigFile cfg;
  cfg.load( argc>=2 ? argv[1] : "./settings.cfg" );

  Settings settings = make_settings( cfg );
  settings.num_subdomains() = 1u;

  // Output HDF5 options (felbm_gpu-only cfg keys, read directly).
  //   output_deflate : gzip level 0-9. DEFAULT 0 (OFF) — ParaView's Xdmf3 reader can
  //                    crash reading chunked+gzipped datasets through XDMF, so keep
  //                    the default viz-ready. Enable for disk savings, but decompress
  //                    (h5repack -f NONE) before loading into ParaView.
  //   output_float32 : write float32 instead of float64 (opt-in; XDMF-safe).
  int  const output_deflate = cfg.exist("output_deflate")
                            ? util::to_value<int>( cfg.get_value("output_deflate") ) : 0;
  bool const output_f32     = cfg.exist("output_float32")
                            ? util::to_value<bool>( cfg.get_value("output_float32") ) : false;

  // Prototypes: matrix-free streaming and central-gradient instead of stored CSR SpMV.
  bool const stream_mf      = cfg.exist("stream_matrix_free")
                            ? util::to_value<bool>( cfg.get_value("stream_matrix_free") ) : false;
  bool const grad_mf        = cfg.exist("grad_matrix_free")
                            ? util::to_value<bool>( cfg.get_value("grad_matrix_free") ) : false;

  // GPU selection on a multi-GPU box. `gpu_device = N` picks device N; -1 (default)
  // leaves it to the driver / CUDA_VISIBLE_DEVICES. Must be set before any CUDA use.
  int const gpu_device = cfg.exist("gpu_device")
                       ? util::to_value<int>( cfg.get_value("gpu_device") ) : -1;
  if( gpu_device >= 0 ) GPU_CHECK( cudaSetDevice( gpu_device ) );
  { int dev=-1; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop,dev);
    std::cout << "felbm_gpu: using GPU " << dev << " (" << prop.name << ", "
              << (prop.totalGlobalMem>>20) << " MB)\n"; }

  util::LogInfo::on()       = settings.logging();
  util::LogInfo::console()  = settings.console();
  util::LogInfo::filename() = settings.output_dir() + settings.log_filename();

  if( settings.model().find("single_phase") != std::string::npos ){
    std::cerr << "felbm_gpu: this build is the MULTI-phase GPU solver; model=single_phase not supported.\n";
    return 1;
  }
  if( settings.use_open_bnd() )
    std::cerr << "felbm_gpu: WARNING — open boundaries not ported yet; body-force/periodic only.\n";
  if( settings.use_mrt() )
    std::cerr << "felbm_gpu: MRT collision enabled.\n";

  VelocitySetD3Q19 vs;
  Domain domain = make_domain( settings, vs );

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

  // one-time geometry for ParaView/XDMF (honours the output_xdmf flag)
  if( settings.output_xdmf() )
    write_geometry_h5( settings.output_dir()+"geometry.h5", sd, output_deflate );

  // The multiphase parameters (tau0/tau1, rho0/rho1, interface_width,
  // surface_tension, phi, mobility_coeff, density_solid) live in the SEPARATE
  // file named by `param_file` — load it and hand THAT to make_parameters (the
  // CPU scheduler does the same). Passing the main settings.cfg would leave every
  // parameter at zero.
  util::ConfigFile param_cfg;
  param_cfg.load( settings.param_file() );
  ParametersMultiPhase param = make_parameters<ParametersMultiPhase>( param_cfg );
  param.forcing_factor() = 1.0;   // sane default; overwritten per step below

  int const n = (int)sd.size_sites();

  // ---- initial condition on the host ----
  DiscreteDistribution h( sd.size_sites(), vs, 0. );
  DiscreteDistribution g( sd.size_sites(), vs, 0. );
  {
    InitializerMultiPhase * in = NULL;
    std::string const & f = settings.fluid_initializer();
    char const * chosen = "uniform (fallback)";
    if(      f.find("uniform")          != std::string::npos ){ in = new InitializerMultiPhase_Uniform(         param, vs, sd, settings ); chosen="uniform"; }
    else if( f.find("random")           != std::string::npos ){ in = new InitializerMultiPhase_Random(          param, vs, sd, settings ); chosen="random"; }
    else if( f.find("single_interface") != std::string::npos ){ in = new InitializerMultiPhase_SingleInterface( param, vs, sd, settings ); chosen="single_interface"; }
    else if( f.find("sinestripe")       != std::string::npos ){ in = new InitializerMultiPhase_SineStripe(      param, vs, sd, settings ); chosen="sinestripe"; }
    else if( f.find("spherical_droplet")!= std::string::npos ){ in = new InitializerMultiPhase_SphericalDroplet(param, vs, sd, settings ); chosen="spherical_droplet"; }
    else if( f.find("droplet_pipe")     != std::string::npos ){ in = new InitializerMultiPhase_DropletPipe(     param, vs, sd, settings ); chosen="droplet_pipe"; }
    else{                                                        in = new InitializerMultiPhase_Uniform(         param, vs, sd, settings );
      std::cerr << "felbm_gpu: WARNING — fluid_initializer=\""<<f<<"\" not recognised (state/binary_file not ported); using uniform.\n"; }
    std::cout << "felbm_gpu: fluid_initializer = \""<<f<<"\"  -> "<<chosen<<"\n";
    in->initialize( h, g );
    delete in;
  }

  // ---- GPU engine ----
  MultiPhaseGPU gpu;
  gpu.init( sd, vs, settings, param, stream_mf, grad_mf );
  gpu.upload_state( h.data(), g.data() );
  gpu.record_target_mass();   // M0 for the order-parameter mass corrector

  unsigned int const steps = settings.max_iterations();
  unsigned int const fskip = settings.file_skip()?settings.file_skip():1u;

  std::cout << "felbm_gpu: n_sites="<<n<<"  steps="<<steps
            << "  precision="<<(FELBM_REAL_IS_DOUBLE?"double":"float")<<"\n";

  // ---- particle tracking (option b: reuse the CPU ParticleManager) -----------
  // Tracers live on the host and are advected on the GPU velocity field, which is
  // downloaded (compressed) and scattered into the global arrays each step. This
  // reuses the already-validated CPU advection/interpolation; particles are cheap
  // relative to the LBM step, so the device->host velocity copy is negligible.
  bool const do_particles = settings.particles_enable();
  int  const NG = (int)domain.size_sites();                 // global compressed size
  std::vector<double> ucx(NG,0.),ucy(NG,0.),ucz(NG,0.),unx(NG,0.),uny(NG,0.),unz(NG,0.);
  ParticleManager pm( domain, settings, ucx,ucy,ucz, unx,uny,unz );
  pm.hdf5_float32() = output_f32;      // match field output precision (output_float32)
  pm.hdf5_deflate() = output_deflate;  // match field gzip level (output_deflate; 0 = off)
  SubDomain::idx_vector const & l2g = sd.local_to_global_index();
  std::vector<real_t> uvtmp(n);

  auto scatter_vel = [&]( real_t const* d, std::vector<double>& out ){
    copy_d2h( uvtmp.data(), d, n );
    for( int i=0;i<n;++i ) out[ l2g[i] ] = (double)uvtmp[i];
  };

  unsigned int pskip = settings.particles_file_skip();
  if( !pskip ) pskip = fskip;
  if( !pskip ) pskip = 1u;
  // Particle output format, read directly from the cfg (so felbm_gpu doesn't depend
  // on felbm_local's Settings carrying a `particles_format` key). Defaults to HDF5:
  // ParticleManager::output_state dispatches on the extension and writes datasets
  // position / velocity / id — the velocity is interpolated from the (GPU-fed)
  // velocity field the manager holds. Requires a felbm_local with the HDF5 particle
  // writer; set `particles_format = csv` for the plain-text id,x,y,z fallback.
  std::string pfmt = cfg.exist("particles_format") ? cfg.get_value("particles_format") : std::string("h5");
  std::transform( pfmt.begin(), pfmt.end(), pfmt.begin(), [](unsigned char c){ return (char)std::tolower(c); } );
  std::string const pext = (pfmt=="csv") ? ".csv" : ".h5";

  // Throttle the (expensive) device->host velocity copy: refresh the velocity
  // snapshot every `particles_velocity_skip` LBM steps and hold it constant in
  // between, while still advecting the tracers every step (so the D_perp time
  // resolution is unchanged). Valid when the flow is quasi-steady over the
  // interval — the regime in which D_perp is measured. felbm_gpu-only cfg key,
  // read directly here (default 1 = refresh every step).
  unsigned int vskip = cfg.exist("particles_velocity_skip")
                     ? util::to_value<unsigned int>( cfg.get_value("particles_velocity_skip") ) : 1u;
  if( !vskip ) vskip = 1u;

  if( do_particles ){
    pm.initialize();
    std::cout << "felbm_gpu: seeded " << pm.n_particles() << " particles.\n";
  }

  // Time-dependent body-force scaling (matches the CPU scheduler loop). For the
  // GRL runs forcing_timedep = "constant" -> forcing_factor = 1. It multiplies the
  // separate `forcing` vector; with forcing = 0 (gravity-driven) it is inert but
  // must still be a valid number.
  std::string const ftd = settings.forcing_timedep();
  double const fperiod   = settings.forcing_period();
  double const PI        = 3.14159265358979323846;
  auto forcing_factor_at = [&]( unsigned int t )->double {
    if( ftd.find("sinusoidal") != std::string::npos ) return std::sin( 2.0*PI*double(t)/fperiod );
    if( ftd.find("square")     != std::string::npos ){ double x=std::sin(2.0*PI*double(t)/fperiod); return (x<0)?-1.0:((x>0)?1.0:0.0); }
    return 1.0;   // constant
  };

  // ---- log + timeseries files ------------------------------------------------
  // The reused host code logs setup lines through util::Log (to console when
  // console=true); here we ALSO write a proper run log + timeseries, so a headless
  // GRL run leaves a record. logf mirrors progress to stdout and the file.
  std::ofstream logf;
  if( settings.logging() )
  {
    logf.open( (settings.output_dir()+settings.log_filename()).c_str() );
    if( logf.is_open() )
    {
      logf << settings << "\n" << param << "\n";
      logf << "## felbm_gpu  n_sites="<<n<<"  steps="<<steps
           << "  precision="<<(FELBM_REAL_IS_DOUBLE?"double":"float")
           << "  collision="<<(settings.use_mrt()?"MRT":"BGK")
           << "  correct_op_mass="<<(settings.correct_op_mass()?"on":"off")
           << "  target_op_mass="<<gpu.target_mass
           << "  output=("<<(output_f32?"float32":"float64")<<", gzip "<<output_deflate<<")\n";
      logf.flush();
    }
  }
  auto logline = [&]( std::string const & s ){
    std::cout<<s<<"\n";
    if( logf.is_open() ){ logf<<s<<"\n"; logf.flush(); }
  };

  unsigned int lskip = settings.log_skip(); if(!lskip) lskip=fskip; if(!lskip) lskip=1u;
  std::ofstream tsf;
  if( !settings.timeseries_file().empty() )
  {
    tsf.open( (settings.output_dir()+settings.timeseries_file()).c_str() );
    if( tsf.is_open() ){ tsf<<"# iter   max_speed   mean_ux   mean_uy   mean_uz   mean_v2   total_order_parameter\n"; tsf.flush(); }
  }

  cudaDeviceSynchronize();
  auto _t0 = std::chrono::steady_clock::now();

  std::vector<double> hc,hrho,hux,huy,huz,hp;
  for( unsigned int t=0; t<=steps; ++t )
  {
    bool const file_now = (t % fskip == 0u);
    bool const log_now  = (t % lskip == 0u);
    if( file_now || log_now )
    {
      gpu.download( hc,hrho,hux,huy,huz,hp );
      if( file_now )
      {
        std::ostringstream fn; fn<<settings.output_dir()<<settings.output_name()<<"_"<<t<<".h5";
        write_fields_h5( fn.str(), n, hrho,hc,hux,huy,huz,hp, output_deflate, output_f32 );
        std::ostringstream m; m<<"  step "<<t<<"  wrote "<<fn.str(); logline( m.str() );
      }
      if( log_now )
      {
        double umax=0., ctot=0., sux=0., suy=0., suz=0., sv2=0.;
        for( int i=0;i<n;++i ){
          double ux=hux[i], uy=huy[i], uz=huz[i];
          double u2=ux*ux+uy*uy+uz*uz;
          if(u2>umax) umax=u2;
          sux+=ux; suy+=uy; suz+=uz; sv2+=u2; ctot+=hc[i];
        }
        umax=std::sqrt(umax);
        double const invn=1.0/(double)n;
        double const mux=sux*invn, muy=suy*invn, muz=suz*invn, mv2=sv2*invn;   // means over the pore space
        if( tsf.is_open() ){ tsf<<t<<"  "<<umax<<"  "<<mux<<"  "<<muy<<"  "<<muz<<"  "<<mv2<<"  "<<ctot<<"\n"; tsf.flush(); }
        std::ostringstream m; m<<"  step "<<t<<"   max|u|="<<umax<<"   <uy>="<<muy<<"   <v2>="<<mv2<<"   total_c="<<ctot; logline( m.str() );
      }
    }
    if( do_particles && t % pskip == 0u )
    {
      std::ostringstream pf; pf<<settings.output_dir()<<"particles_"<<t<<pext;
      pm.output_state( pf.str() );
    }
    if( t < steps )
    {
      gpu.P.forcing_factor = (real_t)forcing_factor_at( t );
      gpu.step();
      gpu.apply_mass_correction();   // no-op unless correct_op_mass = true
      if( do_particles )
      {
        if( t % vskip == 0u )   // refresh the velocity snapshot every vskip steps
        {
          scatter_vel( gpu.d_ux, unx );
          scatter_vel( gpu.d_uy, uny );
          scatter_vel( gpu.d_uz, unz );
          ucx=unx; ucy=uny; ucz=unz;   // held constant until the next refresh
        }
        pm.update();
      }
    }
  }

  cudaDeviceSynchronize();
  auto _t1 = std::chrono::steady_clock::now();
  double const sec = std::chrono::duration<double>( _t1 - _t0 ).count();
  double const mlups = sec>0.0 ? (double)n * (double)steps / sec / 1.0e6 : 0.0;
  { std::ostringstream m; m<<"felbm_gpu: "<<steps<<" steps, "<<n<<" sites in "<<sec<<" s  ->  "
      <<mlups<<" MLUPS  (streaming="<<(stream_mf?"mf":"CSR")<<", grad="<<(grad_mf?"mf":"CSR")<<")";
    logline( m.str() ); }

  gpu.free();
  logline( "felbm_gpu: done." );
  return 0;
}
