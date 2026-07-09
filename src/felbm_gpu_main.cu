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
  settings.num_subdomains() = 1u;

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

  ParametersMultiPhase param = make_parameters<ParametersMultiPhase>( cfg );

  int const n = (int)sd.size_sites();

  // ---- initial condition on the host ----
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

  // ---- GPU engine ----
  MultiPhaseGPU gpu;
  gpu.init( sd, vs, settings, param );
  gpu.upload_state( h.data(), g.data() );

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

  std::vector<double> hc,hrho,hux,huy,huz,hp;
  for( unsigned int t=0; t<=steps; ++t )
  {
    if( t % fskip == 0u )
    {
      gpu.download( hc,hrho,hux,huy,huz,hp );
      std::ostringstream fn; fn<<settings.output_dir()<<settings.output_name()<<"_"<<t<<".h5";
      write_fields_h5( fn.str(), n, hrho,hc,hux,huy,huz,hp );
      std::cout << "  step "<<t<<"  wrote "<<fn.str()<<"\n";
    }
    if( do_particles && t % pskip == 0u )
    {
      std::ostringstream pf; pf<<settings.output_dir()<<"particles_"<<t<<pext;
      pm.output_state( pf.str() );
    }
    if( t < steps )
    {
      gpu.step();
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

  gpu.free();
  std::cout << "felbm_gpu: done.\n";
  return 0;
}
