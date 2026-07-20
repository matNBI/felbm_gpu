//=============================================================================
//  compare_cpu_gpu — CPU vs GPU validation harness for the multiphase engine.
//
//  Builds a small fully-periodic all-fluid box with a spherical droplet, runs
//  the CPU EngineMultiPhase and the GPU MultiPhaseGPU from the SAME initial
//  distributions for N steps, and reports the max / mean absolute difference of
//  the h and g distributions (the fundamental state — if these match, all
//  derived fields match). Because the GPU uploads the CPU's exact stencil
//  operators, any real discrepancy points at one pointwise kernel.
//
//  Memory: default 48^3 all-fluid ~= 0.8 GB device (well under 2 GB). Raise N3
//  toward ~64 to approach the 2 GB budget.
//
//  Usage:  ./compare_cpu_gpu [steps=1] [N=48] [ratio=5] [geom=fluid|spheres] [coll=bgk|mrt] [mf=0|1] [mfg=0|1] [fused=0|1] [fusecoll=0|1] [mrtfast=0|1]
//    mf=1  matrix-free streaming;  mfg=1  matrix-free operators (both ~exact);
//    fused=1  fold dir-derivatives into equilibria+force (implies mfg);
//    fusecoll=1  fully fuse equilibria+force+collision+apply (implies fused).
//    geom=spheres inserts solid sphere obstacles, exercising the halfway
//    bounce-back streaming + biased-difference near-wall stencils (the GRL
//    porous regime). geom=fluid (default) is the all-fluid periodic box.
//    coll=mrt uses the multiple-relaxation-time collision (mrt_lambda=0.1875).
//=============================================================================

#include <lbm.h>
#include <util.h>

#include <felbm_gpu/multiphase_gpu.cuh>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

using namespace lbm;
using namespace felbm_gpu;

// expose the CPU engine's distributions for an exact IC transfer + comparison
struct EngineMPAccess : public lbm::EngineMultiPhase
{
  using lbm::EngineMultiPhase::EngineMultiPhase;
  double const * h_data() const { return m_data_h.data(); }
  double const * g_data() const { return m_data_g.data(); }
  double const * mu()     const { return m_fields.chemical_potential().data(); }
};

// compare a GPU device field (real_t, n) to a CPU host field (double, n)
static void report_dev( char const* name, real_t const* d_gpu, double const* cpu, int n )
{
  std::vector<real_t> hbuf(n);
  felbm_gpu::copy_d2h( hbuf.data(), d_gpu, n );
  double max_abs=0, max_ref=0; int argmax=-1, bad=0;
  for( int i=0;i<n;++i ){
    double gv=(double)hbuf[i], cv=cpu[i];
    if(!std::isfinite(gv)||!std::isfinite(cv)){ ++bad; continue; }
    double dd=std::fabs(gv-cv);
    if(dd>max_abs){ max_abs=dd; argmax=i; }
    max_ref=std::fmax(max_ref,std::fabs(cv));
  }
  std::printf("  %-4s: max|Δ|=%.3e  (max|cpu|=%.3e at %d%s)\n",
              name, max_abs, max_ref, argmax, bad?", NON-FINITE present":"");
}

static void report( char const* name, std::vector<real_t> const& gpu, double const* cpu, int Vn )
{
  double max_abs=0, sum_abs=0, max_ref=0;
  long   cnt=0;
  int    argmax=-1, gpu_bad=0, cpu_bad=0, first_bad=-1;
  for( int j=0;j<Vn;++j ){
    double gv=(double)gpu[j], cv=cpu[j];
    bool gf=std::isfinite(gv), cf=std::isfinite(cv);
    if(!gf){ ++gpu_bad; if(first_bad<0) first_bad=j; }
    if(!cf){ ++cpu_bad; if(first_bad<0) first_bad=j; }
    if( gf && cf ){
      double d=std::fabs(gv-cv);
      if(d>max_abs){ max_abs=d; argmax=j; }
      sum_abs+=d; ++cnt;
      max_ref=std::fmax(max_ref,std::fabs(cv));
    }
  }
  std::printf("  %-3s : max|Δ|=%.3e  mean|Δ|=%.3e  over %ld finite  (max|cpu|=%.3e at %d)\n",
              name, max_abs, cnt?sum_abs/(double)cnt:0.0, cnt, max_ref, argmax);
  if( gpu_bad || cpu_bad )
    std::printf("        !! non-finite values: GPU=%d  CPU=%d  (first at idx %d)\n",
                gpu_bad, cpu_bad, first_bad);
}

int main( int argc, char** argv )
{
  int    steps = argc>1?atoi(argv[1]):1;
  unsigned N   = argc>2?(unsigned)atoi(argv[2]):48u;
  double ratio = argc>3?atof(argv[3]):5.0;
  std::string geom = argc>4?argv[4]:"fluid";
  std::string coll = argc>5?argv[5]:"bgk";     // bgk | mrt
  bool mf          = argc>6?(atoi(argv[6])!=0):false;   // 1 = matrix-free streaming
  bool mfg         = argc>7?(atoi(argv[7])!=0):false;   // 1 = matrix-free grad_cd
  bool fu          = argc>8?(atoi(argv[8])!=0):false;   // 1 = fused equilibria+force (implies mfg)
  bool fc          = argc>9?(atoi(argv[9])!=0):false;   // 1 = fully fused collision (implies fu)
  bool mrtfast     = argc>10?(atoi(argv[10])!=0):false; // 1 = real_t MRT moment transform

  double const sigma=0.01, iw=4.0;
  double const R = 0.25*N;

  VelocitySetD3Q19 vs;

  // --- parameters (mirror test_mp_droplet derivations) ---
  ParametersMultiPhase pm;
  pm.add_phase_density(ratio); pm.add_phase_density(1.0);
  pm.add_relaxation_time(1.0); pm.add_relaxation_time(1.0);
  pm.interface_width()=iw; pm.surface_tension()=sigma;
  pm.mobility_coeff()=0.02; pm.phi()=0.0; pm.density_solid()=ratio+1.0;
  pm.kappa()=1.5*iw*sigma; pm.beta()=12.0*sigma/iw;
  pm.mobility()=pm.mobility_coeff()/pm.beta();
  pm.boundary_coefficient()= (std::fabs(pm.kappa())<1e-16)?0.0:pm.phi()/pm.kappa();
  pm.rho_avg()=0.5*(ratio+1.0);
  pm.forcing_factor()=1.0;

  // --- droplet initializer cfg ---
  { FILE* f=fopen("/tmp/felbm_gpu_cmp_drop.cfg","w");
    fprintf(f,"cx = %.1f\ncy = %.1f\ncz = %.1f\nr = %.1f\n",0.5*N,0.5*N,0.5*N,R); fclose(f); }

  // --- settings: fully periodic all-fluid box, BGK, no open boundaries ---
  Settings s; s.verbose()=false; s.use_halfway_bb()=false;
  s.use_mrt()=(coll=="mrt"); s.mrt_lambda()=0.1875;
  s.correct_op_mass()=false; s.use_open_bnd()=false;
  s.size_x()=N; s.size_y()=N; s.size_z()=N;
  s.num_subdomains()=1u; s.slabbing_dir()=0u; s.shift_slabs()=false; s.load_balancing()=false;
  s.in_out_dir()=1u; s.buffer_layers()=2u; s.empty_layers()=0u; s.extrude_buffers()=false;
  s.inlet_fluid()=0u; s.outlet_fluid()=1u;
  s.use_inlet_pressure()=false; s.use_outlet_pressure()=false; s.use_inlet_velocity()=false;
  s.use_inlet_fluid()=false; s.use_outlet_fluid()=false; s.copy_to_buffers()=false;
  s.acceleration()=Vector3d(0,0,0); s.forcing()=Vector3d(0,0,0);
  s.forcing_timedep()=std::string("constant");
  s.fluid_initializer()=std::string("spherical_droplet");
  s.fluid_init_cfg_file().load("/tmp/felbm_gpu_cmp_drop.cfg");

  // optional solid sphere obstacles (exercises bounce-back + biased stencils)
  bool const with_obstacles = (geom=="spheres");
  if( with_obstacles )
  {
    unsigned r = (N>=32u)?6u:3u;
    FILE* f=fopen("/tmp/felbm_gpu_cmp_spheres.cfg","w");
    fprintf(f,"seed = 12345\nradius = %u\nmax_number = 6\ndistance = 3\nis_hele_shaw = false\n", r);
    fclose(f);
    s.domain_geometry()=std::string("spheres_repulsive");
    s.domain_cfg_file().load("/tmp/felbm_gpu_cmp_spheres.cfg");
    s.use_halfway_bb()=true;   // activate the halfway bounce-back + biased near-wall stencils
  }

  Domain domain = make_domain( s, vs );
  if( with_obstacles )
  {
    DomainInitializer_SpheresRepulsive di( s, vs );
    di.initialize( domain );
  }
  DomainManager dm = make_subdomains( domain, s );
  SubDomain const & sd = dm.subdomain(0u);

  int const Vn = (int)vs.size()*(int)sd.size_sites();

  unsigned const n_fluid = sd.size_sites();
  std::printf("compare_cpu_gpu: N=%u^3  geom=%s  fluid=%u/%u (%.1f%% solid)  coll=%s  ratio=%.1f  steps=%d  precision=%s\n",
              N, geom.c_str(), n_fluid, N*N*N, 100.0*(1.0-(double)n_fluid/((double)N*N*N)),
              coll.c_str(), ratio, steps, FELBM_REAL_IS_DOUBLE?"double":"float");

  // --- CPU engine (auto-initialises h,g + fields) ---
  EngineMPAccess eng( vs, s, pm, domain, sd );

  // --- GPU engine, same initial distributions ---
  MultiPhaseGPU gpu;
  gpu.mrt_fast_transform = mrtfast;
  gpu.init( sd, vs, s, pm, mf, mfg, fu, fc );
  gpu.upload_state( eng.h_data(), eng.g_data() );

  // --- advance both ---
  for( int t=0;t<steps;++t ){ eng.run_time_step(); gpu.step(); }

  // --- compare the distributions after `steps` steps ---
  std::vector<real_t> hh(Vn), gg(Vn);
  copy_d2h( hh.data(), gpu.d_h, Vn );
  copy_d2h( gg.data(), gpu.d_g, Vn );

  std::printf("After %d step(s), CPU vs GPU distribution difference:\n", steps);
  report( "h", hh, eng.h_data(), Vn );
  report( "g", gg, eng.g_data(), Vn );

  // Field-level localisation (both hold the start-of-last-step fields; clean at
  // steps=1). Whichever field first diverges points at the responsible kernel.
  std::printf("Field difference (start-of-last-step; unambiguous at steps=1):\n");
  int const nn=(int)sd.size_sites();
  report_dev("c",   gpu.d_c,   eng.concentration().data(), nn);
  report_dev("rho", gpu.d_rho, eng.density().data(),       nn);
  report_dev("mu",  gpu.d_mu,  eng.mu(),                   nn);
  report_dev("ux",  gpu.d_ux,  eng.u_x().data(),           nn);
  report_dev("uy",  gpu.d_uy,  eng.u_y().data(),           nn);
  report_dev("uz",  gpu.d_uz,  eng.u_z().data(),           nn);
  report_dev("p",   gpu.d_p,   eng.pressure().data(),      nn);

  gpu.free();
  std::printf("done.\n");
  return 0;
}
