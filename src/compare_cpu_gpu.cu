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
//  Usage:  ./compare_cpu_gpu [steps=1] [N=48] [ratio=5] [geom=fluid|spheres|openbnd] [coll=bgk|mrt] [mf=0|1] [mfg=0|1] [fused=0|1] [fusecoll=0|1] [mrtfast=0|1] [inplace=0|1]
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
  bool inplace     = argc>11?(atoi(argv[11])!=0):false; // 1 = in-place streaming (no h2/g2)
  int  inlet_mode_arg = argc>12?atoi(argv[12]):0;      // openbnd: 0 single, 1 alternate, 2 split

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
  // geom=openbnd: pressure-driven inlet/outlet on an all-fluid duct. Exercises
  // OpenBoundaryOperator / k_open_bnd, which the periodic cases never touch.
  bool const open_bnd = (geom=="openbnd");
  s.correct_op_mass()=false; s.use_open_bnd()=open_bnd;
  s.size_x()=N; s.size_y()=N; s.size_z()=N;
  s.num_subdomains()=1u; s.slabbing_dir()=0u; s.shift_slabs()=false; s.load_balancing()=false;
  s.in_out_dir()=1u; s.buffer_layers()=2u; s.empty_layers()=0u; s.extrude_buffers()=false;
  s.inlet_fluid()=0u; s.outlet_fluid()=1u;
  s.use_inlet_pressure()=false; s.use_outlet_pressure()=false; s.use_inlet_velocity()=false;
  s.use_inlet_fluid()=false; s.use_outlet_fluid()=false; s.copy_to_buffers()=false;
  s.inlet_mode()=std::string("single"); s.inlet_period()=0.0; s.inlet_duty()=0.5;
  s.inlet_ramp()=0.0; s.inlet_split_dir()=std::string("x"); s.inlet_split_pos()=0.0;
  if( open_bnd )
  {
    s.use_inlet_pressure()=true;  s.pressure_inlet() =1.003;
    s.use_outlet_pressure()=true; s.pressure_outlet()=1.000;
    s.use_inlet_fluid()=(inlet_mode_arg!=3); // mode 3: local values, isolates the schedule
    s.use_outlet_fluid()=false;   // open drain: both phases leave freely
    s.empty_layers()=2u; s.extrude_buffers()=true;
    // injection mode from argv[12]: 0 single, 1 alternate (time), 2 split (space)
    if( inlet_mode_arg==1 ){ s.inlet_mode()=std::string("alternate");
                             s.inlet_period()=20.0; s.inlet_duty()=0.5; s.inlet_ramp()=4.0; }
    if( inlet_mode_arg==4 ){ s.inlet_mode()=std::string("stripes");
                             s.inlet_split_dir()=std::string("x");
                             s.inlet_period()=8.0; s.inlet_duty()=0.5;
                             s.inlet_split_pos()=0.0; s.inlet_ramp()=2.0; }
    if( inlet_mode_arg==2 ){ s.inlet_mode()=std::string("split");
                             s.inlet_split_dir()=std::string("x");
                             s.inlet_split_pos()=0.5*N; s.inlet_ramp()=3.0; }
  }
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
  gpu.stream_inplace     = inplace;
  gpu.init( sd, vs, s, pm, mf, mfg, fu, fc );

  if( open_bnd )   // mirror the driver's open-boundary setup
  {
    SubDomain::idx_vector const & iv = sd.inlet_verts();
    SubDomain::idx_vector const & ov = sd.outlet_verts();
    std::vector<int> inlet( iv.begin(), iv.end() ), outlet( ov.begin(), ov.end() );
    auto & Pp = gpu.params();
    Pp.open_bnd=1;
    Pp.use_in_vel  = s.use_inlet_velocity()  ?1:0;
    Pp.use_in_prs  = s.use_inlet_pressure()  ?1:0;
    Pp.use_in_fluid= s.use_inlet_fluid()     ?1:0;
    Pp.use_out_prs = s.use_outlet_pressure() ?1:0;
    Pp.use_out_fluid=s.use_outlet_fluid()    ?1:0;
    Pp.u_in_x=(real_t)s.u_inlet_x(); Pp.u_in_y=(real_t)s.u_inlet_y(); Pp.u_in_z=(real_t)s.u_inlet_z();
    Pp.p_in=(real_t)s.pressure_inlet(); Pp.p_out=(real_t)s.pressure_outlet();
    Pp.c_out_fixed=(real_t)(1.0-s.outlet_fluid());
    Pp.rho_out_fixed=(real_t)pm.phase_density(s.outlet_fluid());
    Pp.inlet_c_a=(real_t)(1.0-s.inlet_fluid());
    Pp.inlet_mode=(s.inlet_mode()=="alternate")?1:((s.inlet_mode()=="split")?2:((s.inlet_mode()=="stripes")?3:0));
    Pp.inlet_period=(real_t)s.inlet_period(); Pp.inlet_duty=(real_t)s.inlet_duty();
    Pp.inlet_ramp=(real_t)s.inlet_ramp(); Pp.split_pos=(real_t)s.inlet_split_pos();
    Pp.split_axis=(s.inlet_split_dir()=="y")?1:((s.inlet_split_dir()=="z")?2:0);
    std::vector<real_t> icoord;
    if( Pp.inlet_mode==2 || Pp.inlet_mode==3 ){ icoord.resize(inlet.size());
      for(size_t q=0;q<inlet.size();++q)
        icoord[q]=(real_t)sd.idx_to_position((unsigned)inlet[q])[(unsigned)Pp.split_axis]; }
    gpu.set_open_bnd( inlet, outlet, icoord );
    std::printf("  open boundaries: %zu inlet, %zu outlet nodes, inlet_mode=%s\n",
                inlet.size(), outlet.size(), s.inlet_mode().c_str());
  }

  gpu.upload_state( eng.h_data(), eng.g_data() );

  // --- advance both ---
  for( int t=0;t<steps;++t ){ eng.run_time_step(); gpu.step(); }

  // --- compare the distributions after `steps` steps ---
  std::vector<real_t> hh(Vn), gg(Vn);
  copy_d2h( hh.data(), gpu.d_h, Vn );
  copy_d2h( gg.data(), gpu.d_g, Vn );

  // --- debug: inspect the first inlet node on both sides ---
  if( open_bnd && getenv("FELBM_BC_DEBUG") )
  {
    SubDomain::idx_vector const & iv = sd.inlet_verts();
    if( !iv.empty() ){
      unsigned id = iv[0];
      std::printf("  [dbg] inlet node id=%u  n=%d\n", id, (int)sd.size_sites());
      std::printf("  [dbg] CPU h(0..3)[id] = %.6e %.6e %.6e %.6e\n",
        eng.h_data()[0*sd.size_sites()+id], eng.h_data()[1*sd.size_sites()+id],
        eng.h_data()[2*sd.size_sites()+id], eng.h_data()[3*sd.size_sites()+id]);
      std::printf("  [dbg] GPU h(0..3)[id] = %.6e %.6e %.6e %.6e\n",
        hh[0*sd.size_sites()+id], hh[1*sd.size_sites()+id],
        hh[2*sd.size_sites()+id], hh[3*sd.size_sites()+id]);
      double csum_c=0, csum_g=0;
      for(unsigned k=0;k<vs.size();++k){ csum_c+=eng.h_data()[k*sd.size_sites()+id];
                                         csum_g+=hh[k*sd.size_sites()+id]; }
      { SubDomain::idx_vector const & bv = sd.buffer_verts();
        bool inbuf=false; for(size_t q=0;q<bv.size();++q) if(bv[q]==id){inbuf=true;break;}
        std::printf("  [dbg] buffer_verts=%zu  node %u in buffer: %s\n",
                    bv.size(), id, inbuf?"YES":"no"); }
      std::printf("  [dbg] node %u: is_solid=%d is_streamed=%d\n", id,
                  (int)sd.is_solid(id), (int)sd.is_streamed(id));
      std::printf("  [dbg] c at inlet: CPU=%.6f  GPU=%.6f   (P.inlet_c_a=%.3f rho0=%.3f rho1=%.3f)\n",
        csum_c, csum_g, (double)gpu.params().inlet_c_a,
        (double)gpu.params().rho0, (double)gpu.params().rho1);
    }
  }


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
