#include <type_traits>
#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#            SIMULATION STOP            #
#.......................................#
time.stop_time               =   -100.00     # Max (simulated) time to evolve
time.max_step                =   2000       # Max number of time steps

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#         TIME STEP COMPUTATION         #
#.......................................#
time.fixed_dt         =  50       # Use this constant dt if > 0
time.cfl              =   0.5       # CFL factor

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#            INPUT AND OUTPUT           #
#.......................................#
time.plot_interval            =  1000       # Steps between plot files
time.checkpoint_interval      =  -100       # Steps between checkpoint files

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#               PHYSICS                 #
#.......................................#
incflo.density        =  1.0             # Reference density
incflo.use_godunov = 0
transport.viscosity = 0.005
turbulence.model = Laminar

incflo.use_explicit_convection = false
incflo.diffusion_type = 1 
diffusion.mol_gradient_relax_factor = 1.0 

ICNS.source_terms = BodyForce
BodyForce.magnitude = 6e-2 0 0
incflo.physics = ChannelFlow
ChannelFlow.density = 1.0
ChannelFlow.Laminar = true
ChannelFlow.Turbulent_DNS = false
ChannelFlow.Mean_Velocity = 1.0

io.output_default_variables = 1

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#        ADAPTIVE MESH REFINEMENT       #
#.......................................#
amr.n_cell              = 32 32 16    # Grid cells at coarsest AMRlevel
amr.max_level           = 0           # Max AMR level in hierarchy

#¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨#
#              GEOMETRY                 #
#.......................................#
geometry.prob_lo        =   0.0  0.0  0.0  # Lo corner coordinates
geometry.prob_hi        =   6.0  1.0  1.0  # Hi corner coordinates
geometry.is_periodic    =   1   0   1   # Periodicity x y z (0/1)

# Boundary conditions
ylo.type =   "no_slip_wall"
yhi.type =   "no_slip_wall"

incflo.verbose  = 0

diffusion.max_coarsening_level = 0
