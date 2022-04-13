#include "amr-wind/equation_systems/DiffusionOpsIC.H"
#include "amr-wind/utilities/console_io.H"

#include "amr-wind/LinOps/AMReX_MLABecCecLaplacian.H"

namespace amr_wind {
namespace pde {

template <typename LinOpIc>
ConvDiffSolverIface<LinOpIc>::ConvDiffSolverIface(
    PDEFields& fields, const bool mesh_mapping, const std::string& prefix)
    : m_pdefields(fields)
    , m_density(fields.repo.get_field("density"))
    , m_options(prefix, m_pdefields.field.name() + "_" + prefix)
    , m_mesh_mapping(mesh_mapping)
{
    amrex::LPInfo isolve = m_options.lpinfo();
    amrex::LPInfo iapply;

    amrex::Real omega = 0.99;

    iapply.setMaxCoarseningLevel(0);
    isolve.setMaxCoarseningLevel(0);

    const auto& mesh = m_pdefields.repo.mesh();

    const int ncomp = m_pdefields.field.num_comp();

    amrex::BCRec const* bcrec = m_pdefields.field.bcrec().data();
    amrex::BCRec const* d_bcrec = m_pdefields.field.bcrec_device().data();

    m_solver.reset(new LinOpIc(
        mesh.Geom(0, mesh.finestLevel()), mesh.boxArray(0, mesh.finestLevel()),
        mesh.DistributionMap(0, mesh.finestLevel()), isolve, {}, ncomp));
    m_applier.reset(new LinOpIc(
        mesh.Geom(0, mesh.finestLevel()), mesh.boxArray(0, mesh.finestLevel()),
        mesh.DistributionMap(0, mesh.finestLevel()), iapply, {}, ncomp));

    m_solver->setMaxOrder(m_options.max_order);
    m_applier->setMaxOrder(m_options.max_order);

    m_solver->setRelaxation(omega);
    m_applier->setRelaxation(omega);

    m_solver->setGradientRelaxation(m_options.mol_gradient_relax_factor);
    m_applier->setGradientRelaxation(m_options.mol_gradient_relax_factor);

    m_solver->setBoundaryDiscretization(bcrec[0], d_bcrec[0]);
    m_applier->setBoundaryDiscretization(bcrec[0], d_bcrec[0]);

    // It is the sub-classes responsibility to set the linear solver BC for the
    // operators.
}

template <typename LinOpIc>
void ConvDiffSolverIface<LinOpIc>::setup_operator(
    LinOpIc& linop,
    const amrex::Real alpha,
    const amrex::Real beta,
    const amrex::Real gamma,
    const FieldState fstate)
{
    BL_PROFILE("amr-wind::setup_operator");
    auto& repo = m_pdefields.repo;
    const int nlevels = repo.num_active_levels();

    linop.setScalars(alpha, beta, gamma);
    for (int lev = 0; lev < nlevels; ++lev) {
        linop.setLevelBC(lev, &m_pdefields.field(lev));
    }
    this->set_acoeffs(linop, fstate);
    set_bcoeffs(linop);
    set_ccoeffs(linop);
}

template <typename LinOpIc>
void ConvDiffSolverIface<LinOpIc>::set_acoeffs(
    LinOpIc& linop, const FieldState fstate)
{
    BL_PROFILE("amr-wind::set_acoeffs");
    auto& repo = m_pdefields.repo;
    const int nlevels = repo.num_active_levels();
    const auto& density = m_density.state(fstate);
    Field const* mesh_detJ = m_mesh_mapping
                                 ? &(repo.get_mesh_mapping_detJ(FieldLoc::CELL))
                                 : nullptr;
    std::unique_ptr<ScratchField> rho_times_detJ =
        m_mesh_mapping ? repo.create_scratch_field(
                             1, m_density.num_grow()[0], FieldLoc::CELL)
                       : nullptr;

    for (int lev = 0; lev < nlevels; ++lev) {
        if (m_mesh_mapping) {
            (*rho_times_detJ)(lev).setVal(0.0);
            amrex::MultiFab::AddProduct(
                (*rho_times_detJ)(lev), density(lev), 0, (*mesh_detJ)(lev), 0,
                0, 1, m_density.num_grow()[0]);
            linop.setACoeffs(lev, (*rho_times_detJ)(lev));
        } else {
            linop.setACoeffs(lev, density(lev));
        }
    }
}

template <typename LinOpIc>
void ConvDiffSolverIface<LinOpIc>::setup_solver(amrex::MLMG& mlmg)
{
    BL_PROFILE("amr-wind::setup_solver");
    // Set all MLMG options
    m_options(mlmg);
}

template <typename LinOpIc>
void ConvDiffSolverIface<LinOpIc>::linsys_solve_impl()
{
    BL_PROFILE("amr-wind::linsys_solve_impl");
    FieldState fstate = FieldState::New;
    auto& repo = this->m_pdefields.repo;
    auto& field = this->m_pdefields.field;
    if (field.in_uniform_space()) {
        amrex::Abort(
            "For diffusion solve, velocity should not be in uniform mesh "
            "space.");
    }
    const auto& density = m_density.state(fstate);
    const int nlevels = repo.num_active_levels();
    const int ndim = field.num_comp();
    auto rhs_ptr = repo.create_scratch_field("rhs", field.num_comp(), 0);

    // Always multiply with rho since there is no diffusion term for density
    for (int lev = 0; lev < nlevels; ++lev) {
        auto& rhs = (*rhs_ptr)(lev);

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid();
             ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& rhs_a = rhs.array(mfi);
            const auto& fld = field(lev).const_array(mfi);
            const auto& rho = density(lev).const_array(mfi);

            amrex::ParallelFor(
                bx, ndim,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                    rhs_a(i, j, k, n) = rho(i, j, k) * fld(i, j, k, n);
                });
        }
    }

    amrex::MLMG mlmg(*this->m_solver);
    this->setup_solver(mlmg);

    mlmg.solve(
        field.vec_ptrs(), rhs_ptr->vec_const_ptrs(), this->m_options.rel_tol,
        this->m_options.abs_tol);

    io::print_mlmg_info(field.name() + "_solve", mlmg);
}

template <typename LinOpIc>
void ConvDiffSolverIface<LinOpIc>::linsys_solve(const amrex::Real dt)
{
    FieldState fstate = FieldState::New;
    this->setup_operator(*this->m_solver, 1.0, dt, -dt, fstate);
    this->linsys_solve_impl();
}

template class ConvDiffSolverIface<amrex::MLABecCecLaplacian>;

} // namespace pde
} // namespace amr_wind
