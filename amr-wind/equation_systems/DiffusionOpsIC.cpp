#include "amr-wind/equation_systems/DiffusionOpsIC.H"
#include "amr-wind/utilities/console_io.H"

#include "amr-wind/LinOps/AMReX_MLABecCecLaplacian.H"

namespace amr_wind {
namespace pde {

template <typename LinOpIc>
ConvDiffSolverIface<LinOpIc>::ConvDiffSolverIface(
    PDEFields& fields, const bool mesh_mapping, const std::string& prefix)
    : m_pdefields_cd(fields)
    , m_density_cd(fields.repo.get_field("density"))
    , m_options_cd(prefix, m_pdefields_cd.field.name() + "_" + prefix)
    , m_mesh_mapping(mesh_mapping)
{
    amrex::LPInfo isolve = m_options_cd.lpinfo();
    amrex::LPInfo iapply;

    amrex::Real omega = 0.99;

    iapply.setMaxCoarseningLevel(0);
    isolve.setMaxCoarseningLevel(0);

    const auto& mesh = m_pdefields_cd.repo.mesh();

    const int ncomp = m_pdefields_cd.field.num_comp();

    m_solver_cd.reset(new LinOpIc(
        mesh.Geom(0, mesh.finestLevel()), mesh.boxArray(0, mesh.finestLevel()),
        mesh.DistributionMap(0, mesh.finestLevel()), isolve, {}, ncomp));
    m_applier_cd.reset(new LinOpIc(
        mesh.Geom(0, mesh.finestLevel()), mesh.boxArray(0, mesh.finestLevel()),
        mesh.DistributionMap(0, mesh.finestLevel()), iapply, {}, ncomp));

    m_solver_cd->setMaxOrder(m_options_cd.max_order);
    m_applier_cd->setMaxOrder(m_options_cd.max_order);

    m_solver_cd->setRelaxation(omega);
    m_applier_cd->setRelaxation(omega);

    // It is the sub-classes responsibility to set the linear solver BC for the
    // operators.
}

template <typename LinOpIc>
void ConvDiffSolverIface<LinOpIc>::setup_operator_cd(
    LinOpIc& linop,
    const amrex::Real alpha,
    const amrex::Real beta,
    const amrex::Real gamma,
    const FieldState fstate)
{
    BL_PROFILE("amr-wind::setup_operator");
    auto& repo = m_pdefields_cd.repo;
    const int nlevels = repo.num_active_levels();

    linop.setScalars(alpha, beta, gamma);
    for (int lev = 0; lev < nlevels; ++lev) {
        linop.setLevelBC(lev, &m_pdefields_cd.field(lev));
    }
    this->set_acoeffs_cd(linop, fstate);
    set_bcoeffs_cd(linop);
    set_ccoeffs_cd(linop);
}

template <typename LinOpIc>
void ConvDiffSolverIface<LinOpIc>::set_acoeffs_cd(
    LinOpIc& linop, const FieldState fstate)
{
    BL_PROFILE("amr-wind::set_acoeffs");
    auto& repo = m_pdefields_cd.repo;
    const int nlevels = repo.num_active_levels();
    const auto& density = m_density_cd.state(fstate);
    Field const* mesh_detJ = m_mesh_mapping
                                 ? &(repo.get_mesh_mapping_detJ(FieldLoc::CELL))
                                 : nullptr;
    std::unique_ptr<ScratchField> rho_times_detJ =
        m_mesh_mapping ? repo.create_scratch_field(
                             1, m_density_cd.num_grow()[0], FieldLoc::CELL)
                       : nullptr;

    for (int lev = 0; lev < nlevels; ++lev) {
        if (m_mesh_mapping) {
            (*rho_times_detJ)(lev).setVal(0.0);
            amrex::MultiFab::AddProduct(
                (*rho_times_detJ)(lev), density(lev), 0, (*mesh_detJ)(lev), 0,
                0, 1, m_density_cd.num_grow()[0]);
            linop.setACoeffs(lev, (*rho_times_detJ)(lev));
        } else {
            linop.setACoeffs(lev, density(lev));
        }
    }
}

template <typename LinOpIc>
void ConvDiffSolverIface<LinOpIc>::setup_solver_cd(amrex::MLMG& mlmg_cd)
{
    BL_PROFILE("amr-wind::setup_solver");
    // Set all MLMG options
    m_options_cd(mlmg_cd);
}

template <typename LinOpIc>
void ConvDiffSolverIface<LinOpIc>::linsys_solve_impl_cd()
{
    BL_PROFILE("amr-wind::linsys_solve_impl_cd");
    FieldState fstate = FieldState::New;
    auto& repo = this->m_pdefields_cd.repo;
    auto& field = this->m_pdefields_cd.field;
    if (field.in_uniform_space()) {
        amrex::Abort(
            "For diffusion solve, velocity should not be in uniform mesh "
            "space.");
    }
    const auto& density = m_density_cd.state(fstate);
    const int nlevels = repo.num_active_levels();
    const int ndim = field.num_comp();
    auto rhs_ptr_cd = repo.create_scratch_field("rhs", field.num_comp(), 0);

    // Always multiply with rho since there is no diffusion term for density
    for (int lev = 0; lev < nlevels; ++lev) {
        auto& rhs_cd = (*rhs_ptr_cd)(lev);

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(rhs_cd, amrex::TilingIfNotGPU()); mfi.isValid();
             ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& rhs_a = rhs_cd.array(mfi);
            const auto& fld = field(lev).const_array(mfi);
            const auto& rho = density(lev).const_array(mfi);

            amrex::ParallelFor(
                bx, ndim,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                    rhs_a(i, j, k, n) = rho(i, j, k) * fld(i, j, k, n);
                });
        }
    }

    amrex::MLMG mlmg_cd(*this->m_solver_cd);
    this->setup_solver_cd(mlmg_cd);

    mlmg_cd.solve(
        field.vec_ptrs(), rhs_ptr_cd->vec_const_ptrs(),
        this->m_options_cd.rel_tol, this->m_options_cd.abs_tol);

    io::print_mlmg_info(field.name() + "_solve", mlmg_cd);
}

template <typename LinOpIc>
void ConvDiffSolverIface<LinOpIc>::linsys_solve_cd(const amrex::Real dt)
{
    FieldState fstate = FieldState::New;
    this->setup_operator_cd(*this->m_solver_cd, 1.0, dt, -dt, fstate);
    this->linsys_solve_impl_cd();
}

template class ConvDiffSolverIface<amrex::MLABecCecLaplacian>;

} // namespace pde
} // namespace amr_wind
