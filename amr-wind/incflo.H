#ifndef INCFLO_H_
#define INCFLO_H_

#include <AMReX_AmrCore.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_iMultiFab.H>
#include <hydro_NodalProjector.H>

#include "amr-wind/incflo_enums.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/core/SimTime.H"
#include "amr-wind/core/FieldRepo.H"

namespace amr_wind {
namespace pde {
class PDEBase;
}
class RefinementCriteria;
class RefineCriteriaManager;
} // namespace amr_wind

/**
 *  \defgroup driver AMR-Wind driver code
 *
 *  Code that implements initialization and time-stepping logic.
 *
 *  This module documents the high-level driver code that is manages the
 *  initialization and execution of the time-integration algorithm when invoked
 *  from the executable. Currently there are two phases during AMR-Wind
 * execution:
 *
 *  <ol>
 *
 *  <li> **Initialization** \ref incflo::ReadParameters "Read input parameters",
 *  \ref incflo::InitData "initialize mesh, physics modules, etc.".
 *
 *  <li> **Time loop** Run user-defined number of timesteps \ref incflo::Evolve
 *  "in a loop" that \ref incflo::Advance "advances" the flow-field.
 *
 *  </ol>
 */

/** AMR-Wind driver class
 *  \ingroup driver
 *
 */
class incflo : public amrex::AmrCore
{
public:
    incflo();
    ~incflo() override;

    // Initialize multilevel AMR data
    void InitData();

    // Evolve solution to final time through repeated calls to Advance()
    void Evolve();

    // Tag cells for refinement
    void
    ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow)
        override;

    // Make a new level from scratch using provided BoxArray and
    // DistributionMapping Only used during initialization
    void MakeNewLevelFromScratch(
        int lev,
        amrex::Real time,
        const amrex::BoxArray& new_grids,
        const amrex::DistributionMapping& new_dmap) override;

    // Make a new level using provided BoxArray and DistributionMapping,
    // and fill with interpolated coarse level data
    void MakeNewLevelFromCoarse(
        int lev,
        amrex::Real time,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override;

    // Remake an existing level using provided BoxArray and DistributionMapping,
    // and fill with existing fine and coarse data
    void RemakeLevel(
        int lev,
        amrex::Real time,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm) override;

    // Delete level data
    void ClearLevel(int lev) override;

    void init_mesh();
    void init_amr_wind_modules();
    void prepare_for_time_integration();
    bool regrid_and_update();
    void pre_advance_stage1();
    void pre_advance_stage2();
    void do_advance();
    void advance();
    void prescribe_advance();
    void post_advance_work();

    amr_wind::CFDSim& sim() { return m_sim; }
    const amr_wind::SimTime& time() const { return m_time; }
    amr_wind::FieldRepo& repo() { return m_repo; }
    const amr_wind::FieldRepo& repo() const { return m_repo; }

    amr_wind::pde::PDEBase& icns() { return m_sim.pde_manager().icns(); }
    const amr_wind::pde::PDEBase& icns() const
    {
        return m_sim.pde_manager().icns();
    }
    amr_wind::pde::PDEMgr::TypeVector& scalar_eqns()
    {
        return m_sim.pde_manager().scalar_eqns();
    }
    const amr_wind::pde::PDEMgr::TypeVector& scalar_eqns() const
    {
        return m_sim.pde_manager().scalar_eqns();
    }

    amr_wind::Field& velocity() const { return m_repo.get_field("velocity"); }
    amr_wind::Field& density() const { return m_repo.get_field("density"); }
    amr_wind::Field& temperature() const
    {
        return m_repo.get_field("temperature");
    }
    amr_wind::Field& pressure() const { return m_repo.get_field("p"); }
    amr_wind::Field& grad_p() const { return m_repo.get_field("gp"); }

    // for cuda
    void ComputeDt(bool explicit_diffusion);
    void ComputePrescribeDt();

    void set_inflow_velocity(
        int lev, amrex::Real time, amrex::MultiFab& vel, int nghost);

    void ApplyPredictor(bool incremental_projection = false);
    void ApplyCorrector();
    void ApplyPrescribeStep();

    void ApplyProjection(
        amrex::Vector<amrex::MultiFab const*> density,
        amrex::Real time,
        amrex::Real scaling_factor,
        bool incremental);

    //! Initialize Physics instances as well as PDEs (include turbulence models)
    void init_physics_and_pde();

    void ReadCheckpointFile();

private:
    //
    // member variables
    //

    amr_wind::CFDSim m_sim;
    amr_wind::SimTime& m_time;
    amr_wind::FieldRepo& m_repo;

    std::unique_ptr<amr_wind::RefineCriteriaManager> m_mesh_refiner;

    // Be verbose?
    int m_verbose = 0;

    // Initial projection / iterations
    bool m_do_initial_proj = true;
    int m_initial_iterations = 3;

    bool m_constant_density = true;

    bool m_use_godunov = true;

    // Prescribe advection velocity
    bool m_prescribe_vel = false;

    //! number of cells on all levels including covered cells
    amrex::Long m_cell_count{-1};

    DiffusionType m_diff_type = DiffusionType::Crank_Nicolson;

    //
    // end of member variables
    //

    amrex::FabFactory<amrex::FArrayBox> const& Factory(int lev) const noexcept
    {
        return m_repo.factory(lev);
    }

    bool need_divtau() const
    {
        return (m_use_godunov || (DiffusionType::Implicit != m_diff_type));
    }

    amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>
    get_projection_bc(amrex::Orientation::Side side) const noexcept;

    ///////////////////////////////////////////////////////////////////////////
    //
    // setup
    //
    ///////////////////////////////////////////////////////////////////////////

    void set_background_pressure();
    void ReadParameters();
    void InitialProjection();
    void InitialIterations();

    ///////////////////////////////////////////////////////////////////////////
    //
    // utilities
    //
    ///////////////////////////////////////////////////////////////////////////
    void PrintMaxValues(const std::string& header);
    void PrintMaxVelLocations(const std::string& header);
    void PrintMaxVel(int lev) const;
    void PrintMaxGp(int lev) const;
    void CheckForNans(int lev) const;
};

#endif
