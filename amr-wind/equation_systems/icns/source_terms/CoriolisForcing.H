#ifndef CORIOLISFORCING_H
#define CORIOLISFORCING_H

#include "amr-wind/core/FieldRepo.H"
#include "amr-wind/equation_systems/icns/MomentumSource.H"

namespace amr_wind::pde::icns {

/** Account for Coriolis forcing due to earth's rotation
 *  \ingroup icns_src we_abl
 *
 */
class CoriolisForcing : public MomentumSource::Register<CoriolisForcing>
{
public:
    static std::string identifier() { return "CoriolisForcing"; }

    explicit CoriolisForcing(const CFDSim& /*sim*/);

    ~CoriolisForcing() override;

    void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;

private:
    const Field& m_velocity;

    ///@{
    /** Orientation of the grid w.r.t. to planetary coordinate system
     */
    amrex::Vector<amrex::Real> m_east{{1.0, 0.0, 0.0}};
    amrex::Vector<amrex::Real> m_north{{0.0, 1.0, 0.0}};
    amrex::Vector<amrex::Real> m_up{{0.0, 0.0, 1.0}};
    ///@}

    //! Latitude where the Coriolis forcing is computed (+ve northern
    //! hemisphere)
    amrex::Real m_latitude{0.0};

    amrex::Real m_sinphi{0.0};
    amrex::Real m_cosphi{0.0};

    //! `2.0 * \Omega`
    amrex::Real m_coriolis_factor{0.0};
};

} // namespace amr_wind::pde::icns

#endif /* CORIOLISFORCING_H */
