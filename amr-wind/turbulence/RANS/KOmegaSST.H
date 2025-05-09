#ifndef KOMEGASST_H
#define KOMEGASST_H

#include <string>
#include "amr-wind/turbulence/TurbModelBase.H"

namespace amr_wind::turbulence {

/** K-Omega-SST RANS turbulence model
 *
 * This also serves as the base class for all k-omega type RANS models
 *
 * Turbulence model described in
 *
 * Menter, F., Esch, T.,
 * "Elements of Industrial Heat Transfer Prediction",
 * 16th Brazilian Congress of Mechanical Engineering (COBEM),
 * Nov. 2001.
 *
 * with updated coefficients from
 *
 * Menter, F. R., Kuntz, M., and Langtry, R.,
 * "Ten Years of Industrial Experience with the SST Turbulence Model",
 * Turbulence, Heat and Mass Transfer 4, 2003,
 * pp. 625 - 632.
 *
 * Also see
 *
 * http://turbmodels.larc.nasa.gov/sst.html
 *
 */
template <typename Transport>
class KOmegaSST : public TurbModelBase<Transport>
{
public:
    static std::string identifier()
    {
        return "KOmegaSST-" + Transport::identifier();
    }

    explicit KOmegaSST(CFDSim& sim);

    ~KOmegaSST() override;

    std::string model_name() const override { return "KOmegaSST"; }

    //! Update the turbulent viscosity field
    void update_turbulent_viscosity(
        const FieldState fstate, const DiffusionType diff_type) override;

    //! No post advance work for this model
    void post_advance_work() override {}

    //! Update the effective scalar diffusivity field
    void update_scalar_diff(Field& deff, const std::string& name) override;

    //! Parse turbulence model coefficients
    void parse_model_coeffs() override;

    //! Return turbulence model coefficients
    TurbulenceModel::CoeffsDictType model_coeffs() const override;

protected:
    Field& m_vel;

    Field& m_f1;
    Field& m_shear_prod;
    Field& m_buoy_term;
    Field& m_diss;
    Field& m_sdr_src;
    Field& m_sdr_diss;
    Field& m_rho;
    Field& m_walldist;

    Field* m_tke;
    Field* m_sdr;

    //! Turbulence constants
    amrex::Real m_beta_star{0.09};
    amrex::Real m_alpha1{0.5555555555555556};
    amrex::Real m_alpha2{0.44};
    amrex::Real m_beta1{0.075};
    amrex::Real m_beta2{0.0828};
    amrex::Real m_sigma_k1{0.85};
    amrex::Real m_sigma_k2{1.0};
    amrex::Real m_sigma_omega1{0.5};
    amrex::Real m_sigma_omega2{0.856};
    amrex::Real m_a1{0.31};
    amrex::Real m_tke_amb{0.0};
    amrex::Real m_sdr_amb{0.0};

    //! Buoyancy modified constants
    bool m_include_buoyancy{false};
    amrex::Real m_buoyancy_factor = 0.0;
    amrex::Real m_sigma_t{0.85};
    amrex::Vector<amrex::Real> m_gravity{{0.0, 0.0, -9.81}};

    DiffusionType m_diff_type = DiffusionType::Crank_Nicolson;
};

} // namespace amr_wind::turbulence

#endif /* KOMEGASST_H */
