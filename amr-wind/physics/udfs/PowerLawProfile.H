#ifndef POWERLAWPROFILE_H
#define POWERLAWPROFILE_H

#include "AMReX_Geometry.H"
#include "AMReX_Gpu.H"

namespace amr_wind {

class Field;

namespace udf {

struct PowerLawProfile
{
    struct DeviceOp
    {
        int idir{2};
        amrex::Real zoffset{0.0};
        amrex::Real zref;
        amrex::Real uref;
        amrex::Real shear_exp;
        amrex::Real umin{0.0};
        amrex::Real umax{100.0};
        amrex::Real uvec[AMREX_SPACEDIM];

        AMREX_GPU_DEVICE
        inline void operator()(
            const amrex::IntVect& iv,
            amrex::Array4<amrex::Real> const& field,
            amrex::GeometryData const& geom,
            const amrex::Real /*unused*/,
            amrex::Orientation /*unused*/,
            const int comp,
            const int dcomp,
            const int orig_comp) const
        {
            const auto* problo = geom.ProbLo();
            const auto* dx = geom.CellSize();
            const auto zco = problo[idir] + (iv[idir] + 0.5) * dx[idir];

            const amrex::Real zeff = zco - zoffset;
            amrex::Real pfac =
                (zeff > 0.0) ? std::pow((zeff / zref), shear_exp) : 0.0;
            pfac = amrex::min(amrex::max(pfac, umin), umax);

            field(iv[0], iv[1], iv[2], dcomp + comp) =
                uvec[orig_comp + comp] * pfac;
        }
    };

    using DeviceType = DeviceOp;

    static std::string identifier() { return "PowerLawProfile"; }

    explicit PowerLawProfile(const Field& fld);

    DeviceType device_instance() const { return m_op; }

    DeviceOp m_op;

}; // namespace udf

} // namespace udf
} // namespace amr_wind

#endif /* POWERLAWPROFILE_H */
