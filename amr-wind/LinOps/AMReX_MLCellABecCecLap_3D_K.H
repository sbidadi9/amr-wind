#ifndef AMREX_MLCELLABECCECLAP_3D_K_H_
#define AMREX_MLCELLABECCECLAP_3D_K_H_
#include <AMReX_Config.H>

namespace amrex {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int coarsen_overset_mask(
    Box const& bx,
    Array4<int> const& cmsk,
    Array4<int const> const& fmsk) noexcept
{
    int nerrors = 0;
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
    for (int k = lo.z; k <= hi.z; ++k) {
        for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                int ii = 2 * i;
                int jj = 2 * j;
                int kk = 2 * k;
                cmsk(i, j, k) =
                    fmsk(ii, jj, kk) + fmsk(ii + 1, jj, kk) +
                    fmsk(ii, jj + 1, kk) + fmsk(ii + 1, jj + 1, kk) +
                    fmsk(ii, jj, kk + 1) + fmsk(ii + 1, jj, kk + 1) +
                    fmsk(ii, jj + 1, kk + 1) + fmsk(ii + 1, jj + 1, kk + 1);
                if (cmsk(i, j, k) == 8) {
                    cmsk(i, j, k) = 1;
                } else if (cmsk(i, j, k) != 0) {
                    ++nerrors;
                }
            }
        }
    }
    return nerrors;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void coarsen_overset_mask(
    int i,
    int j,
    int k,
    Array4<int> const& cmsk,
    Array4<int const> const& fmsk) noexcept
{
    int ii = 2 * i;
    int jj = 2 * j;
    int kk = 2 * k;
    cmsk(i, j, k) = fmsk(ii, jj, kk) + fmsk(ii + 1, jj, kk) +
                    fmsk(ii, jj + 1, kk) + fmsk(ii + 1, jj + 1, kk) +
                    fmsk(ii, jj, kk + 1) + fmsk(ii + 1, jj, kk + 1) +
                    fmsk(ii, jj + 1, kk + 1) + fmsk(ii + 1, jj + 1, kk + 1);
    if (cmsk(i, j, k) == 8) {
        cmsk(i, j, k) = 1;
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mllinop_apply_innu_xlo_ic(
    int i,
    int j,
    int k,
    Array4<Real> const& rhs,
    Array4<int const> const& mask,
    Array4<Real const> const& bcoef,
    Array4<Real const> const& ccoef,
    BoundCond bct,
    Real /*bcl*/,
    Array4<Real const> const& bcval,
    Real fac_vis,
    Real fac_conv,
    bool has_bcoef,
    bool has_ccoef,
    int icomp) noexcept
{
    if (bct == AMREX_LO_NEUMANN && mask(i, j, k) == 2) {
        Real b = (has_bcoef) ? bcoef(i + 1, j, k, icomp) : Real(1.0);
        Real c = (has_ccoef) ? ccoef(i + 1, j, k, icomp) : Real(1.0);
        rhs(i + 1, j, k, icomp) -=
            (fac_vis * b * bcval(i, j, k, icomp) +
             fac_conv * c * bcval(i, j, k, icomp));
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mllinop_apply_innu_xhi_ic(
    int i,
    int j,
    int k,
    Array4<Real> const& rhs,
    Array4<int const> const& mask,
    Array4<Real const> const& bcoef,
    Array4<Real const> const& ccoef,
    BoundCond bct,
    Real /*bcl*/,
    Array4<Real const> const& bcval,
    Real fac_vis,
    Real fac_conv,
    bool has_bcoef,
    bool has_ccoef,
    int icomp) noexcept
{
    if (bct == AMREX_LO_NEUMANN && mask(i, j, k) == 2) {
        Real b = (has_bcoef) ? bcoef(i, j, k, icomp) : Real(1.0);
        Real c = (has_ccoef) ? ccoef(i + 1, j, k, icomp) : Real(1.0);
        rhs(i - 1, j, k, icomp) +=
            (fac_vis * b * bcval(i, j, k, icomp) +
             fac_conv * c * bcval(i, j, k, icomp));
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mllinop_apply_innu_ylo_ic(
    int i,
    int j,
    int k,
    Array4<Real> const& rhs,
    Array4<int const> const& mask,
    Array4<Real const> const& bcoef,
    Array4<Real const> const& ccoef,
    BoundCond bct,
    Real /*bcl*/,
    Array4<Real const> const& bcval,
    Real fac_vis,
    Real fac_conv,
    bool has_bcoef,
    bool has_ccoef,
    int icomp) noexcept
{
    if (bct == AMREX_LO_NEUMANN && mask(i, j, k) == 2) {
        Real b = (has_bcoef) ? bcoef(i, j + 1, k, icomp) : Real(1.0);
        Real c = (has_ccoef) ? ccoef(i, j + 1, k, icomp) : Real(1.0);
        rhs(i, j + 1, k, icomp) -=
            (fac_vis * b * bcval(i, j, k, icomp) +
             fac_conv * c * bcval(i, j, k, icomp));
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mllinop_apply_innu_yhi_ic(
    int i,
    int j,
    int k,
    Array4<Real> const& rhs,
    Array4<int const> const& mask,
    Array4<Real const> const& bcoef,
    Array4<Real const> const& ccoef,
    BoundCond bct,
    Real /*bcl*/,
    Array4<Real const> const& bcval,
    Real fac_vis,
    Real fac_conv,
    bool has_bcoef,
    bool has_ccoef,
    int icomp) noexcept
{
    if (bct == AMREX_LO_NEUMANN && mask(i, j, k) == 2) {
        Real b = (has_bcoef) ? bcoef(i, j, k, icomp) : Real(1.0);
        Real c = (has_ccoef) ? ccoef(i, j, k, icomp) : Real(1.0);
        rhs(i, j - 1, k, icomp) +=
            (fac_vis * b * bcval(i, j, k, icomp) +
             fac_conv * c * bcval(i, j, k, icomp));
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mllinop_apply_innu_zlo_ic(
    int i,
    int j,
    int k,
    Array4<Real> const& rhs,
    Array4<int const> const& mask,
    Array4<Real const> const& bcoef,
    Array4<Real const> const& ccoef,
    BoundCond bct,
    Real /*bcl*/,
    Array4<Real const> const& bcval,
    Real fac_vis,
    Real fac_conv,
    bool has_bcoef,
    bool has_ccoef,
    int icomp) noexcept
{
    if (bct == AMREX_LO_NEUMANN && mask(i, j, k) == 2) {
        Real b = (has_bcoef) ? bcoef(i, j, k + 1, icomp) : Real(1.0);
        Real c = (has_ccoef) ? ccoef(i, j, k + 1, icomp) : Real(1.0);
        rhs(i, j, k + 1, icomp) -=
            (fac_vis * b * bcval(i, j, k, icomp) +
             fac_conv * c * bcval(i, j, k, icomp));
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mllinop_apply_innu_zhi_ic(
    int i,
    int j,
    int k,
    Array4<Real> const& rhs,
    Array4<int const> const& mask,
    Array4<Real const> const& bcoef,
    Array4<Real const> const& ccoef,
    BoundCond bct,
    Real /*bcl*/,
    Array4<Real const> const& bcval,
    Real fac_vis,
    Real fac_conv,
    bool has_bcoef,
    bool has_ccoef,
    int icomp) noexcept
{
    if (bct == AMREX_LO_NEUMANN && mask(i, j, k) == 2) {
        Real b = (has_bcoef) ? bcoef(i, j, k, icomp) : Real(1.0);
        Real c = (has_ccoef) ? ccoef(i, j, k, icomp) : Real(1.0);
        rhs(i, j, k - 1, icomp) +=
            (fac_vis * b * bcval(i, j, k, icomp) +
             fac_conv * c * bcval(i, j, k, icomp));
    }
}

} // namespace amrex

#endif
