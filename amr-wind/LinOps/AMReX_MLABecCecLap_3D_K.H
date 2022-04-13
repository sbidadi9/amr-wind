#ifndef AMREX_MLABECCECLAP_3D_K_H_
#define AMREX_MLABECCECLAP_3D_K_H_

namespace amrex {

constexpr Real eps = 1.0e-6;

inline Real getFaceStateUpwind(
    Real const& edgeVel, Real const& leftState, Real const& rightState)
{
    Real edstate = 0.0;
    if (edgeVel > eps) {
        edstate = leftState;
    }
    if (edgeVel < -eps) {
        edstate = rightState;
    }
    if (fabs(edgeVel) <= eps) {
        edstate = 0.5 * (leftState + rightState);
    }
    return edstate;
}

inline Real
getCellUpwindCoeff(Real const& leftedgeVel, Real const& rightedgeVel)
{

    Real coeff = 0.0;
    if (leftedgeVel * rightedgeVel <
        0.0) { // Case where velocity change sign, use centered + art. visc
        Real maxvel = std::max(fabs(leftedgeVel), fabs(rightedgeVel));
        coeff += 0.5 * (rightedgeVel - leftedgeVel); // centered part
        coeff += maxvel;                             // art. visc.
    } else { // same sign velocity, use upwind
        if (leftedgeVel < -eps)
            coeff -= leftedgeVel;
        else if (fabs(leftedgeVel) <= eps)
            coeff -= 0.5 * leftedgeVel;
        if (rightedgeVel > eps)
            coeff += rightedgeVel;
        else if (fabs(rightedgeVel) <= eps)
            coeff += 0.5 * rightedgeVel;
    }

    return coeff;
}

inline Real
getLeftCellUpwindCoeff(Real const& leftedgeVel, Real const& rightedgeVel)
{

    Real coeff = 0.0;
    if (leftedgeVel * rightedgeVel < 0.0) {
        Real maxvel = std::max(fabs(leftedgeVel), fabs(rightedgeVel));
        coeff -= 0.5 * leftedgeVel;
        coeff -= 0.5 * maxvel;
    } else {
        if (leftedgeVel > eps)
            coeff -= leftedgeVel;
        else if (fabs(leftedgeVel) < eps)
            coeff -= 0.5 * leftedgeVel;
    }
    return coeff;
}

inline Real
getRightCellUpwindCoeff(Real const& leftedgeVel, Real const& rightedgeVel)
{

    Real coeff = 0.0;
    if (leftedgeVel * rightedgeVel < 0.0) {
        Real maxvel = std::max(fabs(leftedgeVel), fabs(rightedgeVel));
        coeff += 0.5 * rightedgeVel;
        coeff += 0.5 * maxvel;
    } else {
        if (rightedgeVel < -eps)
            coeff += rightedgeVel;
        else if (fabs(rightedgeVel) < eps)
            coeff += 0.5 * rightedgeVel;
    }
    return coeff;
}

inline Real getCellUpwindCoeff(
    Real const& leftedgeVel,
    Real const& rightedgeVel,
    Real const& relaxleft,
    Real const& relaxright)
{

    Real coeff = 0.0;
    if (leftedgeVel * relaxleft * rightedgeVel * relaxright <
        0.0) { // Case where velocity change sign, use centered + art. visc
        Real maxvel = std::max(
            fabs(leftedgeVel * relaxleft), fabs(rightedgeVel * relaxright));
        coeff += 0.5 * (rightedgeVel * relaxright -
                        leftedgeVel * relaxleft); // centered part
        coeff += maxvel;                          // art. visc.
    } else { // same sign velocity, use upwind
        if (leftedgeVel < -eps)
            coeff -= leftedgeVel * relaxleft;
        else if (fabs(leftedgeVel) < eps)
            coeff -= 0.5 * leftedgeVel * relaxleft;
        if (rightedgeVel > eps)
            coeff += rightedgeVel * relaxright;
        else if (fabs(rightedgeVel) < eps)
            coeff += 0.5 * rightedgeVel * relaxright;
    }

    return coeff;
}

inline Real getFaceUpwindFlux(
    Real const& edgeVel, Real const& leftState, Real const& rightState)
{

    Real Flux = 0.0;
    if (edgeVel > eps)
        Flux = edgeVel * leftState;
    else if (edgeVel < -eps)
        Flux = edgeVel * rightState;
    else
        Flux = edgeVel * 0.5 * (leftState + rightState);
    return Flux;
}

inline Real incflo_xslope(
    int i,
    int j,
    int k,
    int n,
    amrex::Array4<amrex::Real const> const& vcc) noexcept
{
    amrex::Real dl = 2.0 * (vcc(i, j, k, n) - vcc(i - 1, j, k, n));
    amrex::Real dr = 2.0 * (vcc(i + 1, j, k, n) - vcc(i, j, k, n));
    amrex::Real dc = 0.5 * (vcc(i + 1, j, k, n) - vcc(i - 1, j, k, n));
    amrex::Real slope = amrex::min(
        amrex::Math::abs(dl), amrex::Math::abs(dc), amrex::Math::abs(dr));
    slope = (dr * dl > 0.0) ? slope : 0.0;
    return (dc > 0.0) ? slope : -slope;
}

inline Real incflo_yslope(
    int i,
    int j,
    int k,
    int n,
    amrex::Array4<amrex::Real const> const& vcc) noexcept
{
    amrex::Real dl = 2.0 * (vcc(i, j, k, n) - vcc(i, j - 1, k, n));
    amrex::Real dr = 2.0 * (vcc(i, j + 1, k, n) - vcc(i, j, k, n));
    amrex::Real dc = 0.5 * (vcc(i, j + 1, k, n) - vcc(i, j - 1, k, n));
    amrex::Real slope = amrex::min(
        amrex::Math::abs(dl), amrex::Math::abs(dc), amrex::Math::abs(dr));
    slope = (dr * dl > 0.0) ? slope : 0.0;
    return (dc > 0.0) ? slope : -slope;
}

inline Real incflo_zslope(
    int i,
    int j,
    int k,
    int n,
    amrex::Array4<amrex::Real const> const& vcc) noexcept
{
    amrex::Real dl = 2.0 * (vcc(i, j, k, n) - vcc(i, j, k - 1, n));
    amrex::Real dr = 2.0 * (vcc(i, j, k + 1, n) - vcc(i, j, k, n));
    amrex::Real dc = 0.5 * (vcc(i, j, k + 1, n) - vcc(i, j, k - 1, n));
    amrex::Real slope = amrex::min(
        amrex::Math::abs(dl), amrex::Math::abs(dc), amrex::Math::abs(dr));
    slope = (dr * dl > 0.0) ? slope : 0.0;
    return (dc > 0.0) ? slope : -slope;
}

std::pair<bool, bool> has_extdir_or_ho(amrex::BCRec const* bcrec, int dir)
{
    std::pair<bool, bool> r{false, false};

    r.first = r.first or (bcrec->lo(dir) == amrex::BCType::ext_dir) or
              (bcrec->lo(dir) == amrex::BCType::hoextrap);
    r.second = r.second or (bcrec->hi(dir) == amrex::BCType::ext_dir) or
               (bcrec->hi(dir) == amrex::BCType::hoextrap);
    return r;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mlabecceclap_adotx(
    Box const& box,
    Array4<Real> const& y,
    Array4<Real const> const& x,
    Array4<Real const> const& a,
    Array4<Real const> const& bX,
    Array4<Real const> const& bY,
    Array4<Real const> const& bZ,
    Array4<Real const> const& cX,
    Array4<Real const> const& cY,
    Array4<Real const> const& cZ,
    GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
    Real alpha,
    Real beta,
    Real eta,
    int ncomp,
    Real gradient_relax_factor,
    const BCRec* bc,
    const BCRec* d_bc,
    Geometry geom) noexcept
{
    const Real dhxsq = beta * dxinv[0] * dxinv[0];
    const Real dhysq = beta * dxinv[1] * dxinv[1];
    const Real dhzsq = beta * dxinv[2] * dxinv[2];
    const Real dhx = eta * dxinv[0];
    const Real dhy = eta * dxinv[1];
    const Real dhz = eta * dxinv[2];

    int nmaxcomp = AMREX_SPACEDIM;

    amrex::Box tmpbox = amrex::surroundingNodes(box);
    const int tmpcomp = nmaxcomp * AMREX_SPACEDIM;

    amrex::FArrayBox tmpfab(tmpbox, tmpcomp);
    amrex::Elixir eli = tmpfab.elixir();

    amrex::Array4<amrex::Real> fx = tmpfab.array(0);
    amrex::Array4<amrex::Real> fy = tmpfab.array(nmaxcomp);
    amrex::Array4<amrex::Real> fz = tmpfab.array(nmaxcomp * 2);

    const Box& domain_box = geom.Domain();

    const int domain_ilo = domain_box.smallEnd(0);
    const int domain_ihi = domain_box.bigEnd(0);
    const int domain_jlo = domain_box.smallEnd(1);
    const int domain_jhi = domain_box.bigEnd(1);
    const int domain_klo = domain_box.smallEnd(2);
    const int domain_khi = domain_box.bigEnd(2);

    Box const& xbx = amrex::surroundingNodes(box, 0);
    Box const& ybx = amrex::surroundingNodes(box, 1);
    Box const& zbx = amrex::surroundingNodes(box, 2);

    auto extdir_lohi = has_extdir_or_ho(bc, static_cast<int>(Direction::x));
    bool has_extdir_lo = extdir_lohi.first;
    bool has_extdir_hi = extdir_lohi.second;

    if ((has_extdir_lo and domain_ilo >= xbx.smallEnd(0) - 1) or
        (has_extdir_hi and domain_ihi <= xbx.bigEnd(0))) {

        amrex::ParallelFor(
            xbx, ncomp,
            [d_bc, x, domain_ilo, domain_ihi, cX,
             fx] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                Real qs;
                if (i <= domain_ilo && (d_bc->lo(0) == BCType::ext_dir)) {
                    qs = x(domain_ilo - 1, j, k, n);
                } else if (
                    i >= domain_ihi + 1 &&

                    (d_bc->hi(0) == BCType::ext_dir)) {
                    qs = x(domain_ihi + 1, j, k, n);
                } else {
                    Real qpls, qmns;

                    qpls = x(i, j, k, n);
                    qmns = x(i - 1, j, k, n);

                    if (cX(i, j, k, n) > eps) {
                        qs = qmns;
                    } else if (cX(i, j, k, n) < -eps) {
                        qs = qpls;
                    } else {
                        qs = 0.5 * (qmns + qpls);
                    }
                }
                fx(i, j, k, n) = qs * cX(i, j, k, n);
            });

    } else {
        amrex::ParallelFor(
            xbx, ncomp,
            [x, cX, fx, gradient_relax_factor] AMREX_GPU_DEVICE(
                int i, int j, int k, int n) noexcept {
                Real qpls, qmns;

                qpls = x(i, j, k, n) - 0.5 * gradient_relax_factor *
                                           incflo_xslope(i, j, k, n, x);
                qmns = x(i - 1, j, k, n) + 0.5 * gradient_relax_factor *
                                               incflo_xslope(i - 1, j, k, n, x);

                Real qs;
                if (cX(i, j, k, n) > eps) {
                    qs = qmns;
                } else if (cX(i, j, k, n) < -eps) {
                    qs = qpls;
                } else {
                    qs = 0.5 * (qmns + qpls);
                }
                fx(i, j, k, n) = qs * cX(i, j, k, n);
            });
    }

    extdir_lohi = has_extdir_or_ho(bc, static_cast<int>(Direction::y));
    has_extdir_lo = extdir_lohi.first;
    has_extdir_hi = extdir_lohi.second;

    if ((has_extdir_lo and domain_jlo >= ybx.smallEnd(1) - 1) or
        (has_extdir_hi and domain_jhi <= ybx.bigEnd(1))) {
        amrex::ParallelFor(
            ybx, ncomp,
            [d_bc, x, domain_jlo, domain_jhi, cY,
             fy] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                Real qs;
                if (j <= domain_jlo && (d_bc->lo(1) == BCType::ext_dir)) {
                    qs = x(i, domain_jlo - 1, k, n);
                } else if (
                    j >= domain_jhi + 1 && (d_bc->hi(1) == BCType::ext_dir)) {
                    qs = x(i, domain_jhi + 1, k, n);
                } else {
                    Real qpls, qmns;

                    qpls = x(i, j, k, n);
                    qmns = x(i, j - 1, k, n);

                    if (cY(i, j, k, n) > eps) {
                        qs = qmns;
                    } else if (cY(i, j, k, n) < -eps) {
                        qs = qpls;
                    } else {
                        qs = 0.5 * (qmns + qpls);
                    }
                }
                fy(i, j, k, n) = qs * cY(i, j, k, n);
            });
    } else {
        amrex::ParallelFor(
            ybx, ncomp,
            [x, cY, fy, gradient_relax_factor] AMREX_GPU_DEVICE(
                int i, int j, int k, int n) noexcept {
                Real qpls, qmns;

                qpls = x(i, j, k, n) - 0.5 * gradient_relax_factor *
                                           incflo_yslope(i, j, k, n, x);
                qmns = x(i, j - 1, k, n) + 0.5 * gradient_relax_factor *
                                               incflo_yslope(i, j - 1, k, n, x);

                Real qs;
                if (cY(i, j, k, n) > eps) {
                    qs = qmns;
                } else if (cY(i, j, k, n) < -eps) {
                    qs = qpls;
                } else {
                    qs = 0.5 * (qmns + qpls);
                }
                fy(i, j, k, n) = qs * cY(i, j, k, n);
            });
    }

    extdir_lohi = has_extdir_or_ho(bc, static_cast<int>(Direction::z));
    has_extdir_lo = extdir_lohi.first;
    has_extdir_hi = extdir_lohi.second;

    if ((has_extdir_lo and domain_jlo >= zbx.smallEnd(2) - 1) or
        (has_extdir_hi and domain_jhi <= zbx.bigEnd(2))) {
        amrex::ParallelFor(
            zbx, ncomp,
            [d_bc, x, domain_klo, domain_khi, cZ,
             fz] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                Real qs;
                if (k <= domain_klo && (d_bc->lo(2) == BCType::ext_dir)) {
                    qs = x(i, j, domain_klo - 1, n);
                } else if (
                    k >= domain_khi + 1 && (d_bc->hi(2) == BCType::ext_dir)) {
                    qs = x(i, j, domain_khi + 1, n);
                } else {
                    Real qpls, qmns;

                    qpls = x(i, j, k, n);
                    qmns = x(i, j, k - 1, n);

                    if (cZ(i, j, k, n) > eps) {
                        qs = qmns;
                    } else if (cZ(i, j, k, n) < -eps) {
                        qs = qpls;
                    } else {
                        qs = 0.5 * (qmns + qpls);
                    }
                }
                fz(i, j, k, n) = qs * cZ(i, j, k, n);
            });
    } else {
        amrex::ParallelFor(
            zbx, ncomp,
            [x, cZ, fz, gradient_relax_factor] AMREX_GPU_DEVICE(
                int i, int j, int k, int n) noexcept {
                Real qpls, qmns;

                qpls = x(i, j, k, n) - 0.5 * gradient_relax_factor *
                                           incflo_zslope(i, j, k, n, x);
                qmns = x(i, j, k - 1, n) + 0.5 * gradient_relax_factor *
                                               incflo_zslope(i, j, k - 1, n, x);

                Real qs;
                if (cZ(i, j, k, n) > eps) {
                    qs = qmns;
                } else if (cZ(i, j, k, n) < -eps) {
                    qs = qpls;
                } else {
                    qs = 0.5 * (qmns + qpls);
                }
                fz(i, j, k, n) = qs * cZ(i, j, k, n);
            });
    }

    amrex::ParallelFor(
        box, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            Real conv_term;
            conv_term =
                dhx * a(i, j, k) * (fx(i + 1, j, k, n) - fx(i, j, k, n)) +
                dhy * a(i, j, k) * (fy(i, j + 1, k, n) - fy(i, j, k, n)) +
                dhz * a(i, j, k) * (fz(i, j, k + 1, n) - fz(i, j, k, n));

            y(i, j, k, n) =
                alpha * a(i, j, k) * x(i, j, k, n) -
                dhxsq *
                    (bX(i + 1, j, k, n) * (x(i + 1, j, k, n) - x(i, j, k, n)) -
                     bX(i, j, k, n) * (x(i, j, k, n) - x(i - 1, j, k, n))) -
                dhysq *
                    (bY(i, j + 1, k, n) * (x(i, j + 1, k, n) - x(i, j, k, n)) -
                     bY(i, j, k, n) * (x(i, j, k, n) - x(i, j - 1, k, n))) -
                dhzsq *
                    (bZ(i, j, k + 1, n) * (x(i, j, k + 1, n) - x(i, j, k, n)) -
                     bZ(i, j, k, n) * (x(i, j, k, n) - x(i, j, k - 1, n))) -
                conv_term;
        });
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mlabecceclap_adotx_os(
    Box const& box,
    Array4<Real> const& y,
    Array4<Real const> const& x,
    Array4<Real const> const& a,
    Array4<Real const> const& bX,
    Array4<Real const> const& bY,
    Array4<Real const> const& bZ,
    Array4<Real const> const& cX,
    Array4<Real const> const& cY,
    Array4<Real const> const& cZ,
    Array4<int const> const& osm,
    GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
    Real alpha,
    Real beta,
    Real eta,
    int ncomp) noexcept
{
    const Real dhxsq = beta * dxinv[0] * dxinv[0];
    const Real dhysq = beta * dxinv[1] * dxinv[1];
    const Real dhzsq = beta * dxinv[2] * dxinv[2];
    const Real dhx = eta * dxinv[0];
    const Real dhy = eta * dxinv[1];
    const Real dhz = eta * dxinv[2];

    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);

    for (int n = 0; n < ncomp; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    if (osm(i, j, k) == 0) {
                        y(i, j, k, n) = 0.0;
                    } else {
                        y(i, j, k, n) =
                            alpha * a(i, j, k) * x(i, j, k, n) -
                            dhxsq * (bX(i + 1, j, k, n) *
                                         (x(i + 1, j, k, n) - x(i, j, k, n)) -
                                     bX(i, j, k, n) *
                                         (x(i, j, k, n) - x(i - 1, j, k, n))) -
                            dhysq * (bY(i, j + 1, k, n) *
                                         (x(i, j + 1, k, n) - x(i, j, k, n)) -
                                     bY(i, j, k, n) *
                                         (x(i, j, k, n) - x(i, j - 1, k, n))) -
                            dhzsq * (bZ(i, j, k + 1, n) *
                                         (x(i, j, k + 1, n) - x(i, j, k, n)) -
                                     bZ(i, j, k, n) *
                                         (x(i, j, k, n) - x(i, j, k - 1, n))) -
                            dhx * (cX(i + 1, j, k, n) * getFaceStateUpwind(
                                                            cX(i + 1, j, k, n),
                                                            x(i, j, k, n),
                                                            x(i + 1, j, k, n)) -
                                   cX(i, j, k, n) * getFaceStateUpwind(
                                                        cX(i, j, k, n),
                                                        x(i - 1, j, k, n),
                                                        x(i, j, k, n))) -
                            dhy * (cY(i, j + 1, k, n) * getFaceStateUpwind(
                                                            cY(i, j + 1, k, n),
                                                            x(i, j, k, n),
                                                            x(i, j + 1, k, n)) -
                                   cY(i, j, k, n) * getFaceStateUpwind(
                                                        cY(i, j, k, n),
                                                        x(i, j - 1, k, n),
                                                        x(i, j, k, n))) -
                            dhz * (cZ(i, j, k + 1, n) * getFaceStateUpwind(
                                                            cZ(i, j, k + 1, n),
                                                            x(i, j, k, n),
                                                            x(i, j, k + 1, n)) -
                                   cZ(i, j, k, n) * getFaceStateUpwind(
                                                        cZ(i, j, k, n),
                                                        x(i, j, k - 1, n),
                                                        x(i, j, k, n)));
                    }
                }
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mlabecceclap_normalize(
    Box const& box,
    Array4<Real> const& x,
    Array4<Real const> const& a,
    Array4<Real const> const& bX,
    Array4<Real const> const& bY,
    Array4<Real const> const& bZ,
    Array4<Real const> const& cX,
    Array4<Real const> const& cY,
    Array4<Real const> const& cZ,
    GpuArray<Real, AMREX_SPACEDIM> const& dxinv,
    Real alpha,
    Real beta,
    Real eta,
    int ncomp) noexcept
{
    const Real dhxsq = beta * dxinv[0] * dxinv[0];
    const Real dhysq = beta * dxinv[1] * dxinv[1];
    const Real dhzsq = beta * dxinv[2] * dxinv[2];
    const Real dhx = eta * dxinv[0];
    const Real dhy = eta * dxinv[1];
    const Real dhz = eta * dxinv[2];

    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);

    for (int n = 0; n < ncomp; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    x(i, j, k, n) /=
                        alpha * a(i, j, k) +
                        dhxsq * (bX(i, j, k, n) + bX(i + 1, j, k, n)) +
                        dhysq * (bY(i, j, k, n) + bY(i, j + 1, k, n)) +
                        dhzsq * (bZ(i, j, k, n) + bZ(i, j, k + 1, n)) -
                        dhx * a(i, j, k) *
                            getCellUpwindCoeff(
                                cX(i, j, k, n), cX(i + 1, j, k, n)) -
                        dhy * a(i, j, k) *
                            getCellUpwindCoeff(
                                cY(i, j, k, n), cY(i, j + 1, k, n)) -
                        dhz * a(i, j, k) *
                            getCellUpwindCoeff(
                                cZ(i, j, k, n), cZ(i, j, k + 1, n));
                }
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mlabecceclap_flux_x(
    Box const& box,
    Array4<Real> const& fx,
    Array4<Real const> const& sol,
    Array4<Real const> const& bx,
    Array4<Real const> const& cx,
    Real fac,
    Real facc,
    int ncomp) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);

    for (int n = 0; n < ncomp; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    fx(i, j, k, n) =
                        -fac * bx(i, j, k, n) *
                            (sol(i, j, k, n) - sol(i - 1, j, k, n)) -
                        facc * getFaceUpwindFlux(
                                   cx(i, j, k, n), sol(i - 1, j, k, n),
                                   sol(i, j, k, n));
                }
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mlabecceclap_flux_xface(
    Box const& box,
    Array4<Real> const& fx,
    Array4<Real const> const& sol,
    Array4<Real const> const& bx,
    Array4<Real const> const& cx,
    Real fac,
    Real facc,
    int xlen,
    int ncomp) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);

    for (int n = 0; n < ncomp; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                int i = lo.x;
                fx(i, j, k, n) = -fac * bx(i, j, k, n) *
                                     (sol(i, j, k, n) - sol(i - 1, j, k, n)) -
                                 facc * getFaceUpwindFlux(
                                            cx(i, j, k, n), sol(i - 1, j, k, n),
                                            sol(i, j, k, n));
                i += xlen;
                fx(i, j, k, n) = -fac * bx(i, j, k, n) *
                                     (sol(i, j, k, n) - sol(i - 1, j, k, n)) -
                                 facc * getFaceUpwindFlux(
                                            cx(i, j, k, n), sol(i - 1, j, k, n),
                                            sol(i, j, k, n));
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mlabecceclap_flux_y(
    Box const& box,
    Array4<Real> const& fy,
    Array4<Real const> const& sol,
    Array4<Real const> const& by,
    Array4<Real const> const& cy,
    Real fac,
    Real facc,
    int ncomp) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);

    for (int n = 0; n < ncomp; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    fy(i, j, k, n) =
                        -fac * by(i, j, k, n) *
                            (sol(i, j, k, n) - sol(i, j - 1, k, n)) -
                        facc * getFaceUpwindFlux(
                                   cy(i, j, k, n), sol(i, j - 1, k, n),
                                   sol(i, j, k, n));
                }
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mlabecceclap_flux_yface(
    Box const& box,
    Array4<Real> const& fy,
    Array4<Real const> const& sol,
    Array4<Real const> const& by,
    Array4<Real const> const& cy,
    Real fac,
    Real facc,
    int ylen,
    int ncomp) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);

    for (int n = 0; n < ncomp; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            int j = lo.y;
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) {
                fy(i, j, k, n) = -fac * by(i, j, k, n) *
                                     (sol(i, j, k, n) - sol(i, j - 1, k, n)) -
                                 facc * getFaceUpwindFlux(
                                            cy(i, j, k, n), sol(i, j - 1, k, n),
                                            sol(i, j, k, n));
            }
            j += ylen;
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) {
                fy(i, j, k, n) = -fac * by(i, j, k, n) *
                                     (sol(i, j, k, n) - sol(i, j - 1, k, n)) -
                                 facc * getFaceUpwindFlux(
                                            cy(i, j, k, n), sol(i, j - 1, k, n),
                                            sol(i, j, k, n));
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mlabecceclap_flux_z(
    Box const& box,
    Array4<Real> const& fz,
    Array4<Real const> const& sol,
    Array4<Real const> const& bz,
    Array4<Real const> const& cz,
    Real fac,
    Real facc,
    int ncomp) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);

    for (int n = 0; n < ncomp; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    fz(i, j, k, n) =
                        -fac * bz(i, j, k, n) *
                            (sol(i, j, k, n) - sol(i, j, k - 1, n)) -
                        facc * getFaceUpwindFlux(
                                   cz(i, j, k, n), sol(i, j, k - 1, n),
                                   sol(i, j, k, n));
                }
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void mlabecceclap_flux_zface(
    Box const& box,
    Array4<Real> const& fz,
    Array4<Real const> const& sol,
    Array4<Real const> const& bz,
    Array4<Real const> const& cz,
    Real fac,
    Real facc,
    int zlen,
    int ncomp) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);

    for (int n = 0; n < ncomp; ++n) {
        int k = lo.z;
        for (int j = lo.y; j <= hi.y; ++j) {
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) {
                fz(i, j, k, n) = -fac * bz(i, j, k, n) *
                                     (sol(i, j, k, n) - sol(i, j, k - 1, n)) -
                                 facc * getFaceUpwindFlux(
                                            cz(i, j, k, n), sol(i, j, k - 1, n),
                                            sol(i, j, k, n));
            }
        }

        k += zlen;
        for (int j = lo.y; j <= hi.y; ++j) {
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) {
                fz(i, j, k, n) = -fac * bz(i, j, k, n) *
                                     (sol(i, j, k, n) - sol(i, j, k - 1, n)) -
                                 facc * getFaceUpwindFlux(
                                            cz(i, j, k, n), sol(i, j, k - 1, n),
                                            sol(i, j, k, n));
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void abeccec_gsrb(
    Box const& box,
    Array4<Real> const& phi,
    Array4<Real const> const& rhs,
    Real alpha,
    Real omega,
    Array4<Real const> const& a,
    Real dhxsq,
    Real dhysq,
    Real dhzsq,
    Real dhx,
    Real dhy,
    Real dhz,
    Array4<Real const> const& bX,
    Array4<Real const> const& bY,
    Array4<Real const> const& bZ,
    Array4<Real const> const& cX,
    Array4<Real const> const& cY,
    Array4<Real const> const& cZ,
    Array4<int const> const& m0,
    Array4<int const> const& m2,
    Array4<int const> const& m4,
    Array4<int const> const& m1,
    Array4<int const> const& m3,
    Array4<int const> const& m5,
    Array4<Real const> const& f0,
    Array4<Real const> const& f2,
    Array4<Real const> const& f4,
    Array4<Real const> const& f1,
    Array4<Real const> const& f3,
    Array4<Real const> const& f5,
    Box const& vbox,
    int redblack,
    int nc) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
    const auto vlo = amrex::lbound(vbox);
    const auto vhi = amrex::ubound(vbox);

    for (int n = 0; n < nc; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    if ((i + j + k + redblack) % 2 == 0) {
                        Real cf0 = (i == vlo.x and m0(vlo.x - 1, j, k) > 0)
                                       ? f0(vlo.x, j, k, n)
                                       : 0.0;
                        Real cf1 = (j == vlo.y and m1(i, vlo.y - 1, k) > 0)
                                       ? f1(i, vlo.y, k, n)
                                       : 0.0;
                        Real cf2 = (k == vlo.z and m2(i, j, vlo.z - 1) > 0)
                                       ? f2(i, j, vlo.z, n)
                                       : 0.0;
                        Real cf3 = (i == vhi.x and m3(vhi.x + 1, j, k) > 0)
                                       ? f3(vhi.x, j, k, n)
                                       : 0.0;
                        Real cf4 = (j == vhi.y and m4(i, vhi.y + 1, k) > 0)
                                       ? f4(i, vhi.y, k, n)
                                       : 0.0;
                        Real cf5 = (k == vhi.z and m5(i, j, vhi.z + 1) > 0)
                                       ? f5(i, j, vhi.z, n)
                                       : 0.0;

                        Real gamma =
                            alpha * a(i, j, k) +
                            dhxsq * (bX(i, j, k, n) + bX(i + 1, j, k, n)) +
                            dhysq * (bY(i, j, k, n) + bY(i, j + 1, k, n)) +
                            dhzsq * (bZ(i, j, k, n) + bZ(i, j, k + 1, n)) -
                            dhx * a(i, j, k) *
                                getCellUpwindCoeff(
                                    cX(i, j, k, n), cX(i + 1, j, k, n)) -
                            dhy * a(i, j, k) *
                                getCellUpwindCoeff(
                                    cY(i, j, k, n), cY(i, j + 1, k, n)) -
                            dhz * a(i, j, k) *
                                getCellUpwindCoeff(
                                    cZ(i, j, k, n), cZ(i, j, k + 1, n));

                        Real g_m_d =
                            gamma - (dhxsq * (bX(i, j, k, n) * cf0 +
                                              bX(i + 1, j, k, n) * cf3) +
                                     dhysq * (bY(i, j, k, n) * cf1 +
                                              bY(i, j + 1, k, n) * cf4) +
                                     dhzsq * (bZ(i, j, k, n) * cf2 +
                                              bZ(i, j, k + 1, n) * cf5));

                        Real rho =
                            dhxsq * (bX(i, j, k, n) * phi(i - 1, j, k, n) +
                                     bX(i + 1, j, k, n) * phi(i + 1, j, k, n)) +
                            dhysq * (bY(i, j, k, n) * phi(i, j - 1, k, n) +
                                     bY(i, j + 1, k, n) * phi(i, j + 1, k, n)) +
                            dhzsq * (bZ(i, j, k, n) * phi(i, j, k - 1, n) +
                                     bZ(i, j, k + 1, n) * phi(i, j, k + 1, n)) +
                            dhx * a(i, j, k) *
                                (getLeftCellUpwindCoeff(
                                     cX(i, j, k, n), cX(i + 1, j, k, n)) *
                                     phi(i - 1, j, k, n) +
                                 getRightCellUpwindCoeff(
                                     cX(i, j, k, n), cX(i + 1, j, k, n)) *
                                     phi(i + 1, j, k, n)) +
                            dhy * a(i, j, k) *
                                (getLeftCellUpwindCoeff(
                                     cY(i, j, k, n), cY(i, j + 1, k, n)) *
                                     phi(i, j - 1, k, n) +
                                 getRightCellUpwindCoeff(
                                     cY(i, j, k, n), cY(i, j + 1, k, n)) *
                                     phi(i, j + 1, k, n)) +
                            dhz * a(i, j, k) *
                                (getLeftCellUpwindCoeff(
                                     cZ(i, j, k, n), cZ(i, j, k + 1, n)) *
                                     phi(i, j, k - 1, n) +
                                 getRightCellUpwindCoeff(
                                     cZ(i, j, k, n), cZ(i, j, k + 1, n)) *
                                     phi(i, j, k + 1, n));

                        Real res =
                            rhs(i, j, k, n) - (gamma * phi(i, j, k, n) - rho);
                        phi(i, j, k, n) = phi(i, j, k, n) + omega / g_m_d * res;
                    }
                }
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void abeccec_gsrb_os(
    Box const& box,
    Array4<Real> const& phi,
    Array4<Real const> const& rhs,
    Real alpha,
    Real omega,
    Array4<Real const> const& a,
    Real dhxsq,
    Real dhysq,
    Real dhzsq,
    Real dhx,
    Real dhy,
    Real dhz,
    Array4<Real const> const& bX,
    Array4<Real const> const& bY,
    Array4<Real const> const& bZ,
    Array4<Real const> const& cX,
    Array4<Real const> const& cY,
    Array4<Real const> const& cZ,
    Array4<int const> const& m0,
    Array4<int const> const& m2,
    Array4<int const> const& m4,
    Array4<int const> const& m1,
    Array4<int const> const& m3,
    Array4<int const> const& m5,
    Array4<Real const> const& f0,
    Array4<Real const> const& f2,
    Array4<Real const> const& f4,
    Array4<Real const> const& f1,
    Array4<Real const> const& f3,
    Array4<Real const> const& f5,
    Array4<int const> const& osm,
    Box const& vbox,
    int redblack,
    int nc) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
    const auto vlo = amrex::lbound(vbox);
    const auto vhi = amrex::ubound(vbox);

    for (int n = 0; n < nc; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    if ((i + j + k + redblack) % 2 == 0) {
                        if (osm(i, j, k) == 0) {
                            phi(i, j, k, n) = 0.0;
                        } else {
                            Real cf0 = (i == vlo.x and m0(vlo.x - 1, j, k) > 0)
                                           ? f0(vlo.x, j, k, n)
                                           : 0.0;
                            Real cf1 = (j == vlo.y and m1(i, vlo.y - 1, k) > 0)
                                           ? f1(i, vlo.y, k, n)
                                           : 0.0;
                            Real cf2 = (k == vlo.z and m2(i, j, vlo.z - 1) > 0)
                                           ? f2(i, j, vlo.z, n)
                                           : 0.0;
                            Real cf3 = (i == vhi.x and m3(vhi.x + 1, j, k) > 0)
                                           ? f3(vhi.x, j, k, n)
                                           : 0.0;
                            Real cf4 = (j == vhi.y and m4(i, vhi.y + 1, k) > 0)
                                           ? f4(i, vhi.y, k, n)
                                           : 0.0;
                            Real cf5 = (k == vhi.z and m5(i, j, vhi.z + 1) > 0)
                                           ? f5(i, j, vhi.z, n)
                                           : 0.0;

                            Real gamma =
                                alpha * a(i, j, k) +
                                dhxsq * (bX(i, j, k, n) + bX(i + 1, j, k, n)) +
                                dhysq * (bY(i, j, k, n) + bY(i, j + 1, k, n)) +
                                dhzsq * (bZ(i, j, k, n) + bZ(i, j, k + 1, n)) -
                                dhx * getCellUpwindCoeff(
                                          cX(i, j, k, n), cX(i + 1, j, k, n)) -
                                dhy * getCellUpwindCoeff(
                                          cY(i, j, k, n), cY(i, j + 1, k, n)) -
                                dhz * getCellUpwindCoeff(
                                          cZ(i, j, k, n), cZ(i, j, k + 1, n));

                            Real g_m_d =
                                gamma - (dhxsq * (bX(i, j, k, n) * cf0 +
                                                  bX(i + 1, j, k, n) * cf3) +
                                         dhysq * (bY(i, j, k, n) * cf1 +
                                                  bY(i, j + 1, k, n) * cf4) +
                                         dhzsq * (bZ(i, j, k, n) * cf2 +
                                                  bZ(i, j, k + 1, n) * cf5));

                            Real rho =
                                dhxsq *
                                    (bX(i, j, k, n) * phi(i - 1, j, k, n) +
                                     bX(i + 1, j, k, n) * phi(i + 1, j, k, n)) +
                                dhysq *
                                    (bY(i, j, k, n) * phi(i, j - 1, k, n) +
                                     bY(i, j + 1, k, n) * phi(i, j + 1, k, n)) +
                                dhzsq *
                                    (bZ(i, j, k, n) * phi(i, j, k - 1, n) +
                                     bZ(i, j, k + 1, n) * phi(i, j, k + 1, n)) +
                                dhx * (getLeftCellUpwindCoeff(
                                           cX(i, j, k, n), cX(i + 1, j, k, n)) *
                                           phi(i - 1, j, k, n) +
                                       getRightCellUpwindCoeff(
                                           cX(i, j, k, n), cX(i + 1, j, k, n)) *
                                           phi(i + 1, j, k, n)) +
                                dhy * (getLeftCellUpwindCoeff(
                                           cY(i, j, k, n), cY(i, j + 1, k, n)) *
                                           phi(i, j - 1, k, n) +
                                       getRightCellUpwindCoeff(
                                           cY(i, j, k, n), cY(i, j + 1, k, n)) *
                                           phi(i, j + 1, k, n)) +
                                dhz * (getLeftCellUpwindCoeff(
                                           cZ(i, j, k, n), cZ(i, j, k + 1, n)) *
                                           phi(i, j, k - 1, n) +
                                       getRightCellUpwindCoeff(
                                           cZ(i, j, k, n), cZ(i, j, k + 1, n)) *
                                           phi(i, j, k + 1, n));

                            Real res = rhs(i, j, k, n) -
                                       (gamma * phi(i, j, k, n) - rho);
                            phi(i, j, k, n) =
                                phi(i, j, k, n) + omega / g_m_d * res;
                        }
                    }
                }
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void tridiagonal_solve(
    Array1D<Real, 0, 31>& a_ls,
    Array1D<Real, 0, 31>& b_ls,
    Array1D<Real, 0, 31>& c_ls,
    Array1D<Real, 0, 31>& r_ls,
    Array1D<Real, 0, 31>& u_ls,
    Array1D<Real, 0, 31>& gam,
    int ilen) noexcept
{
    Real bet = b_ls(0);
    u_ls(0) = r_ls(0) / bet;

    for (int i = 1; i <= ilen - 1; i++) {
        gam(i) = c_ls(i - 1) / bet;
        bet = b_ls(i) - a_ls(i) * gam(i);
        if (bet == 0) amrex::Abort(">>>TRIDIAG FAILED");
        u_ls(i) = (r_ls(i) - a_ls(i) * u_ls(i - 1)) / bet;
    }
    for (int i = ilen - 2; i >= 0; i--) {
        u_ls(i) = u_ls(i) - gam(i + 1) * u_ls(i + 1);
    }
}

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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void overset_rescale_bcoef_x(
    Box const& box,
    Array4<Real> const& bX,
    Array4<int const> const& osm,
    int ncomp,
    Real osfac) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
    for (int n = 0; n < ncomp; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    if ((osm(i - 1, j, k) + osm(i, j, k)) == 1) {
                        bX(i, j, k, n) *= osfac;
                    }
                }
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void overset_rescale_bcoef_y(
    Box const& box,
    Array4<Real> const& bY,
    Array4<int const> const& osm,
    int ncomp,
    Real osfac) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
    for (int n = 0; n < ncomp; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    if ((osm(i, j - 1, k) + osm(i, j, k)) == 1) {
                        bY(i, j, k, n) *= osfac;
                    }
                }
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void overset_rescale_bcoef_z(
    Box const& box,
    Array4<Real> const& bZ,
    Array4<int const> const& osm,
    int ncomp,
    Real osfac) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);
    for (int n = 0; n < ncomp; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    if ((osm(i, j, k - 1) + osm(i, j, k)) == 1) {
                        bZ(i, j, k, n) *= osfac;
                    }
                }
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE void abeccec_chkdiag(
    Box const& box,
    Real alpha,
    Array4<Real const> const& a,
    Real dhxsq,
    Real dhysq,
    Real dhzsq,
    Real dhx,
    Real dhy,
    Real dhz,
    Array4<Real const> const& bX,
    Array4<Real const> const& bY,
    Array4<Real const> const& bZ,
    Array4<Real const> const& cX,
    Array4<Real const> const& cY,
    Array4<Real const> const& cZ,
    int nc) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);

    for (int n = 0; n < nc; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    // Complete diag
                    Real diag = alpha * a(i, j, k) -
                                dhxsq * (-bX(i, j, k, n) - bX(i + 1, j, k, n)) -
                                dhysq * (-bY(i, j, k, n) - bY(i, j + 1, k, n)) -
                                dhzsq * (-bZ(i, j, k, n) - bZ(i, j, k + 1, n)) -
                                dhx * getCellUpwindCoeff(
                                          cX(i, j, k, n), cX(i + 1, j, k, n)) -
                                dhy * getCellUpwindCoeff(
                                          cY(i, j, k, n), cY(i, j + 1, k, n)) -
                                dhz * getCellUpwindCoeff(
                                          cZ(i, j, k, n), cZ(i, j, k + 1, n));

                    // Offdiag
                    Real offdiagdiff =
                        -dhxsq * (bX(i, j, k, n) + bX(i + 1, j, k, n)) -
                        dhysq * (bY(i, j, k, n) + bY(i, j + 1, k, n)) -
                        dhzsq * (bZ(i, j, k, n) + bZ(i, j, k + 1, n));

                    Real offdiagadv =
                        -dhx * (getLeftCellUpwindCoeff(
                                    cX(i, j, k, n), cX(i + 1, j, k, n)) +
                                getRightCellUpwindCoeff(
                                    cX(i, j, k, n), cX(i + 1, j, k, n))) -
                        dhy * (getLeftCellUpwindCoeff(
                                   cY(i, j, k, n), cY(i, j + 1, k, n)) +
                               getRightCellUpwindCoeff(
                                   cY(i, j, k, n), cY(i, j + 1, k, n))) -
                        dhz * (getLeftCellUpwindCoeff(
                                   cZ(i, j, k, n), cZ(i, j, k + 1, n)) +
                               getRightCellUpwindCoeff(
                                   cZ(i, j, k, n), cZ(i, j, k + 1, n)));

                    if (fabs(offdiagdiff + offdiagadv) > fabs(diag)) {
                        amrex::Print()
                            << " Abort LinOp not diagonal dominant at: " << i
                            << " " << j << " " << k << "\n";
                        // amrex::Print() << j << " Val diag: " << diag << ",
                        // offdiagdiff/adv: " << offdiagdiff << " " <<
                        // offdiagadv << "\n"; amrex::Print() << j << " Val
                        // diag: " << diag << ", left/right: " << left << "  "
                        // << right << "\n"; amrex::Print() << j << " " <<
                        // fabs(diag) << " " << fabs(offdiagdiff+offdiagadv) <<
                        // "\n"; amrex::Abort("MLABecCec is not diagonal
                        // dominant !!!");
                    }
                }
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE void abeccec_chkdiag(
    Box const& box,
    Real alpha,
    Array4<Real const> const& a,
    Real dhxsq,
    Real dhysq,
    Real dhzsq,
    Real dhx,
    Real dhy,
    Real dhz,
    Array4<Real const> const& bX,
    Array4<Real const> const& bY,
    Array4<Real const> const& bZ,
    Array4<Real const> const& cX,
    Array4<Real const> const& cY,
    Array4<Real const> const& cZ,
    int nc,
    Array4<Real> const& diag) noexcept
{
    const auto lo = amrex::lbound(box);
    const auto hi = amrex::ubound(box);

    for (int n = 0; n < nc; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    diag(i, j, k) =
                        alpha * a(i, j, k) -
                        dhxsq * (-bX(i, j, k, n) - bX(i + 1, j, k, n)) -
                        dhysq * (-bY(i, j, k, n) - bY(i, j + 1, k, n)) -
                        dhzsq * (-bZ(i, j, k, n) - bZ(i, j, k + 1, n)) -
                        dhx * getCellUpwindCoeff(
                                  cX(i, j, k, n), cX(i + 1, j, k, n)) -
                        dhy * getCellUpwindCoeff(
                                  cY(i, j, k, n), cY(i, j + 1, k, n)) -
                        dhz * getCellUpwindCoeff(
                                  cZ(i, j, k, n), cZ(i, j, k + 1, n));
                }
            }
        }
    }
}

} // namespace amrex
#endif
