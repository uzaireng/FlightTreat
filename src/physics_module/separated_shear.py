"""Separated shear layer and laminar-bubble transition.

This is not an attached-flow closure and it is not a stall correction.
It is used only after a resolved station has already met a separation
criterion. The attached march is not changed on stations that stay attached.

Two regimes are represented.

Laminar free shear, from Thwaites separation until transition in the
shear layer. Wall shear is zero because the layer has left the wall.
The shape factor is held at the separated laminar value: Thwaites' H(λ)
is an attached correlation and is not integrated past λ = -0.09.
Momentum thickness follows the momentum integral with Cf = 0, plus the
zero-pressure laminar growth d(θ²)/ds = 0.45 ν/Ue. That growth is the
same diffusion scale as Thwaites at λ = 0. It is not a second empirical
drag term.

Transition in that shear layer uses Mayle's short-bubble correlation
(ASME Journal of Turbomachinery, 1991):

    Re_s,tr = 700 Re_θ,s^0.7

with Re_s,tr based on the arc length from separation to transition.
This is not Michel's attached criterion and it is not an e^n envelope.

At that station an attached turbulent restart is attempted from a shape
factor of 2.5 when the separated laminar value is higher. Horton's
short-bubble measurements put reattachment near that shape factor. The
value is the initial condition of the attempt, not a force adjustment.
If the attached closure separates on its first resolved step, the
attempt is discarded and the separated turbulent shear layer is marched
from the laminar separated state instead.

Turbulent separated shear. Head's H1(H) correlation is not used above
H = 2.8, where its entrainment function is approaching a singularity.
Entrainment is held at Head's F(H1) evaluated at H = 2.8, the top of
the correlation that the attached march already uses. With Cf = 0,

    dθ/ds = -(H + 2) (θ/Ue) dUe/ds
    d(Ue δ*)/ds = F(H = 2.8) Ue

and H = δ*/θ. Reversed-flow wall shear is not resolved, so Cf stays
zero rather than being invented as a thrust. A direct march of these
two equations does not contain a turbulent reattachment mechanism.
Reattachment occurs only when the attached restart succeeds.

If H leaves [1.05, 8], or a step is not finite and positive, the model
has left the range in which this entrainment assumption is defensible.
The last finite state is held and the caller reports that limit. The
hold is not a prediction that the layer stopped growing.
"""

from __future__ import annotations

import numpy as np

# Mayle (1991), short separation bubble. Long bubbles are a different
# correlation; this code does not switch on an angle of attack.
_MAYLE_SHORT_BUBBLE = 700.0

# Initial shape factor of an attached turbulent restart. Horton's
# short-bubble reattachment sits near 2.5. The restart used here begins
# at 2.75, just inside Head's correlation, so the drop from a separated
# laminar value near 3.0 is one correlation step rather than a jump to
# 2.5. Head then brings H down if the layer reattaches. Discarded if
# that restart separates on its first resolved step.
REATTACHMENT_SHAPE = 2.75

_SHAPE_MIN = 1.05
_SHAPE_MAX = 8.0


def head_entrainment(shape_h: float) -> float:
    """Head's entrainment function F(H1) on the attached correlation.

    The coefficients are the ones in the Head march. H is limited to the
    interval on which that correlation was used, [1.15, 2.8].
    """

    h = float(np.clip(shape_h, 1.15, 2.8))
    h1 = 3.3 + 0.8234 * (h - 1.1) ** (-1.287)
    h1 = max(h1, 3.05)
    return float(0.0306 * (max(h1 - 3.0, 1.0e-6)) ** (-0.6169))


def separated_entrainment() -> float:
    """Entrainment held at the top of Head's attached correlation."""

    return head_entrainment(2.8)


def mayle_transition_distance(theta: float, edge_speed: float, nu: float) -> float:
    """Arc length from laminar separation to free-shear transition.

    Returns a non-negative length. A non-physical Reynolds number gives
    ``nan`` so the caller can keep the laminar shear layer.
    """

    if not np.isfinite(theta) or not np.isfinite(edge_speed) or not np.isfinite(nu):
        return float("nan")
    if theta <= 0.0 or edge_speed <= 0.0 or nu <= 0.0:
        return float("nan")
    re_theta = edge_speed * theta / nu
    re_length = _MAYLE_SHORT_BUBBLE * float(re_theta) ** 0.7
    return float(re_length * nu / edge_speed)


def shape_in_range(shape_h: float) -> bool:
    """True when H is inside the separated-shear validity band."""

    return bool(np.isfinite(shape_h) and _SHAPE_MIN <= float(shape_h) <= _SHAPE_MAX)


def laminar_shear_step(
    theta: float,
    shape_h: float,
    edge_start: float,
    edge_end: float,
    ds: float,
    nu: float,
) -> tuple[float, float] | None:
    """One laminar free-shear step. Returns θ and H, or ``None`` if invalid.

    H is unchanged. θ takes the Cf = 0 momentum term and the
    zero-pressure laminar spreading dθ/ds = 0.225 ν / (Ue θ).
    """

    if (
        not np.isfinite(ds)
        or ds <= 0.0
        or not np.isfinite(nu)
        or nu <= 0.0
        or edge_start <= 0.0
        or edge_end <= 0.0
        or theta <= 0.0
        or not shape_in_range(shape_h)
    ):
        return None
    due = (float(edge_end) - float(edge_start)) / float(ds)
    pressure = -(float(shape_h) + 2.0) * (float(theta) / float(edge_start)) * due * float(ds)
    mixing = 0.225 * float(nu) / (float(edge_start) * float(theta)) * float(ds)
    theta_new = float(theta) + pressure + mixing
    if not np.isfinite(theta_new) or theta_new <= 0.0:
        return None
    return theta_new, float(shape_h)


def _separated_piece(
    theta: float,
    shape_h: float,
    edge_start: float,
    edge_end: float,
    ds: float,
    entrainment: float,
    depth: int,
) -> tuple[float, float] | None:
    due = (float(edge_end) - float(edge_start)) / float(ds)
    dtheta = -(float(shape_h) + 2.0) * (float(theta) / float(edge_start)) * due * float(ds)
    theta_new = float(theta) + dtheta
    delta = float(shape_h) * float(theta)
    flux_next = float(edge_start) * delta + float(entrainment) * 0.5 * (
        float(edge_start) + float(edge_end)
    ) * float(ds)
    if float(edge_end) <= 0.0:
        return None
    delta_new = flux_next / float(edge_end)
    shape_new = delta_new / theta_new if theta_new > 0.0 else float("nan")
    overshoot = (
        not np.isfinite(theta_new)
        or not np.isfinite(delta_new)
        or theta_new <= 0.0
        or delta_new <= 0.0
        or not shape_in_range(shape_new)
        or abs(shape_new - float(shape_h)) > 0.5
    )
    # One explicit step can jump H through the validity band when Ue
    # changes across a coarse panel. Halving the interval is the integrator,
    # not a limit on H. Eight halvings reach 1/256 of the panel.
    if overshoot and ds > 1.0e-5 and depth < 8:
        mid = 0.5 * (float(edge_start) + float(edge_end))
        half = 0.5 * float(ds)
        first = _separated_piece(theta, shape_h, edge_start, mid, half, entrainment, depth + 1)
        if first is None:
            return None
        return _separated_piece(first[0], first[1], mid, edge_end, half, entrainment, depth + 1)
    if (
        not np.isfinite(theta_new)
        or not np.isfinite(shape_new)
        or theta_new <= 0.0
        or not shape_in_range(shape_new)
    ):
        return None
    return float(theta_new), float(shape_new)


def separated_shear_step(
    theta: float,
    shape_h: float,
    edge_start: float,
    edge_end: float,
    ds: float,
    entrainment: float,
) -> tuple[float, float] | None:
    """One turbulent separated step. Returns θ and H, or ``None`` if invalid.

    Cf = 0 in the momentum integral. Displacement flux grows at
    ``entrainment * Ue``. H is δ*/θ after the step. A step that would
    leave the validity band is subdivided before it is rejected.
    """

    if (
        not np.isfinite(ds)
        or ds <= 0.0
        or edge_start <= 0.0
        or edge_end <= 0.0
        or theta <= 0.0
        or not shape_in_range(shape_h)
        or not np.isfinite(entrainment)
        or entrainment < 0.0
    ):
        return None
    return _separated_piece(
        float(theta),
        float(shape_h),
        float(edge_start),
        float(edge_end),
        float(ds),
        float(entrainment),
        0,
    )
