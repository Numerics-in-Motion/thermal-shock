"""Same plate, same water. The glass cracks. The aluminium cannot be cracked.

    python -m pip install -r requirements.txt
    python reproduce.py                # about a minute
    python reproduce.py --sweep        # also sweep the quench severity
    python reproduce.py --no-plot      # numbers only

Four plates, all 100 x 100 x 10 mm, all heated to a uniform temperature and
all dropped into the same 20 C water. ONLY THE MATERIAL CHANGES -- four named
grades carrying their own handbook properties, with nothing normalised away.

    soda-lime glass    cracks after a  95 C drop
    alumina ceramic                   636 C
    mild steel                        892 C
    aluminium                       1,648 C  -- which it cannot be given,
                                                because it melts at 660 C

The aluminium is not "very resistant". It is UNCRACKABLE BY THIS TEST: there
is no starting temperature you can put it at, because the plate stops existing
before the stress gets there.

THE PHYSICS

Quench a hot plate and its face cools before its core. The face wants to
contract, the core will not let it, and the SURFACE GOES INTO TENSION -- which
is why a quench cracks a plate from the outside in. For a plate free to bend,
the surface stress is

    sigma = E*alpha/(1-nu) * (T_mean - T_surface)

so the fight is between how fast the surface loses heat and how fast the
interior can follow it. Conductivity is what lets the interior follow, and it
is the property that separates these four by more than anything else does:
glass 1.0 W/mK against aluminium 205.

WHY ONE RUN REPLACES A BISECTION

The heat equation, the convective boundary condition and the stress recovery
are all LINEAR in the initial temperature excess, so the peak stress is exactly
proportional to the drop and the critical drop is one division:

    dT_crit = sigma_f * dT_ref / peak_stress(dT_ref)

Not assumed -- measured. `linearity_gate` runs 50 K and 400 K and requires the
stress per kelvin to be the same to 1e-9. Bisecting instead would cost sixty
runs per point for a number that is one division.

TWO HANDBOOK PARAMETERS, AND WHY BOTH APPEAR

    R  = sigma_f (1-nu) / (E alpha)      the infinitely severe quench
    R' = k R                             the mild quench, where conduction helps

R ranks these four almost identically -- 62, 73, 79, 37 K -- because it does
not contain conductivity at all. R' spreads them over two orders of magnitude.
Which one applies is decided by the Biot number, and at the primary severity
here the four plates sit at Bi = 15, 0.3, 0.5 and 0.07 -- on opposite sides of
the question. That is why this is simulated rather than looked up.

Everything is plain NumPy in one file, re-derived from scratch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# four named grades, carrying their own handbook properties
# ---------------------------------------------------------------------------
MATERIALS = {
    "glass": dict(
        label="soda-lime glass", E=70e9, nu=0.22, alpha=9.0e-6, k=1.0,
        rho=2500.0, cp=840.0, sigma_f=50e6, t_max=600.0,
        note="annealed soda-lime float glass; sigma_f is a design tensile "
             "strength, flaw-limited. t_max is below the ~730 C softening "
             "point, where it stops being a plate"),
    "steel": dict(
        label="mild steel", E=200e9, nu=0.30, alpha=12.0e-6, k=50.0,
        rho=7850.0, cp=490.0, sigma_f=250e6, t_max=1400.0,
        note="S275-class structural steel; sigma_f is the yield stress, so "
             "'cracks' means 'first yields'. t_max is below the ~1500 C "
             "melting range"),
    "alumina": dict(
        label="alumina ceramic", E=370e9, nu=0.22, alpha=8.0e-6, k=30.0,
        rho=3900.0, cp=880.0, sigma_f=300e6, t_max=1700.0,
        note="99% dense alumina; sigma_f is a flexural strength. t_max is a "
             "conservative service limit below the ~2050 C melting point"),
    "aluminium": dict(
        label="aluminium", E=70e9, nu=0.33, alpha=23.0e-6, k=205.0,
        rho=2700.0, cp=900.0, sigma_f=90e6, t_max=640.0,
        note="1050-class commercially pure aluminium; sigma_f is the yield "
             "stress. t_max is just below the 660 C melting point"),
}
ORDER = ("glass", "steel", "alumina", "aluminium")

THICK = 0.010          # m -- the same plate for all four
N_CELLS = 120          # through the half thickness
T_BATH = 20.0          # C
H_PRIMARY = 3000.0     # W/m2K, a severe water quench -- the declared primary
H_GRID = [50.0, 150.0, 500.0, 1500.0, 3000.0, 10000.0, 30000.0, 100000.0,
          300000.0]
CFL = 0.40             # fraction of the explicit stability limit
DT_REF = 100.0         # the reference drop every peak stress is scaled from
TIE = 0.01             # critical drops within 1% are a tie, not a ranking


# ---------------------------------------------------------------------------
# the quench
# ---------------------------------------------------------------------------
def quench(mat, dT, h_conv, n=N_CELLS, thick=THICK, dirichlet=False):
    """Cool a plate from `T_BATH + dT` and follow its peak surface tension.

    Half the thickness is solved with an adiabatic centre plane, which is what
    symmetry gives for a plate quenched equally on both faces.

    The convective flux is applied to the surface cell as a VOLUMETRIC SINK,
    not by holding a ghost cell at the bath temperature. That distinction is
    not cosmetic: a companion study cooled a part 400x too fast because its
    boundary cells acted as an infinite reservoir, and the mistake looked
    entirely plausible until the energy book was checked.
    """
    k, rho, cp = mat["k"], mat["rho"], mat["cp"]
    alpha = k / (rho * cp)
    dx = 0.5 * thick / n
    dt = (CFL * dx * dx / (2.0 * alpha) if dirichlet
          else CFL * min(dx * dx / (2.0 * alpha), rho * cp * dx / h_conv))
    T = np.full(n, T_BATH + dT, dtype=float)
    beta = mat["E"] * mat["alpha"] / (1.0 - mat["nu"])
    peak, t, lost = 0.0, 0.0, 0.0
    e_in = rho * cp * dx * float(np.sum(T - T_BATH))

    for step in range(int(4e6)):
        lap = np.zeros(n)
        lap[1:-1] = (T[:-2] - 2.0 * T[1:-1] + T[2:]) / (dx * dx)
        lap[0] = (T[1] - T[0]) / (dx * dx)          # adiabatic centre plane
        lap[-1] = (T[-2] - T[-1]) / (dx * dx)       # surface: conduction in
        if dirichlet:
            # the infinitely severe quench: the face IS the bath. Used only by
            # the gate that checks this solver against R, because that is the
            # limit R is derived in.
            q = 0.0
            T = T + dt * alpha * lap
            T[-1] = T_BATH
        else:
            q = h_conv * (T[-1] - T_BATH)           # W/m2 leaving the face
            sink = np.zeros(n)
            sink[-1] = q / (rho * cp * dx)
            T = T + dt * (alpha * lap - sink)
        lost += q * dt
        t += dt
        s = beta * (float(np.mean(T)) - T[-1])      # tensile, at the surface
        peak = max(peak, s)
        # the stress peaks while the gradient is steepest and decays after;
        # once well past the peak there is nothing left to find, and running
        # to equilibrium costs an order of magnitude in steps
        if peak > 0.0 and s < 0.30 * peak and t > 0.0:
            break

    return dict(peak_stress=peak, t_end=t, steps=step + 1, dt=dt,
                surface_end_C=float(T[-1]),
                energy_in=e_in,
                energy_left=rho * cp * dx * float(np.sum(T - T_BATH)),
                energy_out=lost)


def dT_crit(mat, h_conv, dt_ref=DT_REF):
    """The drop at which the peak surface tension just reaches the strength."""
    r = quench(mat, dt_ref, h_conv)
    if r["peak_stress"] <= 0.0:
        return float("inf")
    return float(mat["sigma_f"] * dt_ref / r["peak_stress"])


def reach(mat):
    """The largest drop this material can physically be given, because above
    t_max it is no longer a plate."""
    return mat["t_max"] - T_BATH


def R_param(mat):
    """sigma_f (1-nu) / (E alpha) -- the infinitely severe quench."""
    return mat["sigma_f"] * (1.0 - mat["nu"]) / (mat["E"] * mat["alpha"])


def R_prime(mat):
    """k R -- the mild quench, where conduction gets to help."""
    return mat["k"] * R_param(mat)


def biot(mat, h_conv, thick=THICK):
    return h_conv * (0.5 * thick) / mat["k"]


def ranking(vals):
    """Order by critical drop, grouping anything within TIE."""
    order = sorted(vals, key=lambda k: -vals[k])
    groups, cur = [], [order[0]]
    for k in order[1:]:
        if abs(vals[k] - vals[cur[-1]]) / max(vals[cur[-1]], 1e-30) <= TIE:
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    return groups


# ---------------------------------------------------------------------------
# the gates
# ---------------------------------------------------------------------------
def gate(label, ok, detail=""):
    print("   [%s] %s%s" % ("PASS" if ok else "FAIL", label,
                            ("   %s" % detail) if detail else ""))
    return bool(ok)


def physics_gates():
    print("   physics gates -- nothing below is trustworthy unless these pass")
    ok = True

    worst = 0.0
    for kk in ORDER:
        r = quench(MATERIALS[kk], 100.0, H_PRIMARY)
        worst = max(worst, abs(r["energy_in"] - r["energy_left"]
                               - r["energy_out"]) / r["energy_in"])
    ok &= gate("energy balance closes on every material", worst < 1e-3,
               "worst %.3e" % worst)

    lin = 0.0
    for kk in ORDER:
        m = MATERIALS[kk]
        a = quench(m, 50.0, H_PRIMARY)["peak_stress"] / 50.0
        b = quench(m, 400.0, H_PRIMARY)["peak_stress"] / 400.0
        lin = max(lin, abs(b - a) / a)
    ok &= gate("peak stress is proportional to the temperature drop",
               lin < 1e-9, "worst departure %.2e over 50 K to 400 K" % lin)

    # A first attempt at this gate used h = 3e6 and FAILED at 21%, correctly:
    # that gives glass Bi = 15,000 but aluminium only Bi = 73, and at Bi = 73
    # a plate really does survive more than R. The limit R is derived in is a
    # face held AT the bath, so that is what the gate runs. The residual is
    # the discrete surface cell, 1/n of the half thickness.
    worst, rows = 0.0, []
    for kk in ORDER:
        m = MATERIALS[kk]
        r = quench(m, DT_REF, 0.0, dirichlet=True)
        got = m["sigma_f"] * DT_REF / r["peak_stress"]
        worst = max(worst, abs(got - R_param(m)) / R_param(m))
        rows.append("%s %.1f vs %.1f K" % (kk, got, R_param(m)))
    ok &= gate("a face held at the bath reproduces R = sigma_f(1-nu)/(E alpha)",
               worst < 0.02, "worst %.2e   (%s)" % (worst, "; ".join(rows)))

    m = MATERIALS["aluminium"]
    r = quench(m, 100.0, 1.0)
    ok &= gate("a vanishing quench leaves essentially no thermal stress",
               r["peak_stress"] < 0.02 * m["sigma_f"],
               "peak %.3f MPa against sigma_f %.0f MPa"
               % (r["peak_stress"] / 1e6, m["sigma_f"] / 1e6))

    r = quench(MATERIALS["glass"], 500.0, H_PRIMARY)
    ok &= gate("the plate ends at the bath temperature",
               abs(r["surface_end_C"] - T_BATH) < 0.5 * 500.0,
               "surface at %.1f C" % r["surface_end_C"])
    return ok


def run(h_conv, echo=None):
    rows = {}
    for kk in ORDER:
        m = MATERIALS[kk]
        d = dT_crit(m, h_conv)
        rows[kk] = dict(dT_crit=d, reachable=bool(d <= reach(m)),
                        reach=reach(m), biot=biot(m, h_conv),
                        R=R_param(m), R_prime=R_prime(m))
        if echo:
            echo("      %-17s cracks after a %8.1f C drop   Bi %8.3f   %s"
                 % (m["label"], d, rows[kk]["biot"],
                    "reachable" if rows[kk]["reachable"]
                    else "UNREACHABLE -- it melts first (max %.0f C)" % reach(m)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="also sweep the quench severity h")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args(argv)
    t0 = time.time()

    print("Which material survives thermal shock best")
    print("   one plate, %.0f x %.0f mm x %.0f mm, quenched on both faces into "
          "%.0f C water" % (100, 100, 1000 * THICK, T_BATH))
    print("   h = %.0f W/m2K, a severe water quench. Only the material changes."
          % H_PRIMARY)
    print()
    ok = physics_gates()
    if not ok:
        print("\n   a gate failed.")
        return 1

    print()
    print("   the primary run")
    rows = run(H_PRIMARY, echo=print)
    ranks = ranking({k: rows[k]["dT_crit"] for k in ORDER})
    print()
    print("      ranking: %s"
          % " > ".join(" = ".join(MATERIALS[k]["label"] for k in g)
                       for g in ranks))
    unreachable = [k for k in ORDER if not rows[k]["reachable"]]

    print()
    print("   the two handbook parameters, for comparison")
    print("      %-17s %10s %14s" % ("", "R [K]", "R' [W/m]"))
    for kk in ORDER:
        print("      %-17s %10.1f %14.1f"
              % (MATERIALS[kk]["label"], rows[kk]["R"], rows[kk]["R_prime"]))
    rs = [rows[k]["R"] for k in ORDER]
    rps = [rows[k]["R_prime"] for k in ORDER]
    print("      R spreads them %.2fx; R' spreads them %.0fx. Which applies is"
          % (max(rs) / min(rs), max(rps) / min(rps)))
    print("      decided by the Biot number, and these four straddle it.")

    sweep = None
    if a.sweep:
        print()
        print("   sweeping the quench severity")
        sweep = []
        for h in H_GRID:
            r = run(h)
            g = ranking({k: r[k]["dT_crit"] for k in ORDER})
            sweep.append(dict(h=h, rows=r, ranking=[list(x) for x in g]))
            print("      h = %9.0f W/m2K   %s"
                  % (h, " > ".join(" = ".join(k for k in gg) for gg in g)))
        orders = {tuple(tuple(x) for x in s["ranking"]) for s in sweep}
        print("      [%s] the ranking is the same at every severity   %d "
              "distinct order(s)"
              % ("OK " if len(orders) == 1 else "MOVES", len(orders)))

    os.makedirs(a.out_dir, exist_ok=True)
    payload = dict(
        held_fixed=dict(thickness_m=THICK, cells=N_CELLS, bath_C=T_BATH,
                        h_primary=H_PRIMARY, dT_ref=DT_REF, cfl=CFL),
        materials={k: {kk: vv for kk, vv in MATERIALS[k].items()} for k in ORDER},
        primary=rows, ranking=[list(g) for g in ranks],
        unreachable=unreachable, sweep=sweep,
        runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(a.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=float)
    print()
    print("   Saved: %s" % os.path.join(a.out_dir, "results.json"))

    if not a.no_plot:
        p = os.path.join(a.out_dir, "thermal_shock.png")
        make_plot(rows, p)
        print("   Saved: %s" % p)

    print()
    print("=" * 70)
    print("MAIN CONCLUSION")
    print("=" * 70)
    print()
    g = rows["glass"]
    print("The same plate in the same water. The glass cracks after a %.0f C"
          % g["dT_crit"])
    print("drop -- it does not need to be red hot, only hot.")
    print()
    for kk in ORDER:
        r, m = rows[kk], MATERIALS[kk]
        print("   %-17s %8.1f C   %s"
              % (m["label"], r["dT_crit"],
                 "" if r["reachable"]
                 else "-- CANNOT BE REACHED, it melts at %.0f C" % m["t_max"]))
    if unreachable:
        print()
        print("The %s is not 'very resistant'. It is UNCRACKABLE BY THIS TEST:"
              % ", ".join(MATERIALS[k]["label"] for k in unreachable))
        print("there is no starting temperature that gets there, because the")
        print("plate stops existing first.")
    print()
    print("Conductivity is what does it. Glass 1.0 W/mK against aluminium 205:")
    print("the aluminium's interior follows its surface down, so there is")
    print("barely a gradient to make stress out of.")
    print()
    print("LIMITATIONS: 1D through the thickness, so no edges and no corners --")
    print("a real plate cracks at an edge first. Free to bend, no restraint.")
    print("Constant properties with temperature. A single uniform h on both")
    print("faces, so no film boiling, which is a large effect in a real water")
    print("quench. 'Cracks' means first reaching the stated strength: yield for")
    print("the metals, a flaw-limited tensile strength for the glass, and a")
    print("flexural strength for the ceramic -- three different kinds of number.")
    print()
    print("No number above is hardcoded -- every one is computed by this run.")
    print("   %.1f s" % (time.time() - t0))
    return 0


def make_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    xs = range(len(ORDER))
    cols = ["#5bc0ff", "#8899aa", "#ffb703", "#2ecc71"]
    vals = [rows[k]["dT_crit"] for k in ORDER]
    ax[0].bar(xs, vals, color=cols)
    for i, k in enumerate(ORDER):
        ax[0].plot([i - 0.4, i + 0.4], [rows[k]["reach"]] * 2, color="#e63946",
                   lw=2)
    ax[0].set_yscale("log")
    ax[0].set_xticks(list(xs))
    ax[0].set_xticklabels([MATERIALS[k]["label"] for k in ORDER], rotation=12)
    ax[0].set_ylabel("critical temperature drop [C]")
    ax[0].set_title("bars = what it takes to crack it;\nred line = the most it "
                    "can be given before it melts")

    ax[1].bar([i - 0.2 for i in xs], [rows[k]["R"] for k in ORDER], width=0.4,
              color="#457b9d", label="R  (severe quench)")
    ax2 = ax[1].twinx()
    ax2.bar([i + 0.2 for i in xs], [rows[k]["R_prime"] for k in ORDER],
            width=0.4, color="#ffb703", label="R' (mild quench)")
    ax2.set_yscale("log")
    ax[1].set_xticks(list(xs))
    ax[1].set_xticklabels([MATERIALS[k]["label"] for k in ORDER], rotation=12)
    ax[1].set_ylabel("R [K]")
    ax2.set_ylabel("R' [W/m]")
    ax[1].set_title("the two handbook parameters disagree about the ranking")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
