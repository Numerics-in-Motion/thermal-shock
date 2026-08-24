# Which Material Survives Thermal Shock Best

Four plates. All **100 × 100 × 10 mm**. All heated to a uniform temperature and
dropped into the same 20 °C water. **Only the material changes.**

```
python -m pip install -r requirements.txt
python reproduce.py               # about a minute
python reproduce.py --sweep       # also sweep the quench severity
python reproduce.py --no-plot     # numbers only
```

---

## The result

| | cracks after a drop of | can be held to | |
|---|---|---|---|
| soda-lime glass | **95.3 °C** | 600 °C | |
| alumina ceramic | 636.4 °C | 1700 °C | |
| mild steel | 891.8 °C | 1400 °C | |
| **aluminium** | **1,648.3 °C** | **640 °C** | ❌ **unreachable** |

The glass does not need to be red hot. **Ninety-five degrees** — a plate out of
a warm oven — and the same water that does nothing to the others cracks it.

**The aluminium is not "very resistant". It is uncrackable by this test.** There
is no starting temperature that gets you there, because the plate melts at
660 °C and the stress needs 1,648. The correct answer is not a big number, it is
*"the question does not have an answer for this material"*, and the run says so
rather than printing 1,648 as if you could use it.

---

## The physics

Quench a hot plate and its face cools before its core. The face wants to
contract, the core will not let it, and **the surface goes into tension** —
which is why a quench cracks a plate from the outside in. For a plate free to
bend:

```
σ = E·α/(1−ν) · (T_mean − T_surface)
```

So the fight is between how fast the surface loses heat and how fast the
interior can follow it. **Conductivity is what lets the interior follow**, and
it is what separates these four by more than anything else:

| | E (GPa) | ν | α (µ/K) | **k (W/m·K)** | σ_f (MPa) |
|---|---|---|---|---|---|
| soda-lime glass | 70 | 0.22 | 9.0 | **1.0** | 50 |
| mild steel | 200 | 0.30 | 12.0 | 50 | 250 |
| alumina ceramic | 370 | 0.22 | 8.0 | 30 | 300 |
| aluminium | 70 | 0.33 | 23.0 | **205** | 90 |

Glass 1.0 against aluminium 205. The aluminium's interior follows its surface
down, so there is barely a gradient left to make stress out of.

---

## Why you cannot just look this up

There are two handbook parameters for thermal shock, and **they disagree about
the ranking**:

| | R = σ_f(1−ν)/(E·α) | R′ = k·R |
|---|---|---|
| soda-lime glass | 61.9 K | 61.9 W/m |
| mild steel | 72.9 K | 3,645.8 W/m |
| alumina ceramic | **79.1 K** | 2,371.6 W/m |
| aluminium | 37.5 K | **7,678.0 W/m** |

**R ranks aluminium last. R′ ranks it first.** R spreads the four by 2.1×; R′
spreads them by 124×. R is the infinitely severe quench, where the surface hits
the bath temperature instantly and conduction never gets a chance; R′ is the
mild quench, where it does.

Which applies is decided by the Biot number — and at this severity the four
plates sit at **Bi = 15.0, 0.30, 0.50 and 0.073**. They straddle the question.
That is why this is simulated rather than looked up.

---

## The gates

| gate | result |
|---|---|
| energy balance closes on every material | worst **2.1e-13** |
| peak stress is proportional to the temperature drop | worst departure **1.1e-13** over 50 K to 400 K |
| a face held at the bath reproduces R = σ_f(1−ν)/(E·α) | worst **0.84 %** |
| a vanishing quench leaves essentially no thermal stress | 0.002 MPa against σ_f 90 MPa |
| the plate ends at the bath temperature | ✅ |

**The linearity gate is what makes this cheap.** The heat equation, the
convective boundary condition and the stress recovery are all linear in the
initial excess, so the peak stress is exactly proportional to the drop and the
critical drop is one division:

```
dT_crit = σ_f · dT_ref / peak_stress(dT_ref)
```

Not assumed — measured, to one part in 10¹³. Bisecting instead would cost sixty
runs per point for a number that is one division.

**A note on the R gate, because the first version of it failed.** It originally
used h = 3 × 10⁶ W/m²K on the reasoning that a huge h approximates an infinite
one. It failed at 21 %, and correctly: that gives glass Bi = 15,000 but
aluminium only Bi = 73, and at Bi = 73 a plate really does survive more than R.
The limit R is derived in is a face held **at** the bath, so that is what the
gate runs, and the residual is the discrete surface cell.

**And a note on the boundary condition.** The convective flux is applied to the
surface cell as a volumetric sink, not by holding a ghost cell at the bath
temperature. A companion study cooled a part **400× too fast** because its
boundary cells acted as an infinite reservoir, and it looked entirely plausible
until the energy book was checked. That is why the energy gate is first.

---

## Limitations

* **1D through the thickness** — no edges, no corners. A real plate cracks at an
  edge first, so these are optimistic.
* free to bend, no restraint
* constant properties with temperature
* a single uniform `h` on both faces, so **no film boiling** — which is a large
  effect in a real water quench and would make the early cooling slower
* **"cracks" means three different things.** Yield for the metals, a
  flaw-limited design tensile strength for the glass, a flexural strength for
  the ceramic. They are not the same kind of number and the comparison inherits
  that.

## Files

| | |
|---|---|
| `reproduce.py` | everything: the four materials, the quench, the gates |
| `requirements.txt` | NumPy, and Matplotlib for the figure |
| `results/results.json` | every number this run produced |
| `results/thermal_shock.png` | what cracks it against what it can be given, and the two parameters |

If something in here is wrong, please say so. That is the most useful thing you
can send back.
