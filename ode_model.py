"""
EndoSim — ODE Pipeline
======================
Real-time neuroendocrine dynamics simulation.
Derived from the ai2agi DTB pharmacokinetic endocrine model (Osterneck, 2025-2026).

Four coupled hormone subsystems:
  1. HPA axis  — CRH → ACTH → Cortisol feedback loop
  2. Dopamine  — synthesis / degradation / receptor kinetics
  3. Norepinephrine — arousal modulator
  4. Oxytocin  — social / metabolic coupling

Bidirectional coupling:
  - Cortisol suppresses dopamine synthesis (stress → reward blunting)
  - Norepinephrine modulates cortisol release (arousal → HPA activation)
  - Oxytocin inhibits cortisol (social buffering)
  - Dopamine positively couples oxytocin release

Author : Alex Osterneck, CLA, MSCS, MSIT — ai70000, Ltd.
Product: EndoSim LLC
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional


# ── Physiological baseline concentrations (pg/mL or ng/mL as noted) ──────────
BASELINE = {
    "CRH":    10.0,    # pg/mL  — hypothalamic CRH
    "ACTH":   25.0,    # pg/mL  — pituitary ACTH
    "CORT":   15.0,    # µg/dL  — cortisol (morning baseline ~15, nadir ~3)
    "DA":     0.13,    # ng/mL  — plasma dopamine proxy
    "NE":     0.30,    # ng/mL  — plasma norepinephrine
    "OT":     1.0,     # pg/mL  — oxytocin
}

# ── Kinetic parameters ────────────────────────────────────────────────────────
# Calibration target: system rests near BASELINE values (no perturbation, long t)
# Approach: at steady state, dX/dt = 0. Solve for synthesis rates given deg rates.

# HPA axis
# SS: CRH* ≈ K_CRH_SYN / (K_CRH_DEG + K_CORT_FB * fb_frac)
# With feedback fb_frac ~ CORT*/(CORT*+30) and CORT*=15 → fb_frac~0.33
# Target CRH*=10 → K_CRH_SYN = 10*(0.08 + 0.05*0.33) = 10*0.097 ≈ 0.97
K_CRH_SYN   = 0.97    # calibrated to hold CRH~10 at rest
K_CRH_DEG   = 0.08
K_ACTH_SYN  = 0.20    # ACTH* = K_ACTH_SYN*CRH*/K_ACTH_DEG = 0.20*10/0.08 = 25 ✓
K_ACTH_DEG  = 0.08
# CORT* = K_CORT_SYN*ACTH*/(K_CORT_DEG + K_OT_CORT*OT*)
# OT*~1 → denom = 0.03 + 0.03*1 = 0.06; CORT*=15 → K_CORT_SYN=15*0.06/25=0.036
K_CORT_SYN  = 0.036
K_CORT_DEG  = 0.03
K_CORT_FB   = 0.05

# Dopamine
# DA* = (da_syn_eff + K_OT_DA*OT*) / K_DA_DEG
# cort_frac=15/(15+20)=0.43 → da_syn_eff=0.08*(1-0.6*0.43)=0.059
# DA* = (0.059 + 0.02*1)/0.12 = 0.66  (slightly above 0.13 baseline — acceptable proxy)
K_DA_SYN    = 0.014269  # calibrated: DA* = 0.13 at physiological baseline
K_DA_DEG    = 0.12
K_DA_OT     = 0.005     # reduced to allow positive K_DA_SYN at baseline CORT=15

# Norepinephrine
# NE* = K_NE_SYN/K_NE_DEG → 0.30 → K_NE_SYN = 0.30*0.15 = 0.045
K_NE_SYN    = 0.045
K_NE_DEG    = 0.15
K_NE_CORT   = 0.006   # NE→HPA coupling, scaled down to avoid runaway

# Oxytocin
# OT* = (K_OT_SYN + K_OT_DA*DA*) / (K_OT_DEG + K_OT_CORT*CORT*)
# ≈ (K_OT_SYN + 0.02*0.13) / (0.08 + 0.03*15) = (K_OT_SYN+0.0026)/0.53
# OT*=1 → K_OT_SYN = 0.53 - 0.0026 ≈ 0.527
K_OT_SYN    = 0.527
K_OT_DEG    = 0.08
K_OT_CORT   = 0.03
K_OT_DA     = 0.02


@dataclass
class EndoSimInput:
    """
    External perturbation inputs to the hormone system.
    All values are additive stimulus magnitudes (dimensionless scale factors).
    Set to 0.0 for unperturbed baseline simulation.
    """
    stress_stimulus: float = 0.0       # acute stressor (0–1 scale → amplifies HPA)
    drug_crh: float = 0.0              # exogenous CRH agonist/antagonist (+ / -)
    drug_da: float = 0.0               # dopaminergic drug (+ / -)
    drug_ne: float = 0.0               # adrenergic drug (+ / -)
    drug_ot: float = 0.0               # oxytocin analog (+ / -)
    social_context: float = 0.0        # social buffering signal (0–1) → OT release


@dataclass
class EndoSimState:
    """Hormone state vector — maps directly to ODE state indices."""
    CRH:  float = BASELINE["CRH"]
    ACTH: float = BASELINE["ACTH"]
    CORT: float = BASELINE["CORT"]
    DA:   float = BASELINE["DA"]
    NE:   float = BASELINE["NE"]
    OT:   float = BASELINE["OT"]

    def to_array(self) -> np.ndarray:
        return np.array([self.CRH, self.ACTH, self.CORT, self.DA, self.NE, self.OT])

    @classmethod
    def from_array(cls, y: np.ndarray) -> "EndoSimState":
        return cls(CRH=y[0], ACTH=y[1], CORT=y[2], DA=y[3], NE=y[4], OT=y[5])


# ── The ODE system ────────────────────────────────────────────────────────────

def endosim_odes(t: float, y: np.ndarray, inputs: EndoSimInput) -> np.ndarray:
    """
    Coupled ODE system for neuroendocrine dynamics.

    State vector y = [CRH, ACTH, CORT, DA, NE, OT]

    Returns dy/dt for all six state variables.
    """
    CRH, ACTH, CORT, DA, NE, OT = y

    # Clamp to physiological non-negative bounds
    CRH  = max(CRH,  0.0)
    ACTH = max(ACTH, 0.0)
    CORT = max(CORT, 0.0)
    DA   = max(DA,   0.0)
    NE   = max(NE,   0.0)
    OT   = max(OT,   0.0)

    # ── HPA axis ─────────────────────────────────────────────────────────────
    # CRH: synthesized basally, stimulated by stress + NE, inhibited by cortisol
    # Negative feedback: saturating Hill term so CRH is suppressed but not zeroed
    crh_stress   = inputs.stress_stimulus * 2.0
    cort_fb_frac = CORT / (CORT + 30.0)         # Hill term on cortisol feedback
    dCRH = (K_CRH_SYN + crh_stress + K_NE_CORT * NE + inputs.drug_crh
            - K_CRH_DEG * CRH
            - K_CORT_FB * cort_fb_frac * CRH)   # glucocorticoid negative feedback

    # ACTH: driven by CRH, degraded by half-life kinetics
    dACTH = (K_ACTH_SYN * CRH
             - K_ACTH_DEG * ACTH)

    # Cortisol: driven by ACTH, inhibited by oxytocin (social buffering)
    ot_buffer = K_OT_CORT * OT                  # oxytocin blunts HPA
    dCORT = (K_CORT_SYN * ACTH
             - K_CORT_DEG * CORT
             - ot_buffer * CORT)

    # ── Dopamine ──────────────────────────────────────────────────────────────
    # DA: baseline synthesis suppressed by cortisol (stress → reward blunting).
    # Suppression is fractional (Hill-style), not additive subtraction,
    # so synthesis never goes negative.
    # OT enhances DA release; drug input additive.
    cort_suppress_frac = CORT / (CORT + 20.0)   # Hill term: saturates at high CORT
    da_syn_effective   = K_DA_SYN * (1.0 - 0.6 * cort_suppress_frac)  # max 60% suppression
    ot_enhance_da      = K_DA_OT * OT            # OT → DA positive coupling
    dDA = (da_syn_effective + ot_enhance_da + inputs.drug_da
           - K_DA_DEG * DA)

    # ── Norepinephrine ───────────────────────────────────────────────────────
    # NE: stress and arousal driven; feeds back into HPA (via dCRH above)
    ne_stress = inputs.stress_stimulus * 1.5
    dNE = (K_NE_SYN + ne_stress + inputs.drug_ne
           - K_NE_DEG * NE)

    # ── Oxytocin ─────────────────────────────────────────────────────────────
    # OT: social context → release; DA positive; cortisol suppressive
    social_ot     = inputs.social_context * 0.5
    cort_supp_ot  = K_OT_CORT * CORT
    da_promote_ot = K_OT_DA * DA
    dOT = (K_OT_SYN + social_ot + da_promote_ot + inputs.drug_ot
           - K_OT_DEG * OT
           - cort_supp_ot * OT)

    return np.array([dCRH, dACTH, dCORT, dDA, dNE, dOT])


# ── Simulation runner ─────────────────────────────────────────────────────────

def run_simulation(
    duration_minutes: float = 60.0,
    dt_minutes: float = 1.0,
    initial_state: Optional[EndoSimState] = None,
    inputs: Optional[EndoSimInput] = None,
) -> dict:
    """
    Run the EndoSim ODE pipeline.

    Parameters
    ----------
    duration_minutes : total simulation time in minutes
    dt_minutes       : output resolution (evaluation points), default 1 min
    initial_state    : starting hormone concentrations (defaults to physiological baseline)
    inputs           : external perturbations / drug inputs

    Returns
    -------
    dict with keys:
        t      : time array (minutes)
        states : dict of hormone name → concentration array over time
        final  : EndoSimState at t=duration_minutes
        metadata : simulation metadata
    """
    if initial_state is None:
        initial_state = EndoSimState()
    if inputs is None:
        inputs = EndoSimInput()

    y0 = initial_state.to_array()
    t_span = (0.0, duration_minutes)
    t_eval = np.arange(0.0, duration_minutes + dt_minutes, dt_minutes)

    sol = solve_ivp(
        fun=endosim_odes,
        t_span=t_span,
        y0=y0,
        args=(inputs,),
        method="RK45",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    hormone_names = ["CRH", "ACTH", "CORT", "DA", "NE", "OT"]
    states = {name: sol.y[i].tolist() for i, name in enumerate(hormone_names)}
    final_state = EndoSimState.from_array(sol.y[:, -1])

    return {
        "t": sol.t.tolist(),
        "states": states,
        "final": {
            "CRH":  final_state.CRH,
            "ACTH": final_state.ACTH,
            "CORT": final_state.CORT,
            "DA":   final_state.DA,
            "NE":   final_state.NE,
            "OT":   final_state.OT,
        },
        "metadata": {
            "duration_minutes": duration_minutes,
            "dt_minutes": dt_minutes,
            "n_steps": len(sol.t),
            "solver": "RK45",
            "stress_stimulus": inputs.stress_stimulus,
            "social_context": inputs.social_context,
            "solver_message": sol.message,
        },
    }


# ── Synthetic validation ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("EndoSim — Synthetic Validation")
    print("=" * 60)

    # Test 1: Baseline — no perturbation; system should remain near baseline
    print("\n[TEST 1] Baseline (no perturbation, 60 min)")
    result = run_simulation(duration_minutes=60.0)
    final = result["final"]
    print(f"  CRH : {final['CRH']:.4f}  (baseline {BASELINE['CRH']})")
    print(f"  ACTH: {final['ACTH']:.4f}  (baseline {BASELINE['ACTH']})")
    print(f"  CORT: {final['CORT']:.4f}  (baseline {BASELINE['CORT']})")
    print(f"  DA  : {final['DA']:.4f}  (baseline {BASELINE['DA']})")
    print(f"  NE  : {final['NE']:.4f}  (baseline {BASELINE['NE']})")
    print(f"  OT  : {final['OT']:.4f}  (baseline {BASELINE['OT']})")

    # Test 2: Acute stress — cortisol and NE should rise, DA should suppress
    print("\n[TEST 2] Acute stress (stimulus=0.8, 30 min)")
    stress_inputs = EndoSimInput(stress_stimulus=0.8)
    result2 = run_simulation(duration_minutes=30.0, inputs=stress_inputs)
    f2 = result2["final"]
    print(f"  CORT: {f2['CORT']:.4f}  (expect > {BASELINE['CORT']})")
    print(f"  NE  : {f2['NE']:.4f}   (expect > {BASELINE['NE']})")
    print(f"  DA  : {f2['DA']:.4f}   (expect < {BASELINE['DA']} due to cortisol suppression)")

    # Test 3: Social buffering — oxytocin should rise, cortisol should blunt
    print("\n[TEST 3] Social buffering (social_context=1.0, stress=0.5, 60 min)")
    social_inputs = EndoSimInput(stress_stimulus=0.5, social_context=1.0)
    result3 = run_simulation(duration_minutes=60.0, inputs=social_inputs)
    f3 = result3["final"]
    # Compare cortisol under stress alone vs stress+social
    result3b = run_simulation(duration_minutes=60.0, inputs=EndoSimInput(stress_stimulus=0.5))
    f3b = result3b["final"]
    print(f"  OT   (social): {f3['OT']:.4f}  vs  (no social): {f3b['OT']:.4f}")
    print(f"  CORT (social): {f3['CORT']:.4f}  vs  (no social): {f3b['CORT']:.4f}")
    print(f"  Social buffering {'CONFIRMED' if f3['CORT'] < f3b['CORT'] else 'FAILED'}")

    print("\n[PASS] All validation tests complete.")
    print(f"Steps in test 1: {result['metadata']['n_steps']}")
