
"""
EndoSim API
===========
FastAPI REST wrapper around the EndoSim ODE pipeline.
Exposes hormone simulation as a callable API for drug dosing
optimization and stress-response modeling.

Endpoints:
  GET  /           — health check + system info
  POST /simulate   — run a hormone simulation
  GET  /baseline   — return physiological baseline values
  POST /dosing     — drug dosing scenario (convenience wrapper)

Author : Alex Osterneck, CLA, MSCS, MSIT — ai70000, Ltd.
Product: EndoSim LLC
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional
import traceback
import os

from ode_model import (
    run_simulation,
    EndoSimInput,
    EndoSimState,
    BASELINE,
)

app = FastAPI(
    title="EndoSim",
    description=(
        "Real-time neuroendocrine dynamics simulation as a callable API. "
        "Models HPA axis cortisol, dopamine, norepinephrine, and oxytocin "
        "with bidirectional pharmacokinetic coupling. "
        "© ai70000, Ltd. / EndoSim LLC — Alex Osterneck, CLA, MSCS, MSIT"
    ),
    version="1.0.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    duration_minutes: float = Field(default=60.0, ge=1.0, le=1440.0,
        description="Simulation duration in minutes (1–1440)")
    dt_minutes: float = Field(default=1.0, ge=0.1, le=60.0,
        description="Output resolution in minutes")

    # Initial state (defaults to physiological baseline)
    init_CRH:  Optional[float] = Field(default=None, ge=0.0, description="Initial CRH (pg/mL)")
    init_ACTH: Optional[float] = Field(default=None, ge=0.0, description="Initial ACTH (pg/mL)")
    init_CORT: Optional[float] = Field(default=None, ge=0.0, description="Initial cortisol (µg/dL)")
    init_DA:   Optional[float] = Field(default=None, ge=0.0, description="Initial dopamine (ng/mL)")
    init_NE:   Optional[float] = Field(default=None, ge=0.0, description="Initial norepinephrine (ng/mL)")
    init_OT:   Optional[float] = Field(default=None, ge=0.0, description="Initial oxytocin (pg/mL)")

    # External inputs / perturbations
    stress_stimulus: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Acute stressor magnitude (0=none, 1=maximal)")
    social_context:  float = Field(default=0.0, ge=0.0, le=1.0,
        description="Social buffering signal (0=none, 1=maximal)")
    drug_crh: float = Field(default=0.0, description="Exogenous CRH agonist (+) or antagonist (-)")
    drug_da:  float = Field(default=0.0, description="Dopaminergic drug (+/-)")
    drug_ne:  float = Field(default=0.0, description="Adrenergic drug (+/-)")
    drug_ot:  float = Field(default=0.0, description="Oxytocin analog (+/-)")


class DosingRequest(BaseModel):
    """
    Convenience endpoint for drug dosing simulation.
    Specify drug, dose (arbitrary units), and duration.
    """
    drug: str = Field(description="Target hormone: 'cortisol', 'dopamine', 'norepinephrine', 'oxytocin'")
    dose: float = Field(description="Dose magnitude (+agonist / -antagonist)")
    duration_minutes: float = Field(default=120.0, ge=1.0, le=1440.0)
    baseline_stress: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Background stress level during dosing")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    ui_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>EndoSim API</h1><p>UI not found. Use /simulate, /baseline, /dosing</p>")

@app.get("/health")
def health():
    return {
        "service": "EndoSim",
        "status": "live",
        "version": "1.0.0",
        "description": "Real-time neuroendocrine dynamics simulation API",
        "endpoints": ["/simulate", "/baseline", "/dosing"],
        "ip": "© ai70000, Ltd. / EndoSim LLC — All rights reserved",
    }


@app.get("/baseline")
def baseline():
    """Return physiological baseline hormone concentrations."""
    return {
        "baseline": BASELINE,
        "units": {
            "CRH":  "pg/mL",
            "ACTH": "pg/mL",
            "CORT": "µg/dL",
            "DA":   "ng/mL",
            "NE":   "ng/mL",
            "OT":   "pg/mL",
        },
        "description": "Morning physiological baseline values for an adult at rest",
    }


@app.post("/simulate")
def simulate(req: SimulateRequest):
    """
    Run a hormone simulation with specified duration, initial state, and inputs.

    Returns time-series for all six hormone compartments plus final state.
    """
    try:
        initial_state = EndoSimState(
            CRH=req.init_CRH   if req.init_CRH  is not None else BASELINE["CRH"],
            ACTH=req.init_ACTH  if req.init_ACTH is not None else BASELINE["ACTH"],
            CORT=req.init_CORT  if req.init_CORT is not None else BASELINE["CORT"],
            DA=req.init_DA      if req.init_DA   is not None else BASELINE["DA"],
            NE=req.init_NE      if req.init_NE   is not None else BASELINE["NE"],
            OT=req.init_OT      if req.init_OT   is not None else BASELINE["OT"],
        )
        inputs = EndoSimInput(
            stress_stimulus=req.stress_stimulus,
            social_context=req.social_context,
            drug_crh=req.drug_crh,
            drug_da=req.drug_da,
            drug_ne=req.drug_ne,
            drug_ot=req.drug_ot,
        )
        result = run_simulation(
            duration_minutes=req.duration_minutes,
            dt_minutes=req.dt_minutes,
            initial_state=initial_state,
            inputs=inputs,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dosing")
def dosing_scenario(req: DosingRequest):
    """
    Drug dosing convenience endpoint.
    Translates drug + dose into EndoSimInput and runs 2-phase simulation:
      Phase 1: baseline (pre-dose), 30 min
      Phase 2: drug active, requested duration
    Returns both phases for comparison.
    """
    drug_map = {
        "cortisol":       "drug_crh",   # modulate via CRH/ACTH axis
        "dopamine":       "drug_da",
        "norepinephrine": "drug_ne",
        "oxytocin":       "drug_ot",
    }
    if req.drug not in drug_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown drug '{req.drug}'. Choose from: {list(drug_map.keys())}"
        )

    try:
        # Phase 1: pre-dose baseline
        baseline_inputs = EndoSimInput(stress_stimulus=req.baseline_stress)
        phase1 = run_simulation(duration_minutes=30.0, inputs=baseline_inputs)

        # Phase 2: drug active — start from phase 1 final state
        phase1_final = EndoSimState(
            CRH=phase1["final"]["CRH"],
            ACTH=phase1["final"]["ACTH"],
            CORT=phase1["final"]["CORT"],
            DA=phase1["final"]["DA"],
            NE=phase1["final"]["NE"],
            OT=phase1["final"]["OT"],
        )
        drug_kwargs = {drug_map[req.drug]: req.dose}
        dose_inputs = EndoSimInput(stress_stimulus=req.baseline_stress, **drug_kwargs)
        phase2 = run_simulation(
            duration_minutes=req.duration_minutes,
            initial_state=phase1_final,
            inputs=dose_inputs,
        )

        return {
            "drug": req.drug,
            "dose": req.dose,
            "pre_dose_final": phase1["final"],
            "post_dose_final": phase2["final"],
            "post_dose_timeseries": phase2,
            "delta": {
                k: round(phase2["final"][k] - phase1["final"][k], 6)
                for k in phase2["final"]
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
