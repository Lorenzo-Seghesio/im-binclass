"""
Run_WPred_Pipeline.py
─────────────────────
Sequential pipeline launcher for the weight-prediction models.

Usage
─────
    python Run_WPred_Pipeline.py                       # all materials: PP, ABS, ALL, DOE1
    python Run_WPred_Pipeline.py --material PP
    python Run_WPred_Pipeline.py --material PP ABS
    python Run_WPred_Pipeline.py --material PP ABS ALL

The pipeline config (config/WPred_Pipeline_config.json) controls:
  - n_runs_m1          : how many times M1 is launched per material
  - n_runs_m2          : how many times M2 is launched per material after M1 finishes
  - n_runs_fusion      : how many times Fusion is launched per material after M2
  - n_runs_ref         : how many times Reference is launched per material
  - abort_on_*_error   : stop the loop for that model if any run exits with an error
  - scripts.m1 / m2 / fusion / ref : paths to the script files (relative to project root)

Per-material order: M1 → M2 → Fusion → Reference.
Fusion is skipped automatically when material=ALL (it only supports PP / ABS).
M2 is skipped (and so are Fusion) if no M1 run succeeded.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
PIPELINE_CFG = BASE_DIR / "config" / "WPred_Pipeline_config.json"
ALL_MATERIALS = ["PP", "ABS", "ALL", "DOE1"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as mm:ss or hh:mm:ss."""
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def _banner(text: str, char: str = "═", width: int = 64) -> None:
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _run_script(script: Path, material: str, run_label: str,
                optuna_seed: int | None = None) -> bool:
    """
    Launch *script* as a subprocess with --material *material*.
    Returns True on success (exit code 0), False otherwise.
    """
    cmd = [sys.executable, str(script), "--material", material]
    if optuna_seed is not None and script.name != "Fusion_M1M2_WPred.py":
        cmd.extend(["--optuna-seed", str(optuna_seed)])
    print(f"\n  [{run_label}] Command : {' '.join(cmd)}")
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.monotonic() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"\n  [{run_label}] {status}  — elapsed {_fmt_elapsed(elapsed)}")
    return ok


def _run_loop(script: Path, material: str, n_runs: int,
              label: str, abort_on_error: bool,
              optuna_seed: int | None = None) -> list[bool]:
    """Run *script* n_runs times. Returns list of per-run success flags."""
    results: list[bool] = []
    for i in range(1, n_runs + 1):
        run_label = f"{label}  run {i}/{n_runs}"
        _banner(f"{run_label}  ·  material={material}", char="─")
        ok = _run_script(script, material, run_label, optuna_seed)
        results.append(ok)
        if not ok and abort_on_error:
            print(f"\n  [pipeline] abort_on_error=true → stopping {label} loop.")
            break
    return results


def _print_summary(label: str, results: list[bool]) -> None:
    passed = sum(results)
    total  = len(results)
    print(f"  {label}: {passed}/{total} runs succeeded")
    for i, ok in enumerate(results, 1):
        mark = "✓" if ok else "✗"
        print(f"    {mark}  run {i}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential pipeline: run M1 N times then M2 M times, "
                    "repeated for each requested material.")
    parser.add_argument(
        "--material", type=str.upper,
        nargs="*",                          # 0 or more values
        choices=ALL_MATERIALS,
        default=None,
        metavar="MAT",
        help="Material(s) to process: PP | ABS | ALL | DOE1  (case-insensitive). "
               "Includes DOE1 for the Arburg DoE dataset. "
             "Pass one or more values separated by spaces. "
             "Omit entirely to run all materials in sequence.")
    parser.add_argument(
        "--config", type=Path, default=PIPELINE_CFG,
        help="Path to WPred_Pipeline_config.json "
             f"(default: {PIPELINE_CFG.relative_to(BASE_DIR)})")
    parser.add_argument(
        "--optuna-seed", type=int, default=None,
        help="Optuna sampler seed for exact replay; omit for fresh seeds per HPO launch.")
    args = parser.parse_args()

    # Default to all materials when --material is omitted or empty
    materials: list[str] = args.material if args.material else ALL_MATERIALS

    cfg_path = args.config.resolve()
    if not cfg_path.exists():
        sys.exit(f"[pipeline] Config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())

    n_runs_m1         = int(cfg.get("n_runs_m1", 1))
    n_runs_m2         = int(cfg.get("n_runs_m2", 1))
    n_runs_fusion     = int(cfg.get("n_runs_fusion", 1))
    n_runs_ref        = int(cfg.get("n_runs_ref", 1))
    abort_on_m1_err   = bool(cfg.get("abort_on_m1_error", True))
    abort_on_m2_err   = bool(cfg.get("abort_on_m2_error", False))
    abort_on_fus_err  = bool(cfg.get("abort_on_fusion_error", False))
    abort_on_ref_err  = bool(cfg.get("abort_on_ref_error", False))
    scripts_cfg       = cfg.get("scripts", {})

    m1_script     = BASE_DIR / scripts_cfg.get("m1",     "src/M1_PPPT_to_W.py")
    m2_script     = BASE_DIR / scripts_cfg.get("m2",     "src/M2_PP_to_F.py")
    fusion_script = BASE_DIR / scripts_cfg.get("fusion", "src/Fusion_M1M2_WPred.py")
    ref_script    = BASE_DIR / scripts_cfg.get("ref",    "src/Reference_Models_W_Pred.py")

    for name, path in [("M1", m1_script), ("M2", m2_script),
                       ("Fusion", fusion_script), ("Ref", ref_script)]:
        if not path.exists():
            sys.exit(f"[pipeline] {name} script not found: {path}")

    _banner(f"WPred Pipeline  ·  materials={', '.join(materials)}", char="═")
    print(f"  Config    : {cfg_path.relative_to(BASE_DIR)}")
    print(f"  Materials : {', '.join(materials)}")
    print(f"  M1        : {m1_script.relative_to(BASE_DIR)}  ×{n_runs_m1}"
          f"  (abort_on_error={abort_on_m1_err})")
    print(f"  M2        : {m2_script.relative_to(BASE_DIR)}  ×{n_runs_m2}"
          f"  (abort_on_error={abort_on_m2_err})")
    print(f"  Fusion    : {fusion_script.relative_to(BASE_DIR)}  ×{n_runs_fusion}"
          f"  (abort_on_error={abort_on_fus_err})  [skipped for material=ALL]")
    print(f"  Reference : {ref_script.relative_to(BASE_DIR)}  ×{n_runs_ref}"
          f"  (abort_on_error={abort_on_ref_err})")
    if args.optuna_seed is None:
        print("  Optuna seed: fresh seed generated by each HPO child")
    else:
        print(f"  Optuna seed: {args.optuna_seed} (exact replay override)")

    t_pipeline_start = time.monotonic()

    # Per-material results: {mat: {"m1": [...], "m2": [...], "fusion": [...], "ref": [...]}}
    all_results: dict[str, dict[str, list[bool]]] = {}

    for mat in materials:
        _banner(f"MATERIAL: {mat}", char="█")

        # ── Phase 1: M1 ───────────────────────────────────────────────────────
        _banner(f"[{mat}] PHASE 1 — M1  (PP/PT curves → weight)")
        m1_results = _run_loop(m1_script, mat, n_runs_m1, f"M1/{mat}",
                               abort_on_m1_err, args.optuna_seed)

        m1_any_ok = any(m1_results)
        if not m1_any_ok:
            _banner(f"[{mat}] No M1 run succeeded — skipping M2, Fusion, Reference for {mat}.", char="!")
            all_results[mat] = {"m1": m1_results, "m2": [], "fusion": [], "ref": []}
            continue

        # ── Phase 2: M2 ───────────────────────────────────────────────────────
        _banner(f"[{mat}] PHASE 2 — M2  (process params → f-features)")
        m2_results = _run_loop(m2_script, mat, n_runs_m2, f"M2/{mat}",
                               abort_on_m2_err, args.optuna_seed)

        # ── Phase 3: Fusion ───────────────────────────────────────────────────
        if mat == "ALL":
            _banner(f"[{mat}] PHASE 3 — Fusion  (skipped: Fusion does not support material=ALL)", char="-")
            fusion_results: list[bool] = []
        else:
            _banner(f"[{mat}] PHASE 3 — Fusion  (M1+M2 outputs → weight prediction)")
            fusion_results = _run_loop(fusion_script, mat, n_runs_fusion,
                                       f"Fusion/{mat}", abort_on_fus_err)

        # ── Phase 4: Reference ────────────────────────────────────────────────
        _banner(f"[{mat}] PHASE 4 — Reference  (baseline models)")
        ref_results = _run_loop(ref_script, mat, n_runs_ref, f"Ref/{mat}",
                                abort_on_ref_err, args.optuna_seed)

        all_results[mat] = {"m1": m1_results, "m2": m2_results,
                            "fusion": fusion_results, "ref": ref_results}

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - t_pipeline_start
    _banner(f"Pipeline complete  ·  total {_fmt_elapsed(total_elapsed)}")

    any_failure = False
    for mat in materials:
        print(f"\n  ── {mat} ──")
        m1r  = all_results[mat]["m1"]
        m2r  = all_results[mat]["m2"]
        fusr = all_results[mat]["fusion"]
        refr = all_results[mat]["ref"]
        _print_summary(f"M1/{mat}",     m1r)
        _print_summary(f"M2/{mat}",     m2r)
        if fusr:
            _print_summary(f"Fusion/{mat}", fusr)
        else:
            print(f"  Fusion/{mat}: skipped")
        _print_summary(f"Ref/{mat}",    refr)
        if not all(m1r) or not all(m2r) or not all(fusr) or not all(refr):
            any_failure = True

    sys.exit(1 if any_failure else 0)


if __name__ == "__main__":
    main()

