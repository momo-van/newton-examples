"""Compare two rigorous benchmark logs: baseline vs a variant.

Prints per-solver median totals for cold/warm and the absolute + percent
speedup of the variant over baseline.

Usage:
    python compare_variants.py [variant_name]

Defaults to variant_name='o2_nomathdx'; reads
  results/solver_rigorous.txt              (baseline)
  results/solver_rigorous_<variant>.txt    (variant)
"""

import re
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

SECTION_RE = re.compile(r"^===== iter=(\d+) / (\w+) / (\S+) =====")
PHASE_RE = re.compile(r"^\s+after_(\w+)\s+total=\s*([\d.]+)\s+delta=\s*([\d.]+)")
TOTAL_RE = re.compile(r"^TOTAL: ([\d.]+)s")

SOLVERS = ["kamino_robot_dr_legs", "robot_anymal_d", "cloth_hanging"]
PHASES = ["import", "viewer_ctor", "example_init", "step", "render"]


def parse(log_path: Path):
    """Return (phase_data, total_data) keyed like aggregate_rigorous.py."""
    phase_data = defaultdict(list)
    total_data = defaultdict(list)
    current = None
    if not log_path.exists():
        return phase_data, total_data
    for line in log_path.read_text().splitlines():
        m = SECTION_RE.match(line)
        if m:
            current = (m.group(3), m.group(2))
            continue
        m = PHASE_RE.match(line)
        if m and current is not None:
            phase = m.group(1)
            delta = float(m.group(3))
            phase_data[(current[0], current[1], phase)].append(delta)
            continue
        m = TOTAL_RE.match(line)
        if m and current is not None:
            total_data[current].append(float(m.group(1)))
            current = None
    return phase_data, total_data


def med(xs):
    return stats.median(xs) if xs else float("nan")


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "o2_nomathdx"
    here = Path(__file__).parent
    base_log = here / "results" / "solver_rigorous.txt"
    var_log = here / "results" / f"solver_rigorous_{variant}.txt"

    print(f"baseline: {base_log}")
    print(f"variant : {var_log}  (label={variant})")
    print()

    base_phase, base_total = parse(base_log)
    var_phase, var_total = parse(var_log)

    if not var_total:
        print(f"WARN no variant data parsed from {var_log}")
        return

    print("=" * 96)
    print(f"COLD-START TOTAL (median over N runs)")
    print("=" * 96)
    hdr = f"{'solver':<24} {'baseline (s)':>14} {'variant (s)':>14} {'delta (s)':>12} {'speedup':>10}"
    print(hdr)
    print("-" * len(hdr))
    for s in SOLVERS:
        b = base_total[(s, "cold")]
        v = var_total[(s, "cold")]
        bm = med(b)
        vm = med(v)
        d = bm - vm
        sp = (bm / vm) if vm else float("nan")
        pct = (d / bm * 100.0) if bm else float("nan")
        print(f"{s:<24} {bm:>10.2f}(n={len(b)}) {vm:>10.2f}(n={len(v)}) {d:>+10.2f} ({pct:+5.1f}%)  {sp:>5.2f}x")

    print()
    print("=" * 96)
    print(f"WARM TOTAL (median)")
    print("=" * 96)
    print(hdr)
    print("-" * len(hdr))
    for s in SOLVERS:
        b = base_total[(s, "warm")]
        v = var_total[(s, "warm")]
        bm = med(b)
        vm = med(v)
        d = bm - vm
        sp = (bm / vm) if vm else float("nan")
        pct = (d / bm * 100.0) if bm else float("nan")
        print(f"{s:<24} {bm:>10.2f}(n={len(b)}) {vm:>10.2f}(n={len(v)}) {d:>+10.2f} ({pct:+5.1f}%)  {sp:>5.2f}x")

    print()
    print("=" * 96)
    print("COLD per-phase median deltas (s)")
    print("=" * 96)
    cols = " | ".join(f"{p:>22}" for p in PHASES)
    print(f"{'solver':<24} | {cols}")
    print("-" * (28 + len(cols)))
    for s in SOLVERS:
        for label, data in (("base", base_phase), ("var ", var_phase)):
            cells = []
            for p in PHASES:
                xs = data[(s, "cold", p)]
                m = med(xs)
                cells.append(f"{m:>22.2f}" if xs else f"{'':>22}")
            print(f"{s:<22}{label} | " + " | ".join(cells))

    print()
    print("--- Raw cold totals ---")
    for s in SOLVERS:
        b = [round(x, 2) for x in base_total[(s, "cold")]]
        v = [round(x, 2) for x in var_total[(s, "cold")]]
        print(f"  {s:<22} baseline={b}  variant={v}")


if __name__ == "__main__":
    main()
