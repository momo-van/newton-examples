"""Per-kernel compile-time diff between baseline and a variant log.

For each (solver, kind=cold) section we extract `Module <name> <hash> ... took N ms (compiled)`
lines, strip the trailing hash (which varies between baseline and variant), and median across
iterations. Output: ranked diff per solver.

Usage:
    python diff_kernels.py [variant]    # default variant=o2_nomathdx
"""

import re
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

SECTION_RE = re.compile(r"^===== iter=(\d+) / (\w+) / (\S+) =====")
# Module <name> <hash> load on device 'X' took N.NN ms  (compiled|cached)
MOD_RE = re.compile(r"^Module\s+(.+?)\s+[0-9a-f]{6,}\s+load on device '[^']+' took ([\d.]+) ms\s+\((compiled|cached)\)")
HASH_TAIL_RE = re.compile(r"_[0-9a-f]{6,}$")

SOLVERS = ["kamino_robot_dr_legs", "robot_anymal_d", "cloth_hanging"]


def normalize_name(name: str) -> str:
    """Strip trailing source-hash so the same kernel matches across runs."""
    return HASH_TAIL_RE.sub("", name)


def parse(log_path: Path):
    """Return {(solver, kind, kernel): [ms, ...]} across compiled-only entries."""
    data = defaultdict(list)
    current = None
    if not log_path.exists():
        return data
    for line in log_path.read_text().splitlines():
        m = SECTION_RE.match(line)
        if m:
            current = (m.group(3), m.group(2))
            continue
        m = MOD_RE.match(line)
        if m and current is not None and m.group(3) == "compiled":
            name = normalize_name(m.group(1))
            data[(current[0], current[1], name)].append(float(m.group(2)))
    return data


def med(xs):
    return stats.median(xs) if xs else 0.0


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "o2_nomathdx"
    here = Path(__file__).parent
    base_log = here / "results" / "solver_rigorous.txt"
    var_log = here / "results" / f"solver_rigorous_{variant}.txt"

    base = parse(base_log)
    var = parse(var_log)

    print(f"baseline: {base_log.name}")
    print(f"variant : {var_log.name}")
    print()

    for solver in SOLVERS:
        all_kernels = {k[2] for k in base if k[0] == solver and k[1] == "cold"} | \
                      {k[2] for k in var if k[0] == solver and k[1] == "cold"}
        rows = []
        for k in all_kernels:
            bm = med(base[(solver, "cold", k)])
            vm = med(var[(solver, "cold", k)])
            d = vm - bm  # positive = variant slower
            rows.append((d, bm, vm, k))
        rows.sort(reverse=True)  # largest regression first

        base_total = sum(med(base[(solver, "cold", k)]) for k in all_kernels)
        var_total = sum(med(var[(solver, "cold", k)]) for k in all_kernels)

        print("=" * 110)
        print(f"{solver}   cold-compile sum: baseline={base_total/1000:.1f}s  variant={var_total/1000:.1f}s  delta={var_total/1000 - base_total/1000:+.1f}s")
        print("=" * 110)
        print(f"{'delta ms':>10} {'base ms':>10} {'var ms':>10}   kernel")
        print("-" * 110)

        regressions = [r for r in rows if r[0] > 200]
        improvements = [r for r in rows if r[0] < -200]
        print(f"-- TOP REGRESSIONS (variant slower by >200 ms median) --  ({len(regressions)} kernels)")
        for d, bm, vm, k in regressions[:15]:
            print(f"{d:>+10.0f} {bm:>10.0f} {vm:>10.0f}   {k}")
        print()
        print(f"-- TOP IMPROVEMENTS (variant faster by >200 ms median) --  ({len(improvements)} kernels)")
        for d, bm, vm, k in improvements[-15:]:
            print(f"{d:>+10.0f} {bm:>10.0f} {vm:>10.0f}   {k}")
        print()


if __name__ == "__main__":
    main()
