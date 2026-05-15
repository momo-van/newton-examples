"""Phase-by-phase timing for any newton example, headless (ViewerNull).

Usage:
    python time_example_phases.py <example_key>

Examples:
    python time_example_phases.py kamino_robot_dr_legs
    python time_example_phases.py robot_anymal_d
    python time_example_phases.py cloth_hanging
"""

import importlib
import os
import sys
import time

os.environ.setdefault("NEWTON_CACHE_PATH", r"C:\nc")

EXAMPLE_PKG_MAP = {
    "kamino_robot_dr_legs": "newton.examples.kamino.example_kamino_robot_dr_legs",
    "robot_anymal_d": "newton.examples.robot.example_robot_anymal_d",
    "cloth_hanging": "newton.examples.cloth.example_cloth_hanging",
}


def _apply_warp_variant():
    """Apply benchmark variant configs to Warp before any kernel compiles.

    BENCH_VARIANT=baseline      -> no overrides (default)
    BENCH_VARIANT=o2_nomathdx   -> optimization_level=2, enable_mathdx_gemm=False
    """
    variant = os.environ.get("BENCH_VARIANT", "baseline").lower()
    if variant in ("", "baseline", "default"):
        print(f"[variant] baseline (no Warp overrides)")
        return
    import warp as wp
    if variant == "o2_nomathdx":
        wp.config.optimization_level = 2
        wp.config.enable_mathdx_gemm = False
        print(f"[variant] o2_nomathdx: optimization_level=2, enable_mathdx_gemm=False")
    elif variant == "o3":
        wp.config.optimization_level = 3
        print(f"[variant] o3: optimization_level=3 (mathdx default)")
    elif variant == "o3_nomathdx":
        wp.config.optimization_level = 3
        wp.config.enable_mathdx_gemm = False
        print(f"[variant] o3_nomathdx: optimization_level=3, enable_mathdx_gemm=False")
    elif variant == "o2":
        wp.config.optimization_level = 2
        print(f"[variant] o2: optimization_level=2 (mathdx default)")
    elif variant == "o1":
        wp.config.optimization_level = 1
        print(f"[variant] o1: optimization_level=1 (mathdx default)")
    elif variant == "o0":
        wp.config.optimization_level = 0
        print(f"[variant] o0: optimization_level=0 (mathdx default)")
    elif variant == "nomathdx":
        wp.config.enable_mathdx_gemm = False
        print(f"[variant] nomathdx: enable_mathdx_gemm=False (optimization default)")
    else:
        print(f"[variant] WARN unknown BENCH_VARIANT={variant!r}, using baseline")


def main():
    if len(sys.argv) < 2:
        print("usage: time_example_phases.py <example_key>")
        sys.exit(2)
    example_key = sys.argv[1]
    if example_key not in EXAMPLE_PKG_MAP:
        print(f"unknown example: {example_key}")
        sys.exit(2)
    module_path = EXAMPLE_PKG_MAP[example_key]

    marks = {"start": time.perf_counter()}

    _apply_warp_variant()

    import newton  # noqa: F401
    import newton.examples  # noqa: F401
    mod = importlib.import_module(module_path)
    Example = mod.Example

    marks["after_import"] = time.perf_counter()

    viewer = newton.viewer.ViewerNull(num_frames=1)
    marks["after_viewer_ctor"] = time.perf_counter()

    # Use the example's own parser so we get its defaults, then force
    # headless-friendly fields.
    parser = Example.create_parser()
    ns = parser.parse_args([])
    ns.viewer = "null"
    ns.num_frames = 1
    ns.quiet = True
    ns.headless = True
    ns.benchmark = False
    ns.realtime = False
    ns.test = False
    # device may be missing in some examples; set if None
    if getattr(ns, "device", None) is None:
        ns.device = None

    example = Example(viewer, ns)
    marks["after_example_init"] = time.perf_counter()

    try:
        example.step()
    except Exception as e:
        print(f"WARN step failed: {e}")
    marks["after_step"] = time.perf_counter()

    try:
        example.render()
    except Exception as e:
        print(f"WARN render failed: {e}")
    marks["after_render"] = time.perf_counter()

    try:
        viewer.close()
    except Exception:
        pass
    marks["end"] = time.perf_counter()

    base = marks["start"]
    prev = base
    order = [
        "after_import",
        "after_viewer_ctor",
        "after_example_init",
        "after_step",
        "after_render",
        "end",
    ]
    print(f"=== PHASE TIMING (s) :: {example_key} ===")
    for name in order:
        t = marks[name]
        print(f"  {name:<24s}  total={t - base:7.2f}  delta={t - prev:7.2f}")
        prev = t
    print(f"TOTAL: {marks['end'] - base:.2f}s")


if __name__ == "__main__":
    main()
