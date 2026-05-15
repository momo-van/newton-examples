# Newton 1.2 Cold-Start — Warp Flag Sweep

**Setup:** RTX 5000 Ada Laptop, Warp 1.13.0, CUDA Toolkit 12.9. 3 iters per variant, Warp + NVRTC caches wiped before every cold run. 2026-05-14.

## Recommendation

**Ship `warp.config.enable_mathdx_gemm = False`.** Saves 27% on Kamino cold-start, no measurable effect on MuJoCo or VBD. Do **not** also change `optimization_level` — the default (= O3 on CUDA) is correct for all three solvers.

## Cold-start total (median seconds; % vs baseline)

| Variant | Kamino | MuJoCo | VBD |
|---|---:|---:|---:|
| `baseline` (`opt=None`, mathdx on) | **132.5** | **142.4** | **61.9** |
| `o3` (sanity) | 130.3 (−1.7%) | 139.0 (−2.4%) | 62.5 (+1.0%) |
| **`o3_nomathdx`** ✅ | **96.9 (−27%)** | 139.8 (−2%) | 61.3 (−1%) |
| `o2_nomathdx` | 89.2 (−33%) | 162.3 (+14%) ❌ | 78.4 (+27%) ❌ |
| `o1` | 151.7 (+15%) ❌ | 216.1 (+52%) ❌ | 121.5 (+96%) ❌ |

Warm-start unchanged (±4%) for all variants except `o1`, which regresses warm-start 9–11% on MuJoCo/VBD (larger cached binaries load slower).

## Decomposition

The two flags are independent and have different signatures:

| Effect | Kamino | MuJoCo | VBD |
|---|---:|---:|---:|
| `nomathdx` alone (vs baseline at O3) | **−33s** | +2s | +0s |
| `O2` (vs default O3) | −8s | **+20s** | **+17s** |

The Kamino win in the originally-proposed combined flip was almost entirely the **mathdx** part; bundling O2 with it was net-negative across MuJoCo/VBD.

## Why

- **`enable_mathdx_gemm` gates only `wp.tile_matmul`** (cuBLASDx GEMM path), per `warp/_src/builtins.py:12915`. Kamino's `llt_blocked` (−20.5s) and `fk.kernels` (−14.1s) use `tile_matmul` heavily — without cuBLASDx LTOs they compile in a fraction of the time. MuJoCo's Cholesky goes through `tile_cholesky` (cuSOLVERDx, no equivalent flag) and is unaffected. VBD doesn't use either path heavily.
- **Lowering optimization** (O2, O1) skips inlining/DCE, so larger IR reaches the SASS backend. MuJoCo/VBD kernels (many small helpers, per-case branches) are SASS-bound and regress sharply. Kamino's two giant kernels are optimizer-bound and benefit slightly at O2, but everything (including Kamino) loses badly at O1.
- **Warp default `optimization_level=None` resolves to `3` on CUDA** (`context.py:2964`). Hence `o3` ≈ `baseline`.

## Caveats

- Cold-start only — `enable_mathdx_gemm=False` may slow `tile_matmul` at runtime. Validate with a Kamino-heavy step-time benchmark before shipping. If needed, scope per-module: `wp.set_module_options({"enable_mathdx_gemm": False})`.
- On CUDA toolkits < 12.9, `optimization_level` changes are silently ignored (`context.py:2967`). The mathdx flag works on all versions.

---

*Raw data: `results/solver_rigorous{,_o3,_o3_nomathdx,_o2_nomathdx,_o1}.{txt,csv}` • Tools: `compare_variants.py`, `diff_kernels.py` • Runner: `VARIANT=<name> bash bench_solver_rigorous.sh`*
