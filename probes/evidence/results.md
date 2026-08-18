# Stage-0 probe — raw results

GPU: NVIDIA A10G (sm_86) · driver 595.71.05 · CUDA 13.2 ·
Compute Sanitizer 2026.1.0.0 · cuda-oxide `50d07314` · 2026-08-18

Every probe: **completed**, exit 0, all elements written, `ERROR SUMMARY: 0 errors`.

| probe | block | raw | synccheck | output |
|---|---:|---|---|---|
| safe_barrier | 32/64/128 | completed | 0 errors | all `1` |
| divergent_barrier | 32/64/128 | completed | 0 errors | all `1` |
| warp_divergent_barrier | 64/128 | completed | 0 errors | all `1` |
| mask_full | 32/64/128 | completed | 0 errors | all `0xffffffff` |
| mask_shrunk | 32/64/128 | completed | 0 errors | all `0xffffffff` |

Note the first probe run (v1 of the harness) reported 24/24
`watchdog-fired`; that was the watchdog killing compilation, not the GPU.
See `docs/research-baseline.md` §9.2.
