# PassMoE Optimization Checklist

- [x] Frontier refreshed from current repo artifacts: validated route remains `qwen_fielddrop_base_identity_clixsense_500_raw`.
- [x] Primary optimize submode: `seed`.
- [x] Route mode: `exploit` the rescued PassMoE mechanism with a bounded expert-specialization diagnostic.
- [x] Recent optimization lessons reviewed from `README.md`, `PLAN.md`, `CHECKLIST.md`, and formal artifacts.
- [x] Brief slate checked for family diversity: choose router specialization before broader architecture changes.
- [x] Candidate brief updated: weakly supervised router/expert specialization.
- [x] Candidate ranking updated after smoke result.
- [x] Local smoke shows measurable specialization; promote only as a supplementary diagnostic line.
- [x] Current implementation pool recorded in `CANDIDATE_BOARD.md`.
- [x] Smoke queue defined: tiny CPU route-specialization diagnostic with `--router-specialization-weight > 0`, `--top-k-experts 1`, and `--skip-generation`.
- [x] Full-eval queue defined: optional Qwen/FieldDrop CUDA diagnostic with `--router-specialization-weight > 0`, `--top-k-experts 1`, and no replacement of the formal identity route unless SR validation passes.
- [x] Recent failures classified: unconstrained residual training hurt SR and should not replace the identity-foundation route.
- [x] Stagnation check performed: repeating residual-only training is low-information.
- [x] Family-shift trigger checked: add mechanism evidence rather than another same-loss retry.
- [x] Fusion eligibility checked: no new fusion is needed for this pass.
- [x] Next concrete action: implement default-off router specialization loss and expert specialization analysis.
