# PassMoE Candidate Board

| Candidate ID | Level | Parent | Strategy | Status | Expected Gain | Observed Result | Promote / Archive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cand-specialize-router-001 | implementation | qwen_fielddrop_base_identity_clixsense_500_raw | exploit | qwen_cuda_passed | Show that PassMoE can learn interpretable PII / entropy / leet routing without changing the validated default run | Tiny 64-record weak-label top-1 agreement moved from `0.2813` to `1.0000`; Qwen/FieldDrop CUDA 256-record diagnostic moved from `0.3008` to `1.0000`, with trained bucket weights PII `0.9806`, entropy `0.9720`, leet `0.9832` | promote as supplementary mechanism diagnostic only; not an SR@K result |

## Active Brief

`cand-specialize-router-001` keeps the imported foundation and identity-safe expert path unchanged by default. When enabled explicitly, it adds a weak heuristic target over the three PassMoE expert slots: PII, entropy, and leetspeak/morphology. Success is measured by router specialization metrics, not by claiming a new SR@K main result unless a separate formal run validates it. The Qwen/FieldDrop CUDA run confirms mechanism-level expert separation on the real foundation, while FieldDrop remains an external PassLLM adapter.
