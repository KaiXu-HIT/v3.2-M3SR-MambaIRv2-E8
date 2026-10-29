# E8 verification — 2026-10-29

Workspace: `D:\Code\Python\M3SR-MambaIRv2`.

Passed offline checks:

- E8 readiness acceptance/rejection; protection of original DIV2K and five-set paths; explicit DF2K multi-root configurations; equal added RGB/UDR updates; same-scale fixed teacher; Y-channel/crop=scale for x4/x2/x3.
- All three uncertainty variants at all three scales: CPU reference selective-scan forward/backward, strict state transfer, identical RGB/UDR fresh PixelShuffle at a scale change; non-upsampler mismatch rejection.
- Actual E4 model A/B optimizer steps at x2 and x3 with frozen U3 teacher and correctly sized uncertainty targets; intended gradient/freeze groups verified.
- Known Conv/Linear arithmetic count and nonzero functional scan/projection/attention cost accounting.
- Existing E4 CPU reference suite and E6 independent-seed configuration/report suite.
- Python 3.8 AST syntax checks; `git diff --check`.

The CPU selective-scan recurrence exercises production architecture/model source through the existing isolated loader. It does not validate a compiled CUDA kernel or establish performance.

Local production-CUDA smoke test could not import BasicSR because `mamba_ssm` and `selective_scan_cuda` are absent in the local `mamba` environment (PyTorch 2.7.1+cu118; RTX 3090 detected). No environment/package replacement was performed. E4/E5/E6/E7 measured reports and E6 trained checkpoint plan are also absent locally. Consequently no DF2K training, real five-set scores, measured latency/VRAM, estimator cost or public-model score is claimed.

Server inputs still required: completed passing E4–E7 reports and original seed-specific weights; full DF2K LR/HR/Depth and scale-specific benchmark folders; the original Depth estimator adapter/checkpoint; exact public checkpoints/configs and corresponding official architecture adapters. The scripts validate and measure these inputs when supplied; they do not substitute synthetic performance or guessed public-model results.
