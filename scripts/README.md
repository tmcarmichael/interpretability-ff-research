# Scripts

GPU experiment scripts. Results write to `results/`.

- `run_model.py` -- unified launcher for any HuggingFace model
- `nonlinear_probe.py` -- linear vs MLP probe comparison (per model)
- `split_bootstrap_gpu.py` -- document-level bootstrap (per model)
- `roc_width_sweep.py` -- output predictor bottleneck sweep
- `legacy/` -- per-model scripts from v1/v2 data collection
