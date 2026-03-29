# Examples

- **`minimal_train_and_predict.py`** — fits `FusionAge.model.build_linear_regression()` on random Gaussian features (no PyTorch, no real cohort data).

```bash
# from repository root
python examples/minimal_train_and_predict.py
```

DNN training (`build_fusionage_dnn`, `train_dnn`) needs **PyTorch**; see `environment.yml`.

Full paper analyses (all modality-specific and multimodal notebooks, large matrices, and NHANES/UK Biobank pipelines) are intentionally **not** hosted on GitHub; run those in a secure environment with approved data access.
