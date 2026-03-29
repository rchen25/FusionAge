# FusionAge

**FusionAge: A Multimodal Machine Learning Framework for Biological Age Estimation**

This repository hosts the **FusionAge Python package**, a **minimal runnable example** on synthetic data, and **supplementary tables/figures** for the manuscript. It does **not** include full UK Biobank or NHANES analysis notebooks or large processed matrices (those belong in a controlled, data-access-compliant environment).

> **Publication:** *FusionAge framework for multimodal machine learning-based aging clocks uncovers cardiorespiratory fitness as a major driver of aging and inflammatory drivers of aging in response to spaceflight*  
> Robert Chen, Nicholas Bartelo, Mohith Arikatla, Christopher E. Mason, Olivier Elemento — Weill Cornell Medicine, New York, NY, USA

## Contents

| Path | Purpose |
|------|---------|
| `FusionAge/` | Library: model builders (`model.py`), cross-validation (`training.py`), metrics & age acceleration (`performance_evaluation_stats.py`), PhenoAge/SHAP helpers (`utils.py`). |
| `examples/` | `minimal_train_and_predict.py` — smoke test with **dummy** Gaussian data (no real cohorts). |
| `supplementary_material/` | Journal supplementary tables (Word/Excel) and supplementary figures document from the NPJ Digital Medicine submission. |
| `environment.yml` | Conda-oriented dependencies for the library + example. |

## Quick start

```bash
conda env create -f environment.yml
conda activate fusionage
python examples/minimal_train_and_predict.py
```

## Manuscript alignment

The paper describes **26 clocks** (22 modality-specific + 4 multimodal) trained with linear models, Lasso, ElasticNet, XGBoost, and DNNs. This repo documents the **shared code patterns** for those algorithms; reproducing paper numbers requires the original cohort builds and notebooks under institutional data agreements.

## Data availability (paper)

- UK Biobank: https://biobank.ndph.ox.ac.uk/showcase/  
- NHANES: https://wwwn.cdc.gov/nchs/nhanes/Default.aspx  
- Inspiration4: https://osdr.nasa.gov/bio/repo/data/missions/SpaceX%20Inspiration4  

## License

MIT License.
