# FusionAge

**FusionAge: A Multimodal Machine Learning Framework for Biological Age Estimation**

FusionAge is a framework for training multimodal aging clocks using interpretable nonlinear models, including deep neural networks (DNNs). It integrates diverse data types — biospecimens, physical measures, functional tests, and biomedical imaging — to produce a holistic estimate of biological age that captures the complex, multifactorial nature of human aging.

> **Publication:** *FusionAge framework for multimodal machine learning-based aging clocks uncovers cardiorespiratory fitness as a major driver of aging and inflammatory drivers of aging in response to spaceflight*
>
> Robert Chen, Nicholas Bartelo, Mohith Arikatla, Christopher E. Mason, Olivier Elemento
>
> Weill Cornell Medicine, New York, NY, USA

## Overview

Traditional epigenetic aging clocks are limited because they do not incorporate clinical information and functional tests, and rely on DNA samples and methylation profiling infrastructure which are not easily accessible. FusionAge addresses these shortcomings through:

- **Multimodal Architecture:** Integrates diverse data types — including imaging and functional tests common in healthcare settings — and explicitly tests different fusion strategies (early fusion and late fusion) to create a more holistic model of aging.
- **Data-Driven Feature Modalities:** Groups features by their data collection modality rather than pre-selecting features based on organ systems, allowing unbiased discovery of the most predictive features for disease and mortality.
- **Nonlinear Modeling:** Employs a thorough model selection process including nonlinear models (XGBoost, deep neural networks) that capture complex, non-additive relationships between features and age.
- **Portability and Interpretation:** Uses data modalities common in clinical settings for portability to new datasets, and includes an interpretability module (SHAP-based) for understanding feature contributions to biological age at both population and individual levels.

## Aging Clocks

A total of **26 aging clocks** were trained: 22 modality-specific clocks and 4 multimodal clocks.

### Modality-Specific Clocks (22)

| Category | Modality |
|---|---|
| **Physical Measures** | Anthropometric, Arterial Stiffness, Bone Density, Body Impedance, Vital Signs |
| **Biospecimens** | Blood Chemistry, Urine Chemistry, Metabolomics, Proteomics, Telomere |
| **Functional Tests** | Cognitive Test, Eye Measures, Cardiorespiratory Fitness, Skeletal Muscle Strength, Hearing Test, Spirometry |
| **Imaging** | Abdominal MRI, Brain MRI, Carotid Ultrasound, DEXA Scan, ECG, Heart MRI |

### Multimodal Clocks (4)

| Clock | Architecture | Feature Set |
|---|---|---|
| ALL_EARLYFUSION | Early Fusion | All non-imaging modalities |
| ALL_LATEFUSION | Late Fusion | All non-imaging modalities |
| IMAGING_EARLYFUSION | Early Fusion | All imaging modalities |
| IMAGING_LATEFUSION | Late Fusion | All imaging modalities |

### Regression Algorithms

For each clock, models are trained with: Linear Regression, Lasso, ElasticNet, XGBoost, and Deep Neural Network (DNN).

Clock naming convention: `FusionAge-<ALGORITHM>-<FEATURE_SET>` (e.g., `FusionAge-DNN-ALL_LATEFUSION`).

## Key Results

- **FusionAge-DNN-ALL_LATEFUSION** achieved Pearson R=0.95, MAE=2.0 years, RMSE=2.4 years — outperforming PhenoAge (R=0.75), YeTianSVM (R=0.66), and OrganAge (R=0.81).
- FusionAge-derived biological age is more strongly associated with incident disease and mortality than chronological age, outperforming previously reported linear aging clocks in 24 of 30 aging-associated diseases.
- **Cardiorespiratory fitness** is validated as a major, consistent driver of biological age across both UK Biobank and NHANES cohorts (30% and 21% of aggregate SHAP contributions, respectively).
- Application to the **Inspiration4** astronaut cohort demonstrates FusionAge's utility for detecting spaceflight-induced biological age changes, uncovering putative inflammatory and tissue remodeling pathways (NT5C3A, APLP1, COL6A3).

## Datasets

| Dataset | Description | N |
|---|---|---|
| **UK Biobank** | Training dataset; individuals aged 37–73 | 502,366 |
| **NHANES** | External validation dataset (1999–2017) | 101,316 |
| **Inspiration4** | Spaceflight use case; 4 astronauts, longitudinal timepoints | 4 |

Data sources are publicly available:
- UK Biobank: https://www.ukbiobank.ac.uk
- NHANES: https://wwwn.cdc.gov/nchs/nhanes/Default.aspx
- Inspiration4: https://osdr.nasa.gov/bio/repo/data/missions/SpaceX%20Inspiration4

## Repository Structure

```
FusionAge/
├── FusionAge/                         # Core FusionAge module
│   ├── model.py                       # Model architectures
│   ├── training.py                    # Training pipeline
│   ├── performance_evaluation_stats.py # Evaluation metrics
│   └── utils.py                       # Utility functions
├── data_processing/                   # Data extraction and feature construction
├── model_training_and_evaluation/     # Model training notebooks and results
│   ├── model_*.ipynb                  # Modality-specific clock training
│   ├── validation_*.ipynb             # External validation (NHANES)
│   ├── clock_performance_analysis/    # Performance metrics and comparisons
│   └── disease_mortality_association/ # Disease and mortality association analyses
├── space_aging/                       # Spaceflight (Inspiration4) analysis
├── environment.yml                    # Conda environment specification
└── README.md
```

## Setup

### Environment

```bash
conda env create -f environment.yml
```

### Dependencies

Key packages: Python, scikit-learn (v1.0.2), xgboost (v1.6.2), PyTorch (v1.13.1), SHAP.

See `environment.yml` for the full list of dependencies.

## License

This code is released under the MIT License.
