# Dummy data (synthetic, for testing only)

These CSV files contain **randomly generated** data that mirror the **schema** (column names, directory layout) of the real UK Biobank, NHANES, and Inspiration4 data used in the FusionAge manuscript. They are intended for:

- Verifying that code runs end-to-end
- Understanding expected input/output formats
- CI smoke tests

**No real participant data is included.** Values are drawn from standard Gaussian distributions and do not reflect any biological measurements.

## Layout

```
dummy_data/
├── raw_data/
│   └── dummy_ukb_raw_extract.csv          # Mimics a UK Biobank raw extract
├── processed_data/
│   ├── time0_<MODALITY>.csv               # Modality feature matrices (22 modalities)
│   ├── time0_<MODALITY>_BA_<ALGO>.csv     # With predicted biological age columns
│   └── time2_<IMAGING>*.csv               # Imaging modalities (visit 2)
└── space_aging/
    └── dummy_inspiration4_scores.csv      # 4 astronauts × 3 timepoints × 3 clocks
```

## Real data access

- **UK Biobank:** https://biobank.ndph.ox.ac.uk/showcase/ (requires approved application)
- **NHANES:** https://wwwn.cdc.gov/nchs/nhanes/Default.aspx (public)
- **Inspiration4:** https://osdr.nasa.gov/bio/repo/data/missions/SpaceX%20Inspiration4 (public)
