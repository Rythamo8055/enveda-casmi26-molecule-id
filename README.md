# Enveda CASMI 2026: Molecule ID from Mass Spectra

This repository contains the solution pipeline, model architectures, validation frameworks, and decision logs for the **[Enveda CASMI 2026 Kaggle Competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)**.

## Overview
- **Objective:** Predict 2D chemical structures (SMILES) of small molecules from LC-MS/MS mass spectra.
- **Metric:** Mean Reciprocal Rank @ 25 (`MRR@25`) evaluated on RDKit canonical `InChIKey14` atom connectivity.
- **Novelty Regimes:** Class 1 (Library knowns), Class 2 (Known structure, novel spectra), Class 3 (Novel structures).

## Project Structure
```
├── DEVLOG.md             # Continuous record of requests, proposals, criteria, and decisions
├── README.md             # Project documentation and architecture guide
├── .gitignore            # Git exclusion rules for large datasets, parquets, and weights
├── data/                 # Local data directory (gitignored)
├── src/                  # Source modules
│   ├── preprocessing/    # Spectrum cleaning, de-adducting, formula generation
│   ├── models/           # Spectrum encoders, fingerprint predictors, de novo decoders
│   ├── retrieval/        # Spectral matching and candidate database indexing
│   └── evaluation/       # MRR@25 and InChIKey14 canonicalization utilities
└── notebooks/            # Exploratory and submission notebooks
```

## Architecture Summary
A 3-tier cascading hybrid pipeline:
1. **Tier 1 (Class 1 - Library Search):** Fast spectral entropy and cosine similarity search against ~2.5M training spectra.
2. **Tier 2 (Class 2 - Database Retrieval):** Deep spectrum encoder predicting molecular fingerprints (ECFP4/MACCS), ranked against candidate structures from natural product databases (COCONUT/LOTUS/PubChem) filtered by high-accuracy monoisotopic mass.
3. **Tier 3 (Class 3 - De Novo Generation):** Autoregressive/diffusion sequence generation for unmapped spectra.

## Decision Tracking
All pair-programming decisions, architectural evaluations, and roadmap changes are maintained in [DEVLOG.md](DEVLOG.md).
