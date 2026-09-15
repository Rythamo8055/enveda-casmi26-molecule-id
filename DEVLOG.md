# Project Devlog: Enveda CASMI 2026 (Molecule ID from Mass Spectra)

This devlog tracks all architectural, engineering, and experimental decisions across pair-programming sessions.

Each entry follows this structured decision framework:
1. **User Request**: What was asked.
2. **Submitted Proposals**: Options, analysis, and solution proposals submitted by the assistant.
3. **Selection Criteria**: Criteria and constraints used to evaluate choices.
4. **Final Decision**: The chosen path and rationale.
5. **Decisions Left for Next Steps**: Open decisions and branch points reserved for subsequent iterations.

---

## Session 01: Initial Competition Analysis & System Architecture Design
- **Date**: 2026-09-15
- **Context**: Official launch of the Kaggle Enveda CASMI 2026 challenge.

### 1. User Request
- Analyze the Kaggle competition: `https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra`.
- Establish a structured devlog framework tracking requests, proposals, selection criteria, final decisions, and next steps.
- Initialize and synchronize the codebase with GitHub.

### 2. Submitted Proposals
The assistant fetched live competition metadata, schema, rules, and leaderboard, proposing three primary architectural components:

1. **Problem Framing & Metric Optimization Strategy**:
   - Metric is **MRR@25** evaluated strictly on **InChIKey14** (first block of InChIKey, atom connectivity skeleton after RDKit canonical tautomer standardization).
   - Proposal: Enforce **strict InChIKey14 deduplication** in post-processing. Because stereoisomers and tautomers collapse to identical InChIKey14, submitting tautomer/stereoisomer duplicates in the top-25 candidate list wastes slots and degrades MRR.

2. **Multi-Tier Modeling Strategy for the 3 Novelty Regimes**:
   - **Tier 1 (Class 1 - In public spectral libraries)**: Spectral similarity search against reference training libraries (`train.parquet`, GNPS, MassBank, MoNA).
   - **Tier 2 (Class 2 - Known structure, no public spectra)**: MS-to-Fingerprint retrieval. Predict molecular fingerprints (Morgan, MACCS, PubChem) from MS/MS spectra and rank candidate structures from external databases (PubChem / COCONUT / LOTUS) filtered by precursor mass/formula.
   - **Tier 3 (Class 3 - Novel structures)**: *De novo* structure generation using autoregressive or diffusion models (e.g., Spec2Mol, MSNovelist, ChemBERTa/BART-based decoders).

3. **Validation & Fusion Strategy**:
   - **Multi-Spectrum Aggregation**: Test inputs provide 1 to 16 spectra per molecule at varying collision energies and adducts. Fusion must occur before top-25 ranking (e.g. Reciprocal Rank Fusion or averaged fingerprint embeddings).
   - **Validation Split**: Hold out `enveda-np-examples` (250 compounds on Bruker timsTOF matching test set instrumentation and pipeline) as the primary domain-specific validation benchmark.

### 3. Selection Criteria
- **Metric Maximization**: MRR@25 awards maximum points to correct rank-1 predictions (1.0 vs 0.5 vs 0.04). Top ranks must be high precision.
- **Hardware & Runtime Feasibility**: Kaggle notebook environment imposes a strict $\le 9$-hour GPU/CPU limit with **no internet access** during test re-run. Heavy retrieval databases must be pre-indexed and lightweight enough to run offline within memory constraints.
- **Instrument Domain Shift**: Training data contains mixed instruments (Orbitrap, QTOF, timsTOF) with heterogeneous collision energy scales (NCE vs eV). The test set exclusively uses Bruker timsTOF with collision energies in eV.
- **Novelty Generalization**: Spectral library matching alone will fail on Classes 2 and 3 (~half or more of the test set). Candidate retrieval and generative modeling are mandatory for competitive standing.

### 4. Final Decision
1. **Adopt a 3-Tier Cascading Architecture**:
   - **Tier 1**: Exact/near-exact spectral matching against training data + reference libraries for fast, high-confidence identification of Class 1 compounds.
   - **Tier 2**: Deep spectrum-to-fingerprint network (Transformer/1D-CNN) querying an offline filtered candidate pool (PubChem/COCONUT constrained by precursor $m/z \pm 10\text{ ppm}$) for Class 2.
   - **Tier 3**: Conditional generative fallback for unretrieved/novel molecules (Class 3).
2. **Standardize on InChIKey14 Post-Processing**: Every prediction pipeline must filter candidate lists so all 25 submitted SMILES possess unique InChIKey14 hashes.
3. **Primary Validation Anchor**: Use `enveda-np-examples` as local validation ground truth to benchmark baseline MRR@25 against the public leaderboard (~0.285 initial top score).
4. **Devlog & Repo Setup**: Maintain this `DEVLOG.md` as the source of truth for all architectural decisions, initialized with a public GitHub repository.

### 5. Decisions Left for Next Steps
- [x] **Local Environment Setup**: Verified system specs and created `.venv` via `uv` containing RDKit, Polars, Matchms, Scipy, and Numba.
- [x] **Baseline Implementation**: Built Tier-1 spectral library matcher and end-to-end submission pipeline.
- [ ] **External Candidate DB Size vs Memory**: Determine the candidate database format (e.g., SQLite/Parquet indexed by mass) that fits within Kaggle's memory limits without internet access.
- [ ] **Spectrum Encoder Backbone Selection**: Choose between Peak list Transformer vs binned representations for Tier 2.
- [ ] **Collision Energy & Adduct Conditioning**: Choose embedding strategy for multi-energy spectra.

---

## Session 02: System Capability Verification & Baseline Implementation
- **Date**: 2026-09-15
- **Context**: Selecting baseline strategy, auditing local hardware, and implementing the Tier-1 Spectral Library Matcher.

### 1. User Request
- Define the roadmap to build a competitive, winning solution.
- Verify whether the user's system supports the required development stack (`rdkit`, `matchms`, `scipy`, `numpy`, `polars`, `pyarrow`, `torch`).
- Selected **Option 1 (Fast Spectral Matching Baseline)** to implement first.

### 2. Submitted Proposals
1. **Hardware & Capability Audit**:
   - Analyzed local system: Linux (Fedora 44 / x86_64), 16 vCPUs (13th Gen Intel Core i7-1360P), 15 GiB RAM, 233 GiB available NVMe SSD.
   - Result: Fully supported for rapid local feature processing, spectral indexing, and CPU inference, with Kaggle GPU resources available for heavy deep learning runs.
2. **5-Pillar Winning Strategy**:
   - Pillar 1: High-accuracy mass de-convolution & adduct offset calculation ($\pm 10\text{ ppm}$).
   - Pillar 2: High-precision spectral library engine for Class 1 (entropy/cosine similarity).
   - Pillar 3: MS-to-Fingerprint retrieval with offline candidate database (COCONUT/PubChem) for Class 2.
   - Pillar 4: De novo generative model (Seq2Seq Transformer) for Class 3.
   - Pillar 5: Multi-spectrum aggregation and strict `InChIKey14` candidate deduplication.
3. **Execution Choice**:
   - Evaluated Option 1 (Fast Spectral Matching baseline) vs Option 2 (Direct Deep Learning). Option 1 establishes an immediate empirical score and verified submission pipeline.

### 3. Selection Criteria
- **Execution Speed & Simplicity**: Fast library search establishes an end-to-end baseline in minutes and tests the full submission format.
- **Metric Fit**: Exact mass + cosine similarity provides high precision on Class 1 molecules, directly targeting the initial public LB benchmark (~0.285).
- **Environment Stability**: Pinned `rdkit==2026.3.6` matches Kaggle's evaluation environment.

### 4. Final Decision
1. **Configured Local Environment via `uv`**: Installed `rdkit`, `matchms`, `polars`, `pyarrow`, `scipy`, `numpy`, and `numba` into `.venv`.
2. **Implemented Preprocessing & Adduct Tables**: Created `src/preprocessing/adducts.py` supporting all 10 competition adducts, neutral mass conversion, and ppm tolerance checking.
3. **Implemented Numba-Accelerated Matcher**: Created `src/retrieval/spectral_matcher.py` featuring square-root weighted cosine similarity and binary search indexing (`SpectralLibraryIndex`).
4. **Built End-to-End Pipeline**: Created `src/pipeline.py` with multi-spectrum aggregation per `molecule_id` and strict `InChIKey14` deduplication, verified against synthetic test data.

### 5. Decisions Left for Next Steps
- [ ] **Accept Kaggle Rules**: Accept terms on the Kaggle web interface to allow CLI downloading of `train.parquet` and `test.parquet`.
- [ ] **Benchmark on Local CV**: Run the spectral matcher on `enveda-np-examples` to record our first local MRR@25 baseline score.
- [ ] **Self-Contained Notebook**: Generate a standalone Kaggle Notebook script (`notebooks/baseline_submission.py`) ready for 1-click submission.
- [ ] **Phase 2 Candidate Database**: Curate and index open natural products (COCONUT 2.0 / LOTUS) by neutral mass for Class 2 retrieval.

