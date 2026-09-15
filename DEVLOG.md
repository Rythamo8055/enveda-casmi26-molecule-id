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
- [ ] **External Candidate DB Size vs Memory**: Determine the exact candidate database format (e.g., LMDB, Feather, Parquet, or SQLite with FAISS/HNSW indexing) that fits within Kaggle's 16 GB/30 GB RAM limits without internet access.
- [ ] **Spectrum Encoder Backbone Selection**: Decide between:
  - *Option A*: Binned peak representation with 1D-CNN / MLP.
  - *Option B*: Peak list Transformer (treating $m/z$ and intensity as continuous tokens, e.g. MassFormer / DreaMS style).
- [ ] **Collision Energy & Adduct Conditioning**: Choose whether to embed collision energy (`collision_energy_ev`) and adduct types as conditioning tokens or to normalize fragmentation patterns across energies.
- [ ] **Baseline Implementation**: Build the Tier-1 spectral library matcher first to establish an empirical local CV score and verify submission pipeline validity.
