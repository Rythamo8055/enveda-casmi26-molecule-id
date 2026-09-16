# CASMI 2026 — Priority Implementation Plan

> Goal: Maximize MRR@25 on the Enveda CASMI 2026 Kaggle competition.
> Current baseline: ~0.285 (spectral library matching only).
> Target: 0.70-0.85 (top-5 contention).

---

## Why This Order?

Each technique is ranked by **impact-to-effort ratio**. P0 items give the biggest score jump for the least work. P2 items are polish. The order matters — don't skip ahead.

---

## P0 — Must Do (Expected: +0.25-0.45 MRR)

### 1. Contrastive Spectrum-Molecule Alignment

**What:** Train two encoders — one for spectra, one for molecules — that map matching pairs to nearby points in a shared embedding space. At inference, embed the query spectrum and rank all candidate molecules by cosine similarity.

**Why:** This is the single highest-impact technique. FLARE achieved 43% rank@1 on MassSpecGym. JESTR outperformed SIRIUS by 238%. MVP achieved 36% rank@1 with consensus spectra. Every top-performing method on MassSpecGym uses contrastive learning. Your current MLP-based approach has no cross-modal alignment — it predicts fingerprints in isolation, which is a fundamentally weaker objective.

**Implementation:**
- Spectrum encoder: Transformer or 1D-CNN on peak lists (not binned vectors)
- Molecule encoder: GNN on molecular graph or pretrained ChemBERTa
- Loss: InfoNCE / NT-Xent contrastive loss
- Training: 500K+ spectrum-molecule pairs from train.parquet
- Runtime: Kaggle GPU (T4/P100), ~2-4 hours training

**Expected gain:** +0.20-0.35 MRR

---

### 2. In-Silico Data Augmentation

**What:** Use FIORA, CFM-ID, or FragGenie to simulate MS/MS spectra from molecular structures. Train on both real and synthetic spectra.

**Why:** You have 2.5M real spectra. With simulation, you can get 40M+. SEISMiQ trained on 41M simulated spectra and achieved 29.8% top-1 on MassSpecGym. FIORA generates spectra that outperform CFM-ID. More training data directly improves model generalization, especially for Class 2 and Class 3 molecules where real spectra don't exist.

**Implementation:**
- Download 500K molecules from PubChem/COCONUT/ZINC
- Simulate spectra at 3 collision energies using FIORA or CFM-ID
- Add realistic noise (m/z jitter, intensity scaling, random peak dropping)
- Mix synthetic data with real training data (80/20 ratio)
- Train on the combined dataset

**Expected gain:** +0.10-0.20 MRR

---

### 3. Multi-Spectrum Consensus Aggregation

**What:** When a test molecule has multiple spectra (different collision energies, adducts), don't just average predictions. Build a consensus representation that captures complementary information from each spectrum.

**Why:** Your test molecules have 1-9 spectra each. Naive averaging of fingerprint predictions destroys collision-energy-specific diagnostic peaks. The MVP paper showed that "aggregate-then-rank" using consensus spectra outperforms individual spectrum annotation by 147%. Each collision energy reveals different structural information — low CE shows intact substructures, high CE shows backbone fragments.

**Implementation:**
- Option A: Concatenate all peak lists into a single consensus spectrum, add collision energy as a feature
- Option B: Score each candidate separately per spectrum, then use Reciprocal Rank Fusion (RRF): `score = sum(1/rank_i)` across all spectra
- Option C: Train a learned attention mechanism that weights each spectrum's contribution

**Expected gain:** +0.05-0.10 MRR

---

## P1 — Should Do (Expected: +0.10-0.20 MRR)

### 4. Ensemble of Complementary Models

**What:** Train 2-3 different model architectures (MLP, GNN, Transformer), learn a per-molecule weighting that combines their predictions to maximize ranking.

**Why:** Different architectures capture different aspects of the spectrum-molecule relationship. MLPs capture global patterns, GNNs capture local substructure, Transformers capture peak dependencies. ESP showed that ensembling MLP and GNN improves average rank by 23-37% over either alone. No single model is best for all molecules.

**Implementation:**
- Model A: MLP fingerprint predictor (your current approach, improved)
- Model B: GNN molecular encoder with contrastive loss
- Model C: Transformer spectrum encoder with attention
- Ensemble head: LightGBM or simple linear layer that learns per-molecule weights based on validation performance

**Expected gain:** +0.05-0.10 MRR

---

### 5. BPE Tokenization for SMILES Generation

**What:** Use Byte Pair Encoding to tokenize SMILES strings, allowing the model to learn common molecular substructures (like "c1ccccc1" for benzene ring) as single tokens.

**Why:** Character-level SMILES generation makes many syntax errors. BPE reduces the sequence length and lets the model learn meaningful chemical substructures as atomic units. MS2Mol used BPE and achieved 62% meaningful similarity on dark chemical space — the best de novo result published. For Class 3 molecules where no database match exists, this is your only option.

**Implementation:**
- Train BPE tokenizer on all SMILES in train.parquet + COCONUT
- Use BPE-encoded SMILES as decoder target in seq2seq model
- Encoder: spectrum features
- Decoder: autoregressive BPE SMILES generation
- Generate multiple candidates per spectrum, rank by confidence

**Expected gain:** +0.05-0.10 MRR (primarily for Class 3)

---

### 6. RAG-Enhanced De Novo Generation

**What:** For novel molecules, retrieve the top-10 most similar spectra from training data and use them as few-shot examples to guide structure generation.

**Why:** Pure de novo generation from scratch is unreliable. RAG grounds the generation in real examples. MolE-RAG improved molecular property prediction by 28 ROC-AUC points without any training. CLADD outperformed fine-tuned LLMs on drug discovery tasks. For Class 3 molecules, this bridges the gap between "generate from nothing" and "retrieve from database."

**Implementation:**
- Pre-compute embeddings for all training spectra
- For each test spectrum, retrieve top-10 nearest neighbors
- Feed retrieved (spectrum, SMILES) pairs as context to the generator
- Generator produces candidates biased toward structurally similar knowns

**Expected gain:** +0.03-0.08 MRR

---

## P2 — Nice to Have (Expected: +0.03-0.08 MRR)

### 7. Masked Peak Pretraining (PRISM-style)

**What:** Pretrain the spectrum encoder by masking 20% of peaks and predicting their masses. This teaches the model the "grammar" of fragmentation without requiring labels.

**Why:** PRISM trained on 1.2B spectra and showed 23% improvement on downstream tasks. Self-supervised pretraining on large unlabeled data creates better representations that transfer to downstream tasks. Even at smaller scale (2.5M spectra), this should improve your encoder quality.

**Implementation:**
- Randomly mask 20% of peaks in each training spectrum
- Train encoder to predict masked peak masses
- Use pretrained encoder as initialization for contrastive training
- Adds ~1-2 hours of GPU training time

**Expected gain:** +0.03-0.05 MRR

---

### 8. Bond-Breaking GNN (FIORA-style)

**What:** Model mass spectrometry as a graph-level task where each bond has a learned probability of breaking, producing specific fragment ions.

**Why:** FIORA outperforms CFM-ID and ICEBERG in spectral prediction. This approach captures the physical process of fragmentation more accurately than peak-binning. It also generates better in-silico spectra for data augmentation.

**Implementation:**
- Train GNN to predict bond-breaking probabilities from molecular graph
- Use predicted fragments to score candidate molecules
- Can also generate synthetic spectra for augmentation

**Expected gain:** +0.02-0.05 MRR

---

### 9. Collision Energy Conditioning

**What:** Embed collision energy as a continuous input feature to the spectrum encoder, so the model learns CE-dependent fragmentation patterns.

**Why:** Your test set uses Bruker timsTOF with specific collision energies. Training data has mixed instruments with different CE scales. Conditioning on CE allows the model to generalize across instruments and produce CE-aware predictions.

**Implementation:**
- Add CE as a continuous scalar input (sinusoidal encoding or learned embedding)
- Concatenate with spectrum features before the encoder
- Normalize CE to [0, 1] range across training data

**Expected gain:** +0.02-0.04 MRR

---

### 10. Candidate Re-Ranking Head

**What:** Instead of using raw similarity scores for ranking, train a lightweight gradient-boosted ranker (LambdaMART/ListNet) that takes multiple signals and learns to optimize MRR directly.

**Why:** Your current ranking uses `0.7 * tanimoto + 0.3 * frag_score`. This fixed weighting is suboptimal. A learned ranker can combine spectral similarity, fingerprint similarity, formula score, molecular weight, and other features to produce an MRR-optimized ranking.

**Implementation:**
- Extract features for each (spectrum, candidate) pair: spectral cosine, fingerprint tanimoto, formula match, mass error, fragmentation score
- Train LightGBM with LambdaRank objective on validation set
- Replace fixed scoring with learned ranking

**Expected gain:** +0.02-0.05 MRR

---

## Implementation Timeline

| Week | Focus | Expected MRR |
|------|-------|-------------|
| Week 1 | Contrastive model (P0 #1) + data augmentation (P0 #2) | 0.50-0.55 |
| Week 2 | Multi-spectrum fusion (P0 #3) + first Kaggle submission | 0.55-0.65 |
| Week 3 | Ensemble (P1 #4) + BPE generation (P1 #5) | 0.65-0.75 |
| Week 4 | RAG (P1 #6) + optimizations (P2) | 0.75-0.85 |

---

## What Each Technique Replaces

| Current Component | Problem | Replacement |
|---|---|---|
| 2-layer MLP on binned peaks | No cross-modal learning | Contrastive Transformer encoder |
| BCE loss on fingerprints | Optimizes wrong objective | InfoNCE contrastive loss |
| Naive fingerprint averaging | Destroys CE-specific info | Consensus spectrum or RAF |
| Fixed 0.7/0.3 scoring | Suboptimal weighting | Learned LambdaMART ranker |
| Character-level SMILES | Syntax errors in generation | BPE tokenization |
| Single model | Limited coverage | Ensemble of 3 architectures |
| 30K training samples | Massive underfitting | 500K+ real + 1M synthetic |
| No de novo capability | Class 3 gets zero | BPE seq2seq + RAG |

---

## Hardware Requirements

| Component | Local Machine | Kaggle GPU |
|---|---|---|
| Contrastive training | NO (no GPU) | YES (2-4 hrs) |
| Data augmentation | YES (CPU, hours) | YES (faster) |
| Multi-spectrum fusion | YES (seconds) | YES |
| Ensemble training | YES (minutes) | YES |
| BPE generation training | NO (needs GPU) | YES (1-2 hrs) |
| Inference | YES (all components) | YES |
| Submission upload | YES (via API) | YES |

**Bottom line:** Train on Kaggle GPU, export checkpoints, run inference locally or on Kaggle CPU.

---

*Generated for: enveda-casmi26-molecule-id*
*Last updated: 2026-09-16*
