# Toward Dual-Path Architectures for Neural Network Observability

Can a neural network's internal structure tell you something about its decisions that output confidence does not? This project tests that question systematically: compare representations under different training rules, then try to read decision-quality signals from BP activations using passive observers and co-training.

**Thesis:** Structural legibility and faithful observability are distinct problems. Local training rules produce measurably different representation structure (Phase 1), but passive observers on vanilla BP activations do not recover reliable per-example signal beyond confidence (Phases 2-3). Observability may need to be explicitly trained rather than passively extracted.

## At a glance

**Core question:** Can internal structure in standard BP models be read in a way that adds information beyond output confidence?

**Current answer:** Phase 1 confirms that training objective changes representation structure. Phase 2 shows that passive observer readouts on BP activations mostly collapse after proper controls. A denoising auxiliary produces a small positive signal. The picture is not binary: per-example faithfulness mostly fails, but FF-derived signals still identify causally important neurons under ablation, and denoising co-training hints that the sign of observability can be moved by explicit training.

Checkpoint: Phase 1 established structural legibility. Phases 2-3 found that passive per-example observability mostly fails under proper controls. Phase 4 tests whether observability must be trained explicitly.

| Phase | Question | Result | Takeaway |
|---|---|---|---|
| **Phase 1** | Does training objective change representation structure? | **Yes** | FF induces sparser, lower-rank, more concentrated representations than BP, independent of overlay and normalization confounders. |
| **Phase 2a** | Does FF goodness faithfully read BP activations? | **No** | `sum(h²)` collapses into a confidence proxy after controlling for logit margin and activation norm. |
| **Phase 2b** | Can co-training rescue the observer? | **Weakly** | Denoising produced a small positive partial correlation, but with much weaker raw predictive utility than confidence. |
| **Phase 3** | Do alternative structural observers recover independent signal? | **No** | After controlling for confidence proxies, all passive structural observers collapse to near-zero partial correlation. |
| **Phase 4** | Must observability be explicitly trained? | Planned | Motivated by Phase 2b's small denoising foothold and Phase 3's negative result. |

**Key findings:**

- FF changes representation structure in real, confounder-controlled ways.
- Passive observers on vanilla BP activations do not recover reliable per-example signal beyond confidence.
- FF-based signals fail as per-example monitors but still identify causally important neurons under ablation. Per-example faithfulness and neuron-level causal salience are distinct properties.
- Denoising co-training produces a small positive foothold, suggesting explicit observer shaping may move the sign.

**Bottom line:** Structural legibility is real. Per-example observability mostly fails under passive readout. Neuron-level causal targeting still works. Explicit observability training is the next test.

## Why this matters

The results so far split "observability" into two problems that behave differently.

- **Per-example monitoring** (does the observer flag likely errors on individual inputs?) mostly fails under passive readout. After controlling for confidence, no structural observer on vanilla BP activations adds meaningful signal. If per-example monitoring is possible at all, it likely requires explicit training rather than passive extraction.
- **Neuron-level causal targeting** (does the observer identify neurons whose removal disproportionately harms performance?) still works. FF-derived signals and magnitude rankings both pick out causally important neurons, even though they fail as per-example monitors. This axis of observability survives the controls that kill the first.

The practical implication: if per-example observability requires a second training objective rather than passive readout, the problem shifts from observer design to architecture design. The question becomes whether a BP main path can coexist with an explicitly trained observer path without degrading capability.

### The faithfulness bar

Any observability system must pass three tests:

- **Correlation.** Does the observer signal track decision-relevant metrics (per-example loss, logit margin) beyond what cheap baselines already capture?
- **Intervention.** When neurons are ablated, does observer-guided targeting degrade performance faster than random, in a way that diverges from simple magnitude ranking?
- **Prediction.** Can the observer rank likely failures better than max softmax, entropy, or a linear probe on the same activations?

## Phase 1: structural comparison (complete)

### MNIST (4x500 MLP, 50 epochs, 3 seeds)

|                                 | Local (FF) |    Global (BP) |    BP+norm | BP+overlay |
| ------------------------------- | ---------: | -------------: | ---------: | ---------: |
| Test accuracy                   |     94.57% |     **98.32%** |     98.29% |     95.09% |
| Probe accuracy (label-masked)   |     99.55% |         97.65% |     97.20% | **99.85%** |
| Pruning@90% (live neurons only) |     99.20% |         97.75% |     97.82% | **99.91%** |
| Polysemanticity (classes/neuron) | 1.78 | **1.63** | 2.55 | 4.05 |
| Dead neuron fraction            |      23.9% |       **6.9%** |      13.1% |       9.1% |
| Effective rank (repr. dimensions) |       44.7 |      **164.2** |      145.3 |      112.0 |
| Sparsity                        |  **87.6%** |          81.0% |      75.3% |      55.4% |

BP+norm: same architecture with per-layer L2 normalization matching FF. Normalization is not the confounder; BP+norm performs identically to BP.

BP+overlay: same architecture trained on label-overlaid input (same scheme as FF). **Label overlay is the dominant confounder.** BP+overlay matches or exceeds FF on probe accuracy and pruning robustness. The probe advantage originally attributed to FF was from the input conditioning scheme, not from local learning.

What FF genuinely produces, independent of label overlay: higher activation sparsity, lower effective rank, and more concentrated information in fewer neurons. These are real structural effects of local learning.

### CIFAR-10 (4x500 MLP, 50 epochs, 3 seeds)

|                                 | Local (FF) |    Global (BP) |    BP+norm | BP+overlay |
| ------------------------------- | ---------: | -------------: | ---------: | ---------: |
| Test accuracy                   |     47.49% |     **54.15%** |     53.84% |     28.92% |
| Probe accuracy (label-masked)   | **86.83%** |         49.74% |     53.79% |     99.91% |
| Sparsity                        |  **86.0%** |         76.8% |      71.7% |      75.9% |
| Dead neuron fraction            |      20.5% |       **4.3%** |      11.1% |      20.7% |
| Effective rank                  |      140.0 |      **336.7** |      283.5 |      121.6 |

CIFAR-10 amplifies the structural gaps. FF probe accuracy (86.8%) far exceeds BP (49.7%). BP+overlay collapses to 28.9% task accuracy, confirming label overlay is a severe confounder on harder tasks.

Probes are trained on training-set activations and evaluated on test-set activations (no test-set contamination). Label-masked probing zeros the first n_cls dimensions. Pruning curves use live neurons only. Full analysis in `analyze.ipynb`.

### Scaling study (MNIST, 5 sizes, 3 seeds each)

Do these structural differences hold as models grow? Five configurations from 200K to 8M parameters.

| Size | Params | Acc (FF-BP) | Dead frac (FF-BP) | Eff rank (FF-BP) |
|---|---|---|---|---|
| XS (2x256) | 0.3M | -3.4% | +4.0% | -54 |
| S (4x500) | 1.1M | -3.7% | +17.0% | -120 |
| M (4x1000) | 3.8M | -3.0% | +19.8% | -183 |
| L (6x1000) | 5.8M | -4.1% | +13.0% | -115 |
| XL (8x1000) | 7.8M | -5.4% | +10.4% | -70 |

FF consistently trades accuracy for more concentrated representations. Two patterns persist across all five sizes: higher dead neuron fraction and lower effective rank. The accuracy gap is stable at 3-5%, widening slightly at XL. Sparsity, the most visually striking difference at small scale (FF 89% vs BP 40% at XS), converges as BP models grow deeper and is negligible by M.

Full scaling data in `results/scaling.json` and `assets/scaling.png`.

## Phase 2: observer faithfulness (complete)

### Phase 2a: passive observer test (negative)

FF goodness on vanilla BP activations against baselines (4x500 MLP, MNIST, 50 epochs, 3 seeds):

| Observer          | Spearman vs loss | AUC (error detection) | Within-class rho |
| ----------------- | ---------------: | --------------------: | ---------------: |
| ff_goodness       |           -0.725 |                 0.923 |           +0.887 |
| max_softmax       |           -0.998 |                 0.959 |           +0.801 |
| logit_margin      |           -0.811 |                 0.965 |           +1.000 |
| entropy           |           +0.811 |                 0.964 |           -0.999 |
| activation_norm   |           -0.706 |                 0.917 |           +0.865 |
| probe_confidence  |           -0.629 |                 0.970 |           +0.738 |

Partial correlation of ff_goodness with loss, controlling for logit margin and activation norm: **-0.056** (+/- 0.039 across seeds). The effect is small and inconsistent in sign across seeds. The observer is not tracking decision structure beyond what confidence already captures. `sum(h²)` collapses into activation energy, which is a confidence proxy. Alternative structural observers are tested in Phase 3.

### Phase 2b: co-training search

Two co-training formulations tested, both using `sum(h²)` as the observer:

- **Overlay auxiliary** (BP + FF contrastive loss with label overlay). ff_goodness partial correlation: +0.015, inconsistent across seeds. The overlay creates a train/eval domain mismatch that makes the result uninterpretable.

- **Denoising auxiliary** (BP + FF contrastive loss with noise corruption, no overlay). ff_goodness partial correlation: **+0.070** (p < 0.001). AUC dropped from 0.923 to 0.688: denoising decoupled goodness from confidence without replacing the lost predictive utility.

Denoising co-training produced the only positive significant partial correlation in the project (+0.070). The cost: raw error-detection AUC dropped from 0.923 to 0.688, while max softmax on the same model maintained 0.947. The denoising objective successfully decoupled goodness from confidence but did not replace the lost information. This is the foothold for Phase 4: explicit shaping moved the partial correlation from negative to positive, suggesting that observability may be trainable even though it is not passively readable.

### Intervention

Even when FF-based signals fail as independent per-example quality estimates, they still identify neurons whose removal disproportionately harms performance. This reveals a second axis of observability: per-neuron causal salience is distinct from per-example faithfulness.

At 70% ablation of the last layer (3 seeds, 50 epochs):

| Strategy | Accuracy |
|---|---|
| FF-targeted | 0.845 |
| Magnitude | 0.836 |
| Class-disc | 0.839 |
| Sparsity-guided | 0.893 |
| Random | 0.983 |
| Anti-targeted | 0.983 |

FF-targeted and magnitude-guided ablation are far more destructive than random, confirming these signals identify causally important neurons. Sparsity-guided ablation is less destructive: the most frequently active neurons are background, not decision-makers. The causally important neurons are concentrated among the high-magnitude, selectively firing ones.

## Phase 3: alternative structural observers (negative)

Phase 2 showed FF goodness fails as a passive observer. The natural follow-up: maybe `sum(h²)` is the wrong readout, and simpler structural metrics on the same activations would work. Three alternatives were tested on vanilla BP activations with no retraining or co-training.

| Observer | What it reads | Partial corr | AUC |
|---|---|---|---|
| ff_goodness | Activation energy | -0.056 | 0.923 |
| active_ratio | Per-example neuron sparsity | -0.035 | 0.502 |
| act_entropy | Activation concentration per layer | -0.039 | 0.429 |
| class_similarity | Cosine similarity to class prototype | -0.141 | 0.951 |

All partial correlations are near zero or negative after controlling for logit margin and activation norm. No passive structural observer on vanilla BP activations recovers meaningful independent signal. The raw Spearman correlations are strong, but the independent component vanishes under proper controls. Structural legibility (Phase 1) is real, but per-example observability beyond confidence has not been achieved through passive readout.

## Phase 4: from passive readout to trained observability (planned)

Passive observers on standard BP activations do not recover reliable independent per-example signal beyond confidence. The next step is to test whether observability must be explicitly trained as a separate channel rather than passively extracted from existing representations.

The denoising result provides a small foothold: explicit shaping of the goodness signal during training produced the only positive partial correlation. Phase 4 scales this idea from a single auxiliary loss term to a dedicated observer architecture.

## Limitations

- Small scale (MLPs on MNIST/CIFAR-10). Results may not transfer to transformers or billion-parameter models.
- No SAE comparison. The most important missing baseline.
- No circuit discovery or feature visualization. Statistical proxies only.
- Hyperparameters not swept (FF lr=0.03, BP lr=0.001, auxiliary weight=0.1 based on convention).
- Intervention tests identify causally important neurons but do not provide independent per-example monitoring.

## What this is not

This is not a claim that FF is better than BP, or that FF should replace BP, or that FF is the right observability objective. FF is one instance of a local, layer-wise training signal. The hypothesis is broader: there may exist a class of objectives whose purpose is to produce legible internal representations, and those objectives can run alongside the capability objective without constraining it. The remaining question is whether an explicitly trained observer path can succeed where passive readout did not.

## How to run

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just). Runs on CPU, MPS, or CUDA.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install just  # or: cargo install just

just test          # run tests
just smoke         # pipeline smoke test (~1 min)
just reproduce     # full reproduction (~60 min)
```

Individual experiments:

```bash
just train                  # Phase 1: MNIST, 3 seeds, 50 epochs
just cifar10                # Phase 1: CIFAR-10, 3 seeds, 50 epochs
just scale                  # Phase 1: scaling study, 5 model sizes
just observe                # Phase 2: observer faithfulness test
just observe-aux            # Phase 2b: auxiliary co-training variant
just observe-denoise        # Phase 2b: denoising co-training variant
```

Results go to `results/`. Phase 1 charts are generated by `analyze.ipynb`. Phase 2 generates intervention dose-response plots in `assets/`.

## Repo structure

- `src/train.py` Phase 1: trains FF, BP, and ablation variants, computes confounder-controlled metrics
- `src/scale.py` Phase 1: scaling study across 5 model sizes
- `src/observe.py` Phase 2: observer faithfulness testing (pure observer, auxiliary, denoise modes; intervention curves)
- `analyze.ipynb` generates Phase 1 figures and analysis from result JSON files
- `results/` result data (JSON, committed)
- `assets/` generated charts (committed for README)

## References

| Paper                                                                                 | Relevance                                                                       |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [The Forward-Forward Algorithm (Hinton, 2022)](https://arxiv.org/abs/2212.13345)      | Local, layer-wise training without backpropagation                              |
| [Fractured Entangled Representations (2025)](https://arxiv.org/abs/2505.11581)        | BP+SGD produces entangled representations; alternative optimization does not    |
| [Limits of AI Explainability (2025)](https://arxiv.org/abs/2504.20676)                | Proves global interpretability is impossible; local explanations can be simpler |
| [Infomorphic Networks (PNAS, 2025)](https://www.pnas.org/doi/10.1073/pnas.2408125122) | Local learning rules produce inherently interpretable representations           |
| [Scalable FF (ICML 2025)](https://arxiv.org/abs/2501.03176)                           | Block-local hybrids outperform pure BP                                          |
| [Contrastive FF for ViT (2025)](https://arxiv.org/abs/2502.00571)                     | FF applied to transformers, small performance gap vs BP                         |
| [Deep-CBN Hybrid (2025)](https://www.nature.com/articles/s41598-025-92218-y)          | FF+BP hybrid exceeds prior baselines in molecular prediction                    |
| [Inference-Time Intervention (2023)](https://arxiv.org/abs/2306.03341)                | Precedent for inference-time activation monitoring and steering                 |

## License

[MIT License](LICENSE)
