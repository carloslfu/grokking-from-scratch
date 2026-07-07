# Grokking from scratch — `(a + b) mod 113`

Reproduction of **grokking** (delayed generalization) on modular addition,
using a 1-layer transformer written in raw PyTorch tensors — **no
`nn.Module`, no `nn.Linear`, no `nn.Embedding`**. Every weight is a plain
tensor and the forward pass is explicit einsum/matmul/softmax, so the whole
model is readable top-to-bottom.

Config follows Nanda et al. 2023 ("Progress Measures for Grokking via
Mechanistic Interpretability"); the phenomenon is from Power et al. 2022
("Grokking: Generalization Beyond Overfitting on Small Algorithmic
Datasets", [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)).

## Result

Trained on 30% of all 113² = 12,769 equations; evaluated on the held-out 70%.

| Milestone | Step |
|---|---|
| Train accuracy > 99% (memorized) | **200** |
| Val loss peak (20.7 — maximally overfit) | 1,200 |
| Val accuracy > 50% | 3,300 |
| Val accuracy > 99% (**grokked**) | **4,200** |

The model holds 100% train / ~5% val accuracy for thousands of steps, then
generalizes — a **21× gap** between memorization and generalization:

![learning curves](00_learning_curves.png)

## The learned algorithm is a Fourier circuit

Exactly as in Nanda et al., the grokked network computes modular addition
via a discrete Fourier transform:

- **Embedding spectrum** collapses onto ~5 key frequencies — k = 33, 53,
  34, 49 hold **87.6%** of all non-DC power (k = 22 a weaker fifth). Before
  grokking the spectrum is flat (~2-3% per frequency).

  ![embedding spectrum](01_embedding_spectrum.png)

- **Numbers embed on circles**: projecting each number's embedding onto the
  (cos, sin) directions at each key frequency traces a clean ring.

  ![embedding circles](02_embedding_circles.png)

- **MLP neurons are band-limited** to the same frequencies (2D-FFT
  concentration 0.20 vs 0.08 for an ungrokked model).

  ![top neurons](03_top_neurons.png)

- **Trajectory**: weight norm rises → peaks (~step 300) → decays
  (memorize → clean up, the Omnigrok signature). In the spectrum heatmap the
  memorization "wash" dies off while the key-frequency columns survive.

  ![trajectory](04_trajectory.png)

## Files

| File | What |
|---|---|
| `grok_from_scratch.py` | Model + training. Raw-tensor transformer, full-batch AdamW, saves 18 checkpoints. |
| `analyze.py` | All analyses/plots: learning curves, embedding FFT + circles, neuron grids, attention, trajectory. |
| `training_log.json` | Metrics every 100 steps. |
| `00–04_*.png` | The plots above. |

Not tracked: `checkpoints/` (~16 MB), `params.pt`, `train.log` (regenerable),
and the paper PDF (get it from [arXiv](https://arxiv.org/abs/2201.02177)).

## Reproduce

```bash
python3 grok_from_scratch.py       # ~8 min on Apple Silicon (MPS), M3 Pro
python3 analyze.py --trajectory    # writes the 5 PNGs + prints report
```

## Setup

1-layer decoder-only transformer, d_model 128, 4 heads (d_head 32),
d_mlp 512, no LayerNorm, no biases — **226,176 params**. Vocab 114
(numbers 0–112 + `=`); sequence `[a, b, =]`, loss on the last position.
Full-batch AdamW (batch = all 3,830 train equations), lr 1e-3, weight decay
1.0, betas (0.9, 0.98), 40k steps, seed 0.

Deviation from Nanda: weights init at `1/√fan_out`, which makes `W_in` ~2×
smaller than his `1/√d_model` convention — smaller init is a known grokking
accelerator, so this run groks at ~4k steps instead of his ~10–15k. To get
the longer plateau, init the MLP matrices at `1/√fan_in` instead.

## Ideas to go deeper

- Implement Nanda's **restricted / excluded loss** progress measures with
  `forward_cache` — shows the Fourier circuit forming gradually from ~step
  1k, long before val accuracy moves.
- Sweep weight decay (0, 0.1, 1, 3) and init scale — both shift the
  grokking point dramatically.
- Swap the task: subtraction, multiplication, `x² + y²` — all mod 113.
