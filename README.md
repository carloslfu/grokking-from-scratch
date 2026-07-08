# Grokking, from scratch

Train a tiny transformer on modular arithmetic and something strange happens:
it **memorizes** the training data in 200 steps, then for thousands more
keeps failing the equations it hasn't seen — until, abruptly, it *gets it*,
snapping to ~100% accuracy on equations it was never trained on. OpenAI researchers named this delayed generalization
**grokking** ([Power et al. 2022](https://arxiv.org/abs/2201.02177)).

This repo reproduces the phenomenon end-to-end on a laptop (~8 minutes),
then opens the network up and finds the algorithm it discovered: the model
invents a **Fourier transform** — it places the numbers on circles and adds
them by rotation.

The transformer is written from scratch in raw PyTorch tensors — no
`nn.Module`, no `nn.Linear`, no `nn.Embedding`. Every weight is a plain
tensor and the forward pass is explicit einsum/matmul/softmax, readable
top-to-bottom in [`grok_from_scratch.py`](grok_from_scratch.py). The
architecture and training config follow
[Nanda et al. 2023](https://arxiv.org/abs/2301.05217), the paper that first
reverse-engineered what grokked networks actually learn.

## What is grokking?

The standard mental model of overfitting says: once training accuracy hits
100% and validation accuracy is stuck near zero, you're done — the model has
memorized, and more training won't help. Grokking breaks that intuition.
On small algorithmic datasets, if you keep training *far* past the point of
overfitting (with regularization, especially weight decay), validation
accuracy eventually snaps from chance to perfect. The network abandons its
memorized lookup table in favor of the actual rule — long after train
accuracy hit 100% and it seemingly had nothing left to learn.

## Why does it happen? — the short answer

Two solutions compete inside the network. **Memorization** is easy to find,
so gradient descent finds it first — but it's expensive: storing 3,830
arbitrary facts takes many weights, each reinforced by only a few examples.
**The real rule** is hard to find but cheap to run: a few reused weights
handle every equation. Weight decay taxes all weights all the time, and
that tax decides the race — the memorized weights can't pay it, the reused
ones can. So while train accuracy sits at a flawless 100%, the general
circuit quietly grows and the memorized one decays away.

And the suddenness is an illusion. Open the model up (as this repo does)
and the transition is **gradual inside**: the rule-circuit forms steadily
through the plateau; validation accuracy only snaps upward at the moment
that circuit finally outweighs the memorization it was hiding behind. We
know this because the circuit is directly visible in the weights while
val accuracy still sits in single digits — see
[How grokking happens](#how-grokking-happens) below for the plots.

## The task

Learn `(a + b) mod 113` from examples. There are 113² = 12,769 possible
equations; the model trains on a random 30% (3,830) and is evaluated on the
held-out 70% (8,939) it never sees.

Each equation is three tokens. `37 + 1 = 38` becomes:

```
input   [ 37,  1, 113 ]     ← token IDs for  "37"  "1"  "="
label     38                ← the token the model must predict at '='
```

The vocabulary is 114 symbols: the numbers 0–112 (113 residues — one per
possible value mod 113) plus `=` as token 113. There is no `+` token; every
sequence has the same shape, so the operation is implicit.

One thing to internalize: **the numbers are opaque symbols, not numerals.**
Token 37 is just ID #37 — the model never sees digits, never knows 38 comes
after 37, and can't compute anything *from* the ID. It's handed 30% of the
cells of a 113×113 answer table, like a giant Sudoku, and has to fill in
the rest. The only way to do that better than chance is to discover the
table's hidden structure.

## The model

A minimal GPT: one attention block and one MLP on a residual
stream. 226,176 parameters. No LayerNorm, no biases — stripped to the
studs, which is exactly what makes it possible to read the learned
algorithm out of the weights later.

```
     input:  [ a ,  b ,  = ]
                │
                ▼
   EMBED        resid = W_E[token] + W_pos          W_E: 114×128 lookup table
                │                                   (one learned vector per symbol)
                ▼
   ATTENTION    4 heads, causal mask                the '=' position pulls in
                resid = resid + attn(resid)         the vectors for a and b
                │
                ▼
   MLP          512 ReLU neurons                    combines the a- and b-features
                resid = resid + mlp(resid)          (this is where the math happens)
                │
                ▼
   UNEMBED      logits = resid @ W_U                scores all 114 symbols;
                │                                   highest logit at '=' wins
                ▼
     prediction:  (a + b) mod 113
```

Division of labor in the trained network: the **embedding** stores each
number's representation, **attention** is almost pure transport — measured
on the trained model, 99.9% of the `=` position's attention mass lands on
`a` and `b`, split 50/50 by every head — the **MLP** does the actual
computation, and the **unembedding** reads out the answer.

(A note on the embedding: `W_E[token]` is mathematically one-hot encoding
times a matrix — indexing just skips multiplying all the zeros. Same at the
loss: cross-entropy against an integer label is cross-entropy against a
one-hot target.)

The causal mask is the standard GPT rule — each position may attend only to
itself and earlier positions, so `a` sees nothing, `b` sees `a`, and `=`
sees everything. Only the `=` position's output is used, so the mask isn't
load-bearing here; we keep it to stay faithful to the papers' architecture.

## What happens during training

Full-batch AdamW — the gradient of the entire training set every step —
with strong weight decay (1.0), for 40,000 steps.

| Milestone | Step |
|---|---|
| Train accuracy > 99% (**memorized**) | **200** |
| Val loss peaks at 20.7 (maximally overfit) | 1,200 |
| Val accuracy > 50% | 3,300 |
| Val accuracy > 99% (**grokked**) | **4,200** |

![learning curves](00_learning_curves.png)

Read the left plot: train accuracy (red) is perfect from step 200 onward.
Validation accuracy (green) — the 8,939 equations the model has never seen —
starts at chance (1/113 ≈ 0.9%), is still below 10% at step 1,000, and only
crosses 50% at step 3,300 — then leaps to 99% within the next 900 steps.
Memorizing took 200 steps; generalizing took 4,200 — **twenty-one times
longer**. On the right, validation loss first *rises* to a huge peak
(classic overfitting: the model grows more confidently wrong about unseen
data) before its second descent. During that whole plateau, nothing about
the training loss suggests anything is happening — the change is invisible
unless you look inside.

## Inside the grokked model: a Fourier circuit

Why would a neural network represent numbers as waves? Because **modular
arithmetic is rotation.** `(a + b) mod 113` is "walk `a` hours on a 113-hour
clock, then `b` more." A clock face is a circle, and the natural coordinates
for a point on a circle are cosine and sine. So the model learns to embed
each number `x` as an angle:

```
x   →   cos(2πkx/113),  sin(2πkx/113)         for a handful of frequencies k
```

The `/113` bakes the "mod" into the geometry — `x` and `x+113` land on the
same point, wraparound for free. Addition then needs only multiplication,
via the trig identity the MLP learns to implement:

```
cos(w(a+b)) = cos(wa)·cos(wb) − sin(wa)·sin(wb)
```

and the unembedding scores each candidate answer `c` by, in effect,
`cos(w(a+b−c))` — maximal exactly when `c = (a+b) mod 113`, i.e. when the
rotated vector lines up with the answer's vector.

Why several frequencies instead of one? A single cosine peaks at the right
answer but falls off smoothly — near-miss answers also score well. Summing
the score across ~5 different frequencies sharpens it into a spike:
at the correct `c` every frequency agrees (constructive interference); at
every wrong `c` they disagree and cancel (destructive). Same math as a
Fourier series building a sharp peak out of smooth waves.

The evidence, from this actual trained model:

**The embedding spectrum is 5 spikes.** Run an FFT down the embedding table
and four of the 56 possible frequencies — k = 33, 53, 34, 49 — hold
**87.6%** of the power (k = 22 is a weaker fifth). At initialization the
same spectrum is flat; the spikes grow through the plateau (tracked
step-by-step in [When is the algorithm chosen?](#when-is-the-algorithm-chosen)).

![embedding spectrum](01_embedding_spectrum.png)

**The numbers sit on circles.** Project each number's embedding onto the
(cos, sin) directions at each key frequency:

![embedding circles](02_embedding_circles.png)

**All 512 MLP neurons are tuned to the key frequencies.** Evaluate each
neuron on all 113×113 input pairs and its activation grid is a
single-frequency 2-D wave — the plaid patterns below, one wave in `a` and
one in `b` at the same k (true for 512 of 512 neurons). Classify every
neuron by the peak of its 2-D FFT: 147 land on k = 33, 101 on 49, 97 on 53,
95 on 34, 72 on 22 — **the entire layer, with not one neuron doing anything
else.** (Sharpness: the top neurons put 0.20 of their FFT power in a single
bin, vs 0.07 for the same model just after memorization.)

![top neurons](03_top_neurons.png)

And why *those* frequencies? **No reason — it's a lottery.** 113 is prime,
so every frequency is functionally identical: the map `x → kx mod 113` just
relabels the clock positions. A different seed picks different winners.
When the lottery is *drawn* is more subtle than "at initialization" — we
measured it, and the answer is below in
[When is the algorithm chosen?](#when-is-the-algorithm-chosen)

## How grokking happens

The two solutions compete on weight efficiency:

- **Memorization** needs many bespoke weights — each stores facts about
  specific training equations, receives gradient only from those examples,
  and generalizes to nothing.
- **The Fourier circuit** reuses the same few directions for *every*
  equation — constant gradient reinforcement, tiny total weight norm.

Weight decay (here a strong 1.0) taxes every weight every step: with
lr 1e-3, each weight is multiplied by 0.999 per update — a 0.1% tax, 40,000
times — so any weight the gradients don't actively defend decays to nothing
within a few thousand steps. The memorization circuit — diffuse and weakly
reinforced — can't pay the tax; the Fourier circuit can. Training first
finds the fast, greedy memorization solution, then slowly replaces it with
the efficient one:

![trajectory](04_trajectory.png)

Top: total weight norm rises while memorizing, peaks (~step 300), then
decays — the cleanup. Bottom: the embedding spectrum over training. Early
checkpoints are a diffuse wash across all frequencies (memorization
weights); the wash then fades to black while exactly the key-frequency
columns stay bright. You are watching the lottery being run.

Nanda et al.'s deeper finding, visible in this trajectory: **grokking is
only sudden at the output.** Inside, the Fourier circuit forms gradually
from early in the plateau — the val-accuracy cliff is just the moment it
finally outweighs the memorization circuit it's been hiding behind. And as
[Liu et al.'s "Omnigrok"](https://arxiv.org/abs/2210.01117) showed, the
plateau length is largely a weight-norm story: start with smaller weights
(or decay harder) and the wait shrinks.

Where does the gradient actually *go* during the plateau? Mostly nowhere:
[Prieto et al. 2025](https://arxiv.org/abs/2501.04697) showed that once the
training set is memorized, the gradient aligns with the direction that just
scales the logits up — same predictions, lower loss, nothing learned. That
is measurable in this run: the cosine between the full gradient and the
weight vector builds to −0.79 by step 1k and −0.82 by step 3k (recomputed
from the checkpoints in `verify.py`) — by mid-plateau, ~80% of the gradient
is "make the logits bigger," and weight decay is the counterforce that
keeps pulling the scaling back.

Remove the counterforce and the scaling runs into float32 itself: rerun
this code with wd = 0 (`python3 variants.py wd0`) and the weight norm
climbs unchecked (50 → 96) until softmax probabilities on training samples
round to exactly 1.0 — half of them by step ~2k, then in a jittery
sawtooth of saturation and brief escape, all 3,830 at step ~29k. A
saturated sample's loss is exactly 0.0 in float32, so it sends back *no
gradient at all*; learning starves, and val accuracy hovers near 10% for
the whole 40k steps, never grokking — their "Softmax Collapse."

The causal flip side: keep wd = 0 but delete the
scaling component instead (project each weight's gradient orthogonal to
the weight — their ⊥Grad, `python3 variants.py orthograd`), and this same
model generalizes by step ~1,100 with *no regularization at all* — and
still lands on a Fourier circuit: 92% of embedding power in 5 frequencies,
three of them shared with the decay run's winners and two new (the
lottery, drawn differently). Its endgame is noisier — val hovers at
98–100% with occasional instability dips instead of pinning 1.000 — but
the circuit holds. So the rule-circuit isn't inherently slow to build;
much of the wait is the optimizer spending its gradient budget on logit
scaling instead.

Put together, weight decay is doing two separable jobs here. **Selection**:
the per-step tax that memorization's weights can't pay but the reused
Fourier directions can — the race described above. **Anti-scaling**: the
decay update −λ·w points exactly opposite the logit-scaling direction, so
it's also what keeps the loss numerically alive for that race to be run at
all. The ⊥Grad result separates the two cleanly: with no tax whatsoever,
the general circuit still forms — gradient descent builds it on its own.
Decay doesn't force the good representation into existence so much as
block the scaling cheat and clear the dead memorization weights out of
its way.

The two anti-scaling styles also differ mechanically: decay *counteracts*
the scaling after the gradient has already spent its budget on it — the
−0.8 alignment holds from step ~1k all the way to the grok — while ⊥Grad
deletes it up front and re-spends that budget on learning. That's why it
generalizes ~4× sooner than the decay-driven grok (step ~1.1k vs 4.2k),
and why its weight norm never shrinks (it drifts 50 → 83 across training,
vs rise-then-fall 50 → 62 → 37 under decay). Two
scope notes keep this honest: "weight direction = pure logit scaling" is
exact only for homogeneous models — bias-free, LayerNorm-free, i.e.
exactly this one — and none of it dominates ordinary LLM pretraining,
where one epoch over near-endless data means the training set is never
memorized to begin with.

## When is the algorithm chosen?

Take the 5 frequencies that dominate the *final* model and track them
backwards through the checkpoints: what share of spectrum power did they
hold, and how did they rank among all 56 frequencies?

| Step | Winners' power share | Winners' ranks (of 56) | Train acc | Val acc |
|---|---|---|---|---|
| 0 (init) | 9.1% (≈ baseline) | 22, 8, 16, 41, 31 | 0.8% | 0.9% |
| **100** | 12.1% | **1, 2, 4, 3, 7** | 80% | 2.6% |
| 300 | 14.5% | 1, 2, 3, 4, 6 | 100% | 5.5% |
| 1,000 | 19.0% | 1, 3, 2, 4, 5 | 100% | 9.3% |
| 5,000 | 76.9% | 1, 3, 2, 4, 5 | 100% | 100% |
| 40,000 | 94.8% | 1, 2, 3, 4, 5 | 100% | 100% |

![algorithm selection](05_algorithm_selection.png)

Two findings, both sharper than the folklore:

1. **The winners are *not* chosen at initialization.** At step 0 they rank
   8th–41st, holding ≈ baseline power share — statistically invisible.
   The tempting story "gradient descent amplifies the frequencies that
   started ahead at init" is not what happens here.
2. **They are chosen by step 100 — *during* memorization, not after it.**
   While train accuracy is still climbing to 80%, all five eventual
   winners jump into the top 7 and never leave. Memorization and
   algorithm-seeding are simultaneous processes, not sequential phases.
   Everything after step ~300 is amplification (share 15% → 95%) and
   cleanup — the identity of the final algorithm never changes again for
   39,700 steps.

The general lesson: **the visible transition is never the decision — it's
the announcement.** The val-accuracy jump at step 4,200 announces a choice
that was settled around step 100, forty times earlier. And the "choice"
itself is symmetry breaking, nothing mystical: all 56 frequencies are
mathematically equivalent, so the winner had to come from somewhere
arbitrary — microscopic initialization noise, amplified by early training
dynamics into an irreversible commitment, like a pencil balanced on its
tip falling in *some* direction. (Reproduce with
`python3 analyze.py --trajectory`.)

## Takeaways beyond toy models

Grokking is a small, cheap demonstration of a general failure mode:
**sudden-looking jumps are usually smooth processes measured badly.**
Accuracy is a threshold metric — it moves last, at the moment an internal
circuit finally wins. The continuous signals moved much earlier: in this
run, the weight norm peaked at step ~300 and the Fourier frequencies were
visibly strengthening by step ~1–3k, while val accuracy still sat near
chance. If you only watch the output metric, the most interesting part of
training is invisible.

The same illusion shows up at real scale:

- **"Emergent abilities" of LLMs** often look sudden because exact-match
  accuracy is thresholded — the underlying log-likelihoods improve smoothly
  ([Schaeffer et al. 2023](https://arxiv.org/abs/2304.15004)).
- **Induction heads**, the circuit behind in-context learning, form in an
  abrupt window during real LM training, visible as a bump in the loss
  curve ([Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html))
  — reproduced from scratch, same philosophy as here, in the companion repo
  [induction-heads-from-scratch](https://github.com/carloslfu/induction-heads-from-scratch).
- A smooth **aggregate loss can hide sharp per-skill transitions** that
  average out.

Practical habits this motivates when training large models:

- Track **continuous precursors** (log-prob of correct answers, probe
  accuracy), not just pass/fail evals — they make "emergence" forecastable.
- **Slice evals per skill**; don't trust one aggregate number.
- Keep **log-spaced checkpoints** and study trajectories, not endpoints.
- Log the free internals — **per-layer weight norms, gradient norms** —
  phase transitions announce themselves there first.
- If a skill is stuck at chance, the fix is usually **more/better data,
  not more steps**: time-to-generalize explodes as task data shrinks.
- **Small-data finetuning is the grokking regime** (overparameterized
  model, many epochs, weight decay) — there, "train accuracy is 100% and
  val is flat, stop the run" can be premature.

The caveat that keeps this honest: most plateaus are just plateaus. The
lesson is not "always train longer" — it's to instrument training so that
nothing important is invisible.

## Run it yourself

```bash
python3 grok_from_scratch.py       # trains 40k steps, ~8 min on Apple Silicon (M3 Pro, MPS)
python3 analyze.py --trajectory    # writes the 6 PNGs above + prints a numeric report
python3 verify.py                  # re-checks every number in this README against the artifacts
python3 variants.py all            # the wd=0, ⊥Grad, and Nanda-init side runs (~25 min)
```

Watch the stdout during training: train accuracy hits 1.000 within the
first few hundred steps while val crawls — 0.09 at step 1k, 0.15 at 2k,
0.35 at 3k — then jumps: 0.97 by step 4k, 1.00 by 5k. `analyze.py` prints
the key frequencies and per-head attention patterns for the final model.

Every quantitative claim in this README is checked by
[`verify.py`](verify.py) against the raw artifacts — the committed
training logs plus the params/checkpoints your own run regenerates. (This
run: Apple-silicon MPS, PyTorch 2.x, seed 0. On a different backend the
init draw differs, so the *specific* winning frequencies and milestone
steps will differ — that's the lottery; the committed logs and the
structural checks still verify.)

Requirements: Python 3, PyTorch (MPS, CUDA, or CPU), matplotlib.

## Configuration

| | |
|---|---|
| Model | 1-layer decoder-only transformer, d_model 128, 4 heads (d_head 32), d_mlp 512 |
| | no LayerNorm, no biases — 226,176 params |
| Data | `(a+b) mod 113`, 30% train / 70% val, seed 0 |
| Optimizer | full-batch AdamW, lr 1e-3, weight decay 1.0, betas (0.9, 0.98), 40k steps |
| Checkpoints | 18 log-spaced snapshots → `checkpoints/` |

AdamW rather than Adam is load-bearing: AdamW applies weight decay directly
to the weights instead of folding it into the adaptively-scaled gradient,
and the entire phenomenon runs on decay behaving exactly as configured.

One deviation from Nanda's setup, and it matters: weights here init at
`1/√fan_out` instead of his `1/√d_model` everywhere — which makes `W_in`
2× smaller and the attention projections 2× larger. The effect is
measured, not assumed: `python3 variants.py nanda-init` runs the identical
code with his convention, and it groks at step ~9,100 instead of ~4,200
(at step 4,200 it is still at 8% val) — consistent with the ~10k his paper
reports. Notably this is *not* a smaller-init story: our init has the
**larger** total weight norm (50 vs 42) yet groks 2.2× sooner. What
matters is how the scale is distributed across the matrices, not the
overall amount.

Committed: the training logs (`training_log*.json`), so every
learning-curve claim is checkable without retraining. Not tracked:
`checkpoints/` (~16 MB), `params*.pt`, `*.log` — all regenerated by the
commands above — and the Power et al. PDF
([get it from arXiv](https://arxiv.org/abs/2201.02177)).

## Going further

- Implement Nanda's **restricted / excluded loss** progress measures using
  `forward_cache()` (returns every intermediate activation) — this makes
  the gradual circuit formation visible from ~step 1k, long before val
  accuracy moves.
- **Sweep weight decay** (0, 0.1, 1, 3) and init scale — both shift the
  grokking point dramatically. The extremes are already runnable:
  [`variants.py`](variants.py) has wd=0 (starves in Softmax Collapse) and
  ⊥Grad (the plateau all but disappears — and its ~10-line implementation,
  `orthogonalize_grads_`, is a good read).
- **Implement StableMax** — the other fix from Prieto et al. 2025: swap
  softmax's exponential for a function that grows linearly, so
  probabilities can't saturate to exactly 1.0 — and grokking again needs
  no weight decay.
- **Swap the task**: subtraction, multiplication, `x² + y²` — one-line
  change in `make_data`.

## References

- Power, Burda, Edwards, Babuschkin, Misra (2022).
  [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177).
- Nanda, Chan, Lieberum, Smith, Steinhardt (2023).
  [Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217).
- Liu, Michaud, Tegmark (2022).
  [Omnigrok: Grokking Beyond Algorithmic Data](https://arxiv.org/abs/2210.01117).
- Prieto, Barsbey, Mediano, Birdal (2025).
  [Grokking at the Edge of Numerical Stability](https://arxiv.org/abs/2501.04697).
  ICLR 2025.
- Olsson et al. (2022).
  [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
  — the companion phenomenon, reproduced in
  [induction-heads-from-scratch](https://github.com/carloslfu/induction-heads-from-scratch).
- Schaeffer, Miranda, Koyejo (2023).
  [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004)
