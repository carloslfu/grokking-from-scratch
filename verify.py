"""
Verify every quantitative claim in README.md against the raw artifacts.

  python3 verify.py

Reads training_log.json (committed), params.pt and checkpoints/ (produced
by grok_from_scratch.py), and — if present — the variant logs produced by
variants.py. Prints PASS/FAIL per claim; exits non-zero on any FAIL.

A note on reproducing: training is deterministic for a given seed on a
given backend and PyTorch build (this run: Apple-silicon MPS, seed 0). On
a different backend the random init draw differs, so the *specific*
winning frequencies (33, 53, 34, 49, 22) and exact milestone steps will
differ — that is the frequency lottery the README describes. The committed
training_log*.json files always verify as-is.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

import grok_from_scratch as g
import analyze as an

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  {name}" + (f"   [{detail}]" if detail else ""))


def close(x, y, tol):
    return abs(x - y) <= tol


def first_step(log, key, thresh):
    return next((r["step"] for r in log if r[key] > thresh), None)


def top5_share(params):
    power = an.embedding_power_spectrum(params)[1:]
    top5 = (power.topk(5).indices + 1).tolist()
    share = sum(power[k - 1] for k in top5) / power.sum()
    return top5, share.item()


# =============================================================================
print("== Training dynamics (training_log.json) ==")
log = json.load(open("training_log.json"))
by_step = {r["step"]: r for r in log}

tx, ty, vx, vy = g.make_data(g.SEED)
check("data split 3,830 train / 8,939 val",
      tx.shape[0] == 3830 and vx.shape[0] == 8939,
      f"{tx.shape[0]}/{vx.shape[0]}")

r0 = by_step[0]
check("step 0 at chance (~0.9%) on both splits",
      r0["train_acc"] < 0.02 and r0["val_acc"] < 0.02,
      f"train {r0['train_acc']:.3f}, val {r0['val_acc']:.3f}")

memo = first_step(log, "train_acc", 0.99)
check("memorized (train > 99%) at step 200", memo == 200, f"step {memo}")

peak = max(log, key=lambda r: r["val_loss"])
check("val loss peaks at 20.7 at step 1,200",
      close(peak["val_loss"], 20.7, 0.1) and peak["step"] == 1200,
      f"{peak['val_loss']:.2f} @ {peak['step']}")

v50 = first_step(log, "val_acc", 0.5)
check("val > 50% at step 3,300", v50 == 3300, f"step {v50}")

grok = first_step(log, "val_acc", 0.99)
check("grokked (val > 99%) at step 4,200", grok == 4200, f"step {grok}")

check("plateau ~20x longer than memorization ('twenty times')",
      18 <= grok / memo <= 24, f"ratio {grok / memo:.1f}")

r2k = by_step[2000]
check("mid-plateau: train perfect, val still < 20%",
      r2k["train_acc"] > 0.999 and r2k["val_acc"] < 0.2,
      f"val {r2k['val_acc']:.3f} @ 2000")

check("final val accuracy ~100%", log[-1]["val_acc"] >= 0.995,
      f"{log[-1]['val_acc']:.4f}")

# =============================================================================
print("\n== Model & Fourier circuit (params.pt) ==")
p = an.load_params("params.pt")

n_params = sum(t.numel() for t in p.values())
check("226,176 parameters", n_params == 226_176, f"{n_params:,}")

expected_keys = {"W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O",
                 "W_in", "W_out", "W_U"}
check("9 weight tensors, no biases / LayerNorm",
      set(p.keys()) == expected_keys)

top5, share5 = top5_share(p)
check("top-5 frequencies are 33, 53, 34, 49, 22 (in power order)",
      top5 == [33, 53, 34, 49, 22], f"{top5}")

power = an.embedding_power_spectrum(p)[1:]
share4 = (sum(power[k - 1] for k in top5[:4]) / power.sum()).item()
check("top-4 frequencies hold 87.6% of embedding power",
      close(share4, 0.876, 0.004), f"{share4:.1%}")
check("top-5 frequencies hold ~95%", close(share5, 0.948, 0.004),
      f"{share5:.1%}")

grids = an.neuron_activation_grids(p)
conc = an.rank_neurons_by_concentration(grids)
key_set = set(top5)
peaks = [an.peak_frequency(grids[i]) for i in range(grids.shape[0])]
in_key = sum(1 for ka, kb in peaks if max(abs(ka), abs(kb)) in key_set)
check("all 512 MLP neurons peak at a key frequency",
      in_key == 512, f"{in_key}/512")

top6 = conc.topk(6).values.mean().item()
check("top neurons' FFT concentration ~0.20", close(top6, 0.20, 0.015),
      f"{top6:.3f}")

x = an.all_input_grid()
with torch.no_grad():
    _, cache = g.forward_cache(p, x)
A = cache["attn"][:, :, -1, :].mean(dim=0).cpu()          # (H, [a, b, =])
transport = A[:, :2].sum(dim=1).mean().item()
check("attention from '=' is transport: >99% of mass on a and b",
      transport > 0.99, f"{transport:.3f}")
check("each head splits ~50/50 between a and b",
      all(abs(A[h, 0] - 0.5) < 0.05 for h in range(g.N_HEADS)),
      " ".join(f"{A[h,0]:.2f}" for h in range(g.N_HEADS)))

# =============================================================================
print("\n== Trajectory (checkpoints/) ==")
ckpts = an.list_checkpoints()
check("18 checkpoints", len(ckpts) == 18, f"{len(ckpts)}")

norms, shares, ranks, steps = {}, {}, {}, []
for step, path in ckpts:
    pc = an.load_params(path, device="cpu")
    steps.append(step)
    norms[step] = sum(t.norm().item() ** 2 for t in pc.values()) ** 0.5
    pw = an.embedding_power_spectrum(pc)[1:]
    shares[step] = (sum(pw[k - 1] for k in top5) / pw.sum()).item()
    order = pw.argsort(descending=True).tolist()
    ranks[step] = [order.index(k - 1) + 1 for k in top5]

check("weight norm starts ~50", close(norms[0], 50.4, 0.5),
      f"{norms[0]:.1f}")
peak_step = max(norms, key=norms.get)
check("weight norm peaks ~62 at step 300",
      peak_step == 300 and close(norms[300], 62.2, 0.5),
      f"{norms[peak_step]:.1f} @ {peak_step}")
check("weight norm decays to ~37 by the end", close(norms[40000], 36.5, 1.0),
      f"{norms[40000]:.1f}")

readme_shares = {0: 0.091, 100: 0.121, 300: 0.145, 1000: 0.190,
                 5000: 0.769, 40000: 0.948}
ok = all(close(shares[s], v, 0.002) for s, v in readme_shares.items())
check("winners' power share matches README table",
      ok, " ".join(f"{s}:{shares[s]:.1%}" for s in readme_shares))

readme_ranks = {0: [22, 8, 16, 41, 31], 100: [1, 2, 4, 3, 7],
                300: [1, 2, 3, 4, 6], 1000: [1, 3, 2, 4, 5],
                5000: [1, 3, 2, 4, 5], 40000: [1, 2, 3, 4, 5]}
ok = all(ranks[s] == v for s, v in readme_ranks.items())
check("winners' ranks match README table", ok, f"init {ranks[0]}")

check("winners invisible at init: all ranked 8th or worse, ~baseline share",
      min(ranks[0]) >= 8 and close(shares[0], 5 / 56, 0.025),
      f"ranks {ranks[0]}, share {shares[0]:.1%} vs baseline {5/56:.1%}")

locked = all(max(ranks[s]) <= 7 for s in steps if s >= 100)
check("winners locked into top 7 from step 100 onward", locked)

acc = by_step[100]
check("at step 100 train acc 80%, val 2.6% (still memorizing)",
      close(acc["train_acc"], 0.803, 0.01) and close(acc["val_acc"], 0.026, 0.005),
      f"{acc['train_acc']:.3f}/{acc['val_acc']:.3f}")

# ungrokked comparison for neuron concentration (step 300: memorized, pre-grok)
p300 = an.load_params("checkpoints/params_000300.pt")
conc300 = an.rank_neurons_by_concentration(
    an.neuron_activation_grids(p300)).topk(6).values.mean().item()
check("just-memorized model's neuron concentration ~0.07",
      close(conc300, 0.074, 0.015), f"{conc300:.3f}")

# gradient/weight cosine: the Prieto et al. logit-scaling alignment
tx_d, ty_d = tx.to(g.DEVICE), ty.to(g.DEVICE)
cos = {}
for step in [300, 1000, 3000]:
    pg = {k: v.to(g.DEVICE).requires_grad_(True)
          for k, v in torch.load(f"checkpoints/params_{step:06d}.pt",
                                 weights_only=True).items()}
    loss, _ = g.loss_and_acc(pg, tx_d, ty_d)
    loss.backward()
    gs = torch.cat([t.grad.flatten() for t in pg.values()])
    ws = torch.cat([t.detach().flatten() for t in pg.values()])
    cos[step] = F.cosine_similarity(gs, ws, dim=0).item()
check("gradient-weight cosine builds to -0.8 through the plateau",
      cos[300] > -0.7 and close(cos[1000], -0.79, 0.04)
      and close(cos[3000], -0.82, 0.04),
      f"300:{cos[300]:+.2f} 1k:{cos[1000]:+.2f} 3k:{cos[3000]:+.2f}")

# =============================================================================
print("\n== Variant runs (variants.py) ==")


def load_variant(tag):
    path = f"training_log_{tag}.json"
    return json.load(open(path)) if os.path.exists(path) else None


wd0 = load_variant("wd0")
if wd0 is None:
    print("  (skipped: no training_log_wd0.json — run `python3 variants.py wd0`)")
else:
    check("wd=0 never groks: val hovers near 10% for 40k steps",
          max(r["val_acc"] for r in wd0) < 0.15
          and close(wd0[-1]["val_acc"], 0.10, 0.03),
          f"max {max(r['val_acc'] for r in wd0):.3f}, final {wd0[-1]['val_acc']:.3f}")
    half = next((r["step"] for r in wd0 if r["frac_prob_one"] > 0.5), None)
    check("wd=0: half the train softmaxes round to exactly 1.0 by ~2k",
          half is not None and 1_500 <= half <= 3_000, f"step {half}")
    sat = next((r for r in wd0 if r["frac_prob_one"] == 1.0), None)
    check("wd=0 Softmax Collapse: all 3,830 samples saturated at ~29k",
          sat is not None and 27_000 <= sat["step"] <= 31_000,
          f"step {sat['step'] if sat else None}")
    check("wd=0: saturated samples' losses are exactly 0.0 (gradient starves)",
          sat is not None and sat["frac_loss_zero"] == 1.0)
    check("wd=0: weight norm climbs unchecked, 50 -> ~96",
          close(wd0[0]["weight_norm"], 50.4, 0.5)
          and close(wd0[-1]["weight_norm"], 96, 2),
          f"{wd0[0]['weight_norm']:.0f} -> {wd0[-1]['weight_norm']:.0f}")

og = load_variant("orthograd")
if og is None:
    print("  (skipped: no training_log_orthograd.json — run `python3 variants.py orthograd`)")
else:
    og_grok = first_step(og, "val_acc", 0.99)
    check("⊥Grad + wd=0 groks by ~step 1,100 (vs 4,200 under decay)",
          og_grok is not None and close(og_grok, 1100, 200),
          f"step {og_grok}")
    check("⊥Grad weight norm never shrinks: 50 -> ~83",
          close(og[0]["weight_norm"], 50.4, 0.5)
          and close(og[-1]["weight_norm"], 83, 2)
          and min(r["weight_norm"] for r in og) >= og[0]["weight_norm"] - 0.5,
          f"{og[0]['weight_norm']:.0f} -> {og[-1]['weight_norm']:.0f}")
    after = [r["val_acc"] for r in og if r["step"] >= og_grok]
    check("⊥Grad endgame noisy but holds: val stays high after grokking",
          min(after) > 0.8 and og[-1]["val_acc"] > 0.98,
          f"range {min(after):.2f}-{max(after):.2f}, final {og[-1]['val_acc']:.3f}")
    if os.path.exists("params_orthograd.pt"):
        og_top5, og_share = top5_share(an.load_params("params_orthograd.pt"))
        check("⊥Grad still lands on a Fourier circuit: top-5 share ~92%",
              close(og_share, 0.925, 0.02), f"{og_share:.1%} in {og_top5}")

ni = load_variant("nanda_init")
if ni is None:
    print("  (skipped: no training_log_nanda_init.json — run `python3 variants.py nanda-init`)")
else:
    ni_memo = first_step(ni, "train_acc", 0.99)
    ni_grok = first_step(ni, "val_acc", 0.99)
    check("Nanda 1/sqrt(d_model) init: memorizes at 200, groks at ~9,100",
          ni_memo == 200 and ni_grok is not None and close(ni_grok, 9100, 300),
          f"memorized {ni_memo}, grokked {ni_grok}")
    ni_4200 = next(r for r in ni if r["step"] == 4200)
    check("Nanda init still at ~8% val when fan_out init has fully grokked",
          ni_4200["val_acc"] < 0.15, f"val {ni_4200['val_acc']:.3f} @ 4200")
    check("Nanda init starts at smaller total norm yet groks later (50 vs 42)",
          close(ni[0]["weight_norm"], 42.0, 0.5),
          f"init norm {ni[0]['weight_norm']:.1f}")

# =============================================================================
print("\n== Files ==")
pngs = [f"{i:02d}_{n}.png" for i, n in enumerate(
    ["learning_curves", "embedding_spectrum", "embedding_circles",
     "top_neurons", "trajectory", "algorithm_selection"])]
missing = [f for f in pngs if not os.path.exists(f)]
check("all 6 README plots present", not missing, ", ".join(missing) or "ok")

# =============================================================================
n_fail = sum(1 for _, ok in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
sys.exit(1 if n_fail else 0)
