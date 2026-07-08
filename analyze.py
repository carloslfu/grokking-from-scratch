"""
Inspect a trained grokking model — see the Fourier features.

Run after grok_from_scratch.py has finished (or after at least one
checkpoint exists in checkpoints/). Default: analyzes the final params.

  python3 analyze.py                  # static analysis on params.pt
  python3 analyze.py --trajectory     # + per-checkpoint trajectory plots

Produces .png plots in the current directory and prints numeric summaries.

All FFTs run on CPU: the tensors are tiny (113 x 128) and MPS FFT support
is flaky. Forward passes (neuron grids, attention) still run on DEVICE.
"""

import argparse
import glob
import json
import math
import os
import re

import torch
import matplotlib.pyplot as plt

import grok_from_scratch as g


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def load_params(path, device=None):
    device = device or g.DEVICE
    return {
        k: v.to(device)
        for k, v in torch.load(path, weights_only=True).items()
    }


def list_checkpoints():
    paths = sorted(glob.glob(os.path.join(g.CHECKPOINT_DIR, "params_*.pt")))
    out = []
    for path in paths:
        m = re.search(r"params_(\d+)\.pt$", path)
        if m:
            out.append((int(m.group(1)), path))
    return out


def all_input_grid():
    """All (a, b) pairs in [0,P)^2 as a (P*P, 3) tensor on DEVICE.
    Order: a varies slowly, b varies fast — reshape activations to (P, P)
    with a on axis 0 and b on axis 1.
    """
    a = torch.arange(g.P).repeat_interleave(g.P)
    b = torch.arange(g.P).repeat(g.P)
    eq = torch.full_like(a, g.EQ_TOKEN)
    return torch.stack([a, b, eq], dim=1).to(g.DEVICE)


# -----------------------------------------------------------------------------
# Analysis 0: The grokking curves themselves (from training_log.json)
# -----------------------------------------------------------------------------
def plot_learning_curves(log_path="training_log.json",
                         save_to="00_learning_curves.png"):
    if not os.path.exists(log_path):
        print(f"  (no {log_path} — run training first)")
        return
    with open(log_path) as f:
        log = json.load(f)
    # log-x axis: plot step 0 at x=1
    steps = [max(r["step"], 1) for r in log]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(steps, [r["train_acc"] for r in log], label="train", color="#dc2626")
    axes[0].plot(steps, [r["val_acc"]   for r in log], label="val",   color="#16a34a")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("optimization step")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Grokking: train vs val accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, [r["train_loss"] for r in log], label="train", color="#dc2626")
    axes[1].plot(steps, [r["val_loss"]   for r in log], label="val",   color="#16a34a")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("optimization step")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Loss (log-log)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_to, dpi=130)
    plt.close()
    print(f"  saved {save_to}")

    final = log[-1]
    print(f"  final: train acc {final['train_acc']:.4f} | "
          f"val acc {final['val_acc']:.4f}")


# -----------------------------------------------------------------------------
# Analysis 1: Fourier spectrum of the token embeddings
#
# W_E rows 0..P-1 are the per-number embeddings. If the model learned the
# Fourier algorithm, each embedding dim (column of W_E) is a linear combo of
# cos(2πkx/P), sin(2πkx/P) for a *small* set of frequencies k. Summing
# |FFT|^2 over dims exposes them.
# -----------------------------------------------------------------------------
def embedding_power_spectrum(p):
    E = p["W_E"][:g.P].detach().cpu()         # (P, D_MODEL); drop '=' row
    fft = torch.fft.rfft(E, dim=0)            # (P//2 + 1, D_MODEL) complex
    return (fft.abs() ** 2).sum(dim=1)        # (P//2 + 1,)


def report_key_frequencies(p, top_k=10):
    power = embedding_power_spectrum(p)
    vals, idx = torch.topk(power[1:], top_k)  # exclude DC bin k=0
    idx = idx + 1
    total = power[1:].sum().item()
    print(f"\nW_E power spectrum (top {top_k} non-DC frequencies):")
    print(f"  total non-DC power: {total:.2f}")
    for v, k in zip(vals, idx):
        frac = 100 * v.item() / total
        bar = "#" * int(frac / 2)
        print(f"    k = {k.item():3d}   power = {v.item():10.2f}   "
              f"({frac:5.1f}%)  {bar}")
    return idx.tolist()


def plot_embedding_spectrum(p, save_to="01_embedding_spectrum.png"):
    power = embedding_power_spectrum(p)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(power)), power.numpy(), color="#3b82f6")
    plt.xlabel("frequency k")
    plt.ylabel(r"$\sum_d |\mathrm{FFT}(W_E)[k, d]|^2$")
    plt.title(f"Token-embedding power spectrum (p = {g.P})")
    plt.tight_layout()
    plt.savefig(save_to, dpi=130)
    plt.close()
    print(f"  saved {save_to}")


# -----------------------------------------------------------------------------
# Analysis 2: Embedding circles
#
# For a frequency k the model uses, embeddings contain terms
#   E[x, d] ≈ A_d cos(2πkx/P) + B_d sin(2πkx/P).
# The FFT bin at k gives those coefficient vectors: A = Re(fft), B = -Im(fft).
# Projecting each row E[x] onto the unit vectors A/|A| and B/|B| recovers
# ≈ (|A| cos(2πkx/P), |B| sin(2πkx/P)) — the numbers trace a circle, walked
# once around every P/k increments of x.
# -----------------------------------------------------------------------------
def plot_embedding_circles(p, freqs, save_to="02_embedding_circles.png"):
    E = p["W_E"][:g.P].detach().cpu()
    fftE = torch.fft.rfft(E, dim=0)           # (P//2+1, D_MODEL) complex

    n = len(freqs)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows),
                              squeeze=False)
    for i, k in enumerate(freqs):
        ax = axes[i // cols][i % cols]
        u_cos = fftE[k].real
        u_sin = -fftE[k].imag
        u_cos = u_cos / u_cos.norm().clamp_min(1e-9)
        u_sin = u_sin / u_sin.norm().clamp_min(1e-9)
        xs = E @ u_cos                        # (P,)
        ys = E @ u_sin
        ax.scatter(xs, ys, c=range(g.P), cmap="hsv", s=22)
        ax.set_title(f"k = {k}")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Embeddings projected onto (cos, sin) directions per key "
                 "frequency — grokked = circles", y=1.03)
    plt.tight_layout()
    plt.savefig(save_to, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {save_to}")


# -----------------------------------------------------------------------------
# Analysis 3: MLP neuron activation grids
#
# Evaluate every MLP neuron on every (a, b) pair and reshape to (P, P).
# Neurons implementing the algorithm look like 2D sinusoids in (a + b) —
# diagonal stripes — and their 2D FFT concentrates at (±k, ±k).
# -----------------------------------------------------------------------------
def neuron_activation_grids(p):
    """(D_MLP, P, P) activations at the '=' position, on CPU."""
    x = all_input_grid()
    with torch.no_grad():
        _, cache = g.forward_cache(p, x)
    h = cache["mlp_pre"][:, -1]               # (P*P, D_MLP)
    return h.reshape(g.P, g.P, g.D_MLP).permute(2, 0, 1).cpu()


def rank_neurons_by_concentration(grids):
    """Fraction of each neuron's (non-DC) 2D-FFT power in its single largest
    bin. Near 0.5 = pure sinusoid (power splits between conjugate bins)."""
    p2 = torch.fft.fft2(grids).abs() ** 2
    p2[:, 0, 0] = 0
    total = p2.sum(dim=(1, 2))
    peak = p2.amax(dim=(1, 2))
    return peak / total.clamp_min(1e-9)


def peak_frequency(grid):
    """Signed (f_a, f_b) location of the strongest non-DC 2D-FFT bin."""
    p2 = torch.fft.fft2(grid).abs() ** 2
    p2[0, 0] = 0
    idx = p2.argmax().item()
    ka, kb = divmod(idx, g.P)
    ka = ka if ka <= g.P // 2 else ka - g.P
    kb = kb if kb <= g.P // 2 else kb - g.P
    return ka, kb


def plot_top_neurons(p, save_to="03_top_neurons.png", n_show=6):
    grids = neuron_activation_grids(p)
    conc = rank_neurons_by_concentration(grids)
    top = conc.argsort(descending=True)[:n_show].tolist()

    fig, axes = plt.subplots(2, n_show, figsize=(2.6 * n_show, 5.4))
    for i, neuron in enumerate(top):
        ka, kb = peak_frequency(grids[neuron])
        axes[0, i].imshow(grids[neuron].numpy(), cmap="viridis")
        axes[0, i].set_title(
            f"neuron {neuron}\nconc={conc[neuron]:.2f}  peak=({ka},{kb})",
            fontsize=8,
        )
        axes[0, i].set_xlabel("b"); axes[0, i].set_ylabel("a")
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

        fft_shift = torch.fft.fftshift(
            torch.fft.fft2(grids[neuron]).abs()
        )
        axes[1, i].imshow(fft_shift.log1p().numpy(), cmap="magma")
        axes[1, i].set_title("log |FFT2|", fontsize=8)
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

    fig.suptitle(
        "Top MLP neurons by FFT concentration — grokked = diagonal stripes, "
        "peak at (k, k)", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_to, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {save_to}")
    print(f"  concentration of top-{n_show} neurons: "
          + ", ".join(f"{conc[t]:.2f}" for t in top))


# -----------------------------------------------------------------------------
# Analysis 4: Attention patterns at the '=' position
# -----------------------------------------------------------------------------
def report_attention(p, samples=((3, 7), (40, 80), (100, 12))):
    x = torch.tensor(
        [[a, b, g.EQ_TOKEN] for a, b in samples], device=g.DEVICE
    )
    with torch.no_grad():
        _, cache = g.forward_cache(p, x)
    A = cache["attn"][:, :, -1, :].cpu()      # (B, H, 3)

    print("\nAttention from '=' position (probabilities over [a, b, '=']):")
    header = " | ".join(f"head {h}: a    b    =" for h in range(g.N_HEADS))
    print(f"  {'input':>10s} | {header}")
    for i, (a, b) in enumerate(samples):
        cells = []
        for h in range(g.N_HEADS):
            row = A[i, h].tolist()
            cells.append(f"{row[0]:.2f} {row[1]:.2f} {row[2]:.2f}")
        print(f"  ({a:3d},{b:3d})  | " + " | ".join(cells))


# -----------------------------------------------------------------------------
# Analysis 5: Trajectory — how features evolve across training
# -----------------------------------------------------------------------------
def plot_trajectory(save_to="04_trajectory.png"):
    ckpts = list_checkpoints()
    if not ckpts:
        print("  (no checkpoints — re-run grok_from_scratch.py to capture them)")
        return

    steps, norms, spectra = [], [], []
    for step, path in ckpts:
        p = load_params(path, device="cpu")
        steps.append(step)
        norms.append(sum(t.norm().item() ** 2 for t in p.values()) ** 0.5)
        spectra.append(embedding_power_spectrum(p))
    spectra = torch.stack(spectra)            # (n_ckpt, P//2+1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))

    axes[0].plot(steps, norms, "o-", color="#dc2626")
    axes[0].set_xscale("log")
    axes[0].set_xlim(left=max(min((s for s in steps if s > 0), default=1), 1) / 2)
    axes[0].set_ylabel("total weight norm")
    axes[0].set_title("Weight-norm trajectory (rise → peak → decay = "
                      "memorize → clean up)")
    axes[0].grid(alpha=0.3)

    im = axes[1].imshow(
        spectra.log1p().numpy(),
        aspect="auto", cmap="magma",
        extent=(0, spectra.shape[1] - 1, len(steps) - 0.5, -0.5),
    )
    axes[1].set_yticks(range(len(steps)))
    axes[1].set_yticklabels([str(s) for s in steps], fontsize=8)
    axes[1].set_ylabel("checkpoint step")
    axes[1].set_xlabel("frequency k")
    axes[1].set_title("Embedding power spectrum over training — key "
                      "frequencies emerge as bright columns")
    fig.colorbar(im, ax=axes[1], label="log1p power")

    plt.tight_layout()
    plt.savefig(save_to, dpi=130)
    plt.close()
    print(f"  saved {save_to}")


# -----------------------------------------------------------------------------
# Analysis 6: When is the algorithm chosen?
#
# Take the 5 frequencies that dominate the FINAL model, then walk backwards
# through the checkpoints asking: when did they become special? Two measures
# per checkpoint: their share of total spectrum power (baseline for any 5 of
# 56 frequencies: 5/56 ≈ 8.9%) and their ranks among all 56 frequencies.
# -----------------------------------------------------------------------------
def plot_algorithm_selection(save_to="05_algorithm_selection.png"):
    ckpts = list_checkpoints()
    if not ckpts:
        print("  (no checkpoints — re-run grok_from_scratch.py to capture them)")
        return

    p_final = load_params(ckpts[-1][1], device="cpu")
    power_f = embedding_power_spectrum(p_final)[1:]        # freqs 1..56
    winners = (power_f.topk(5).indices + 1).tolist()
    print(f"  final-model winning frequencies: {winners}")

    steps, shares, ranks = [], [], []
    for step, path in ckpts:
        p = load_params(path, device="cpu")
        power = embedding_power_spectrum(p)[1:]
        share = sum(power[w - 1] for w in winners) / power.sum()
        order = power.argsort(descending=True).tolist()
        rk = [order.index(w - 1) + 1 for w in winners]
        steps.append(step)
        shares.append(share.item())
        ranks.append(rk)
        print(f"    step {step:6d} | share {share.item():.3f} | ranks {rk}")

    xs = [max(s, 1) for s in steps]                        # step 0 → x=1 (log axis)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(xs, shares, "o-", color="#dc2626")
    axes[0].axhline(5 / 56, color="gray", ls="--", lw=1,
                    label="baseline: any 5 of 56 freqs (8.9%)")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("winners' share of spectrum power")
    axes[0].set_title("The final algorithm's 5 frequencies, tracked backwards "
                      "through training")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for i, w in enumerate(winners):
        axes[1].plot(xs, [r[i] for r in ranks], "o-", label=f"k = {w}")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].invert_yaxis()                                 # rank 1 on top
    axes[1].set_ylabel("rank among 56 frequencies")
    axes[1].set_xlabel("optimization step (checkpoints; step 0 shown at x=1)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_to, dpi=130)
    plt.close()
    print(f"  saved {save_to}")


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.pt",
                        help="Path to a checkpoint to analyze.")
    parser.add_argument("--trajectory", action="store_true",
                        help="Also produce multi-checkpoint trajectory plots.")
    args = parser.parse_args()

    print("Learning curves:")
    plot_learning_curves()

    print(f"\nLoading {args.params}")
    p = load_params(args.params)
    print(f"  device {g.DEVICE} | {sum(t.numel() for t in p.values()):,} params")

    key_freqs = report_key_frequencies(p, top_k=10)
    plot_embedding_spectrum(p)
    plot_embedding_circles(p, key_freqs[:5])
    plot_top_neurons(p)
    report_attention(p)

    if args.trajectory:
        print("\nBuilding trajectory plots from checkpoints/ ...")
        plot_trajectory()

        print("\nWhen is the algorithm chosen?")
        plot_algorithm_selection()


if __name__ == "__main__":
    main()
