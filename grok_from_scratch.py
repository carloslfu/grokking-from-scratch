"""
Grokking on (x + y) mod 113 — Nanda config, transformer built from scratch.

No nn.Module, no nn.Linear, no nn.Embedding. Every weight is a raw
torch.Tensor with requires_grad=True, and the forward pass is plain
einsum / matmul / softmax. Read top-to-bottom and you see exactly what
a 1-layer decoder-only transformer is doing.

Reproduces Nanda et al. 2023, "Progress Measures for Grokking via
Mechanistic Interpretability":

  - 1-layer transformer
  - d_model = 128, n_heads = 4, d_head = 32, d_mlp = 512
  - vocab = 114  (numbers 0..112 + an '=' token at id 113)
  - sequence length 3:  [a, b, '=']  → predict (a + b) mod 113 at the last pos
  - full-batch AdamW, lr=1e-3, wd=1.0, betas=(0.9, 0.98)
  - 40k steps, 30% training fraction, no LayerNorm

Expected dynamics (this init groks earlier than Nanda's — see README):
  step  ~200 :  train acc → 100%, val acc still ~1/113 ≈ 0.9% (chance)
  step  ~3k  :  val acc starts climbing (circuit formation visible)
  step  ~4k  :  val acc → 100% (grokked)

Run:
  python3 grok_from_scratch.py
"""

import json
import math
import os
import time

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
P            = 113
D_MODEL      = 128
N_HEADS      = 4
D_HEAD       = D_MODEL // N_HEADS    # 32
D_MLP        = 4 * D_MODEL           # 512
N_CTX        = 3                     # [a, b, '=']
VOCAB        = P + 1                 # '=' is token id P (= 113)
EQ_TOKEN     = P

TRAIN_FRAC   = 0.3
LR           = 1e-3
WD           = 1.0
BETAS        = (0.9, 0.98)
N_STEPS      = 40_000
EVAL_EVERY   = 100
LOG_EVERY    = 1_000
SEED         = 0

# Save full param snapshots at these steps so analyze.py can show the
# trajectory of features (when do Fourier components emerge? when does the
# weight norm peak and shrink?). Log-spaced — denser early and around the grok.
CHECKPOINT_STEPS = [
    0, 100, 300, 1_000, 3_000, 5_000, 7_500,
    10_000, 12_500, 15_000, 17_500, 20_000,
    22_500, 25_000, 27_500, 30_000, 35_000, 40_000,
]
CHECKPOINT_DIR = "checkpoints"

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


# -----------------------------------------------------------------------------
# Data
#
# Every (a, b) pair in [0, P)^2 is one example, labeled (a+b) mod P.
# Each example becomes a 3-token sequence [a, b, '='].
# Train/val split is a deterministic shuffle by SEED.
# -----------------------------------------------------------------------------
def make_data(seed):
    a = torch.arange(P).repeat_interleave(P)               # (P*P,)
    b = torch.arange(P).repeat(P)                          # (P*P,)
    c = (a + b) % P                                        # labels
    eq = torch.full_like(a, EQ_TOKEN)
    inputs = torch.stack([a, b, eq], dim=1)                # (P*P, 3)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(P * P, generator=g)
    n_train = int(TRAIN_FRAC * P * P)
    tr, va = perm[:n_train], perm[n_train:]
    return inputs[tr], c[tr], inputs[va], c[va]


# -----------------------------------------------------------------------------
# Parameters
#
# Every weight is a plain tensor with requires_grad=True, initialized
# N(0, 1/fan_out) — std = 1/sqrt(shape[-1]), the output dimension. This is
# a deliberate deviation from Nanda's 1/sqrt(d_model)-everywhere convention,
# and it groks ~2x earlier (step ~4.2k vs ~9.1k; measured — see README and
# variants.py's nanda-init run).
# -----------------------------------------------------------------------------
def param(*shape, std=None):
    if std is None:
        std = 1.0 / math.sqrt(shape[-1])
    t = torch.randn(*shape, device=DEVICE) * std
    t.requires_grad_(True)
    return t


def init_params():
    return {
        # --- Embeddings ---
        "W_E"  : param(VOCAB,   D_MODEL),                  # token embed
        "W_pos": param(N_CTX,   D_MODEL),                  # position embed

        # --- Attention (single layer; head dim packed into shape[0]) ---
        "W_Q"  : param(N_HEADS, D_MODEL, D_HEAD),
        "W_K"  : param(N_HEADS, D_MODEL, D_HEAD),
        "W_V"  : param(N_HEADS, D_MODEL, D_HEAD),
        "W_O"  : param(N_HEADS, D_HEAD,  D_MODEL),

        # --- MLP ---
        "W_in" : param(D_MODEL, D_MLP),
        "W_out": param(D_MLP,   D_MODEL),

        # --- Unembedding ---
        "W_U"  : param(D_MODEL, VOCAB),
    }


# -----------------------------------------------------------------------------
# Forward pass
#
# x : (B, T) int64 token ids → logits (B, T, VOCAB)
#
# Residual stream notation: `resid` is what flows down the network. Each
# block reads `resid`, computes a delta, and writes `resid = resid + delta`.
# -----------------------------------------------------------------------------
def forward(p, x):
    B, T = x.shape

    # --- Embed: token + learned position. (B, T, D_MODEL) ---
    resid = p["W_E"][x] + p["W_pos"][:T]

    # --- Attention ---
    # Per-head Q/K/V projections. Each is (B, T, H, D_HEAD).
    # einsum reads:  for every (batch, time), project D_MODEL → D_HEAD per head.
    q = torch.einsum("btd,hdk->bthk", resid, p["W_Q"])
    k = torch.einsum("btd,hdk->bthk", resid, p["W_K"])
    v = torch.einsum("btd,hdk->bthk", resid, p["W_V"])

    # Scaled dot-product scores. (B, H, T_q, T_k)
    scores = torch.einsum("bthk,bshk->bhts", q, k) / math.sqrt(D_HEAD)

    # Causal mask: upper triangle (j > i) is "future" — set to -inf so softmax
    # ignores it. Shape (T, T), broadcasts over (B, H).
    causal = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
    )
    scores = scores.masked_fill(causal, float("-inf"))
    attn = scores.softmax(dim=-1)

    # Weighted values: (B, T, H, D_HEAD)
    z = torch.einsum("bhts,bshk->bthk", attn, v)

    # Output projection back to D_MODEL, summing across heads.
    attn_out = torch.einsum("bthk,hkd->btd", z, p["W_O"])
    resid = resid + attn_out

    # --- MLP (post-attention) ---
    h = F.relu(resid @ p["W_in"])    # (B, T, D_MLP)
    resid = resid + h @ p["W_out"]   # (B, T, D_MODEL)

    # --- Unembed ---
    return resid @ p["W_U"]          # (B, T, VOCAB)


def forward_cache(p, x):
    """Same as forward(), but also returns a dict of intermediate tensors.

    Useful for mechanistic interpretability: inspect attention patterns,
    MLP neuron activations, residual stream after each block, etc.

    Returns:
        logits : (B, T, VOCAB)
        cache  : dict with keys
            "embed"        : resid right after token+pos embedding   (B, T, D)
            "q","k","v"    : per-head Q/K/V                          (B, T, H, D_HEAD)
            "scores"       : pre-softmax attention scores            (B, H, T, T)
            "attn"         : post-softmax attention probabilities    (B, H, T, T)
            "z"            : weighted values                         (B, T, H, D_HEAD)
            "attn_out"     : attention block output                  (B, T, D)
            "resid_mid"    : resid after attention residual add      (B, T, D)
            "mlp_pre"      : MLP hidden post-ReLU (the "neurons")    (B, T, D_MLP)
            "mlp_out"      : MLP block output                        (B, T, D)
            "resid_post"   : resid after MLP residual add (final)    (B, T, D)
    """
    B, T = x.shape
    cache = {}

    resid = p["W_E"][x] + p["W_pos"][:T]
    cache["embed"] = resid

    q = torch.einsum("btd,hdk->bthk", resid, p["W_Q"])
    k = torch.einsum("btd,hdk->bthk", resid, p["W_K"])
    v = torch.einsum("btd,hdk->bthk", resid, p["W_V"])
    cache["q"], cache["k"], cache["v"] = q, k, v

    scores = torch.einsum("bthk,bshk->bhts", q, k) / math.sqrt(D_HEAD)
    causal = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
    )
    scores = scores.masked_fill(causal, float("-inf"))
    cache["scores"] = scores
    attn = scores.softmax(dim=-1)
    cache["attn"] = attn

    z = torch.einsum("bhts,bshk->bthk", attn, v)
    cache["z"] = z
    attn_out = torch.einsum("bthk,hkd->btd", z, p["W_O"])
    cache["attn_out"] = attn_out
    resid = resid + attn_out
    cache["resid_mid"] = resid

    h = F.relu(resid @ p["W_in"])
    cache["mlp_pre"] = h
    mlp_out = h @ p["W_out"]
    cache["mlp_out"] = mlp_out
    resid = resid + mlp_out
    cache["resid_post"] = resid

    logits = resid @ p["W_U"]
    return logits, cache


def loss_and_acc(p, x, y):
    logits = forward(p, x)[:, -1]                          # (B, VOCAB) at '='
    loss = F.cross_entropy(logits, y)
    acc  = (logits.argmax(-1) == y).float().mean()
    return loss, acc


def save_checkpoint(p, step, dirpath=CHECKPOINT_DIR):
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, f"params_{step:06d}.pt")
    torch.save({k: v.detach().cpu() for k, v in p.items()}, path)


# -----------------------------------------------------------------------------
# Train loop — full-batch AdamW. No minibatching: we compute the gradient on
# the entire training set every step. Deterministic given the seed.
# -----------------------------------------------------------------------------
def train():
    torch.manual_seed(SEED)

    tx, ty, vx, vy = make_data(SEED)
    tx, ty = tx.to(DEVICE), ty.to(DEVICE)
    vx, vy = vx.to(DEVICE), vy.to(DEVICE)
    print(f"train {tx.shape[0]} | val {vx.shape[0]} | device {DEVICE}")

    p = init_params()
    n = sum(t.numel() for t in p.values())
    print(f"params: {n:,}")

    opt = torch.optim.AdamW(
        list(p.values()), lr=LR, betas=BETAS, weight_decay=WD
    )

    log = []
    t0 = time.time()
    for step in range(N_STEPS + 1):
        train_loss, train_acc = loss_and_acc(p, tx, ty)

        # Eval / log / checkpoint BEFORE the update, so everything recorded
        # at `step` reflects the parameters as they stood at that step.
        if step % EVAL_EVERY == 0 or step == N_STEPS:
            with torch.no_grad():
                val_loss, val_acc = loss_and_acc(p, vx, vy)
            log.append({
                "step":       step,
                "train_loss": train_loss.item(),
                "train_acc":  train_acc.item(),
                "val_loss":   val_loss.item(),
                "val_acc":    val_acc.item(),
            })
            if step % LOG_EVERY == 0 or step == N_STEPS:
                dt = time.time() - t0
                print(
                    f"step {step:6d} | "
                    f"train L={train_loss.item():.4f} A={train_acc.item():.3f} | "
                    f"val L={val_loss.item():.4f} A={val_acc.item():.3f} | "
                    f"{dt:6.1f}s"
                )

        if step in CHECKPOINT_STEPS:
            save_checkpoint(p, step)

        if step == N_STEPS:
            break   # final params already logged + checkpointed; don't update past them

        opt.zero_grad(set_to_none=True)
        train_loss.backward()
        opt.step()

    with open("training_log.json", "w") as f:
        json.dump(log, f)
    torch.save({k: v.detach().cpu() for k, v in p.items()}, "params.pt")
    print("saved training_log.json and params.pt")


if __name__ == "__main__":
    train()
