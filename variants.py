"""
Variant training runs that ground the README's side-claims.

  python3 variants.py wd0          # weight decay 0 → Softmax Collapse, never groks
  python3 variants.py orthograd    # wd 0 + ⊥Grad (Prieto et al. 2025) → groks ~1.1k
  python3 variants.py nanda-init   # all weights at Nanda's 1/√d_model → groks later
  python3 variants.py all

Each run writes training_log_<name>.json (committed — these back specific
numbers in the README) and params_<name>.pt (not committed). Same seed,
data split, model, and step count as the main run; only the named knob
changes.

Beyond the usual metrics, every log row records the total weight norm and,
on the training set, the fraction of samples whose softmax probability of
the correct answer has saturated to exactly 1.0 (and whose loss is exactly
0.0) — the float32 signature of Softmax Collapse.
"""

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

import grok_from_scratch as g


def nanda_init_params():
    """Nanda's convention: every weight ~ N(0, 1/d_model) instead of our
    1/√fan_out. Same shapes, same seed — only the stds differ."""
    std = 1.0 / math.sqrt(g.D_MODEL)
    return {k: g.param(*v.shape, std=std) for k, v in g.init_params().items()}


def orthogonalize_grads_(p):
    """⊥Grad (Prieto et al. 2025): per weight tensor, remove the gradient's
    component along the weight itself (the logit-scaling direction), then
    rescale back to the original gradient norm."""
    for t in p.values():
        grad, w = t.grad, t.detach()
        coef = (w * grad).sum() / (w * w).sum().clamp_min(1e-12)
        perp = grad - coef * w
        t.grad = perp * (grad.norm() / perp.norm().clamp_min(1e-12))


VARIANTS = {
    #  name         weight decay   init                 grad transform
    "wd0":        (0.0,           g.init_params,        None),
    "orthograd":  (0.0,           g.init_params,        orthogonalize_grads_),
    "nanda-init": (1.0,           nanda_init_params,    None),
}


def weight_norm(p):
    return sum(t.norm().item() ** 2 for t in p.values()) ** 0.5


def train_variant(name, n_steps):
    wd, init_fn, grad_tf = VARIANTS[name]
    tag = name.replace("-", "_")

    torch.manual_seed(g.SEED)
    tx, ty, vx, vy = g.make_data(g.SEED)
    tx, ty = tx.to(g.DEVICE), ty.to(g.DEVICE)
    vx, vy = vx.to(g.DEVICE), vy.to(g.DEVICE)

    p = init_fn()
    print(f"[{name}] wd={wd} | init norm {weight_norm(p):.1f} | "
          f"device {g.DEVICE}")

    opt = torch.optim.AdamW(list(p.values()), lr=g.LR, betas=g.BETAS,
                            weight_decay=wd)

    log = []
    t0 = time.time()
    for step in range(n_steps + 1):
        train_loss, train_acc = g.loss_and_acc(p, tx, ty)

        if step % g.EVAL_EVERY == 0 or step == n_steps:
            with torch.no_grad():
                val_loss, val_acc = g.loss_and_acc(p, vx, vy)
                logits = g.forward(p, tx)[:, -1]
                per_sample = F.cross_entropy(logits, ty, reduction="none")
                probs = logits.softmax(-1).gather(1, ty[:, None]).squeeze(1)
            log.append({
                "step":           step,
                "train_loss":     train_loss.item(),
                "train_acc":      train_acc.item(),
                "val_loss":       val_loss.item(),
                "val_acc":        val_acc.item(),
                "weight_norm":    weight_norm(p),
                "frac_prob_one":  (probs == 1.0).float().mean().item(),
                "frac_loss_zero": (per_sample == 0.0).float().mean().item(),
            })
            if step % g.LOG_EVERY == 0 or step == n_steps:
                r = log[-1]
                print(f"[{name}] step {step:6d} | "
                      f"train A={r['train_acc']:.3f} | val A={r['val_acc']:.3f} | "
                      f"norm {r['weight_norm']:5.1f} | "
                      f"P(y)=1 on {r['frac_prob_one']:.3f} of train | "
                      f"{time.time() - t0:6.1f}s")

        if step == n_steps:
            break

        opt.zero_grad(set_to_none=True)
        train_loss.backward()
        if grad_tf is not None:
            grad_tf(p)
        opt.step()

    with open(f"training_log_{tag}.json", "w") as f:
        json.dump(log, f)
    torch.save({k: v.detach().cpu() for k, v in p.items()},
               f"params_{tag}.pt")
    print(f"[{name}] saved training_log_{tag}.json and params_{tag}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=list(VARIANTS) + ["all"])
    parser.add_argument("--steps", type=int, default=g.N_STEPS)
    args = parser.parse_args()

    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    for name in names:
        train_variant(name, args.steps)
