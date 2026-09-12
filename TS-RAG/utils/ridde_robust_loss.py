"""RIDDE retrieval-weight robust relative loss.

Inputs: a deterministic predictor accepting [batch, K] weights; fixed,
strictly positive reference attention weights; paired frozen baseline losses.
Run ``python ridde_robust_loss.py`` for CPU functional checks and a toy example.
This is an integration module, not an implementation of the user's backbone.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RobustConfig:
    rho: float = 0.1
    radius: float = 0.02
    inner_steps: int = 3
    step_size: float = 0.2
    random_restarts: int = 1
    bisection_steps: int = 24
    reduction: str = "sum"  # paper's squared L2 norm; "mean" for per-element MSE


def squared_error(pred: Tensor, target: Tensor, reduction: str = "sum") -> Tensor:
    """One loss per sample; never reduce the batch axis here."""
    if pred.shape != target.shape or pred.ndim < 2:
        raise ValueError("pred and target must have identical [batch, ...] shapes")
    values = (pred.float() - target.float()).square().flatten(1)
    if reduction == "sum":
        return values.sum(1)
    if reduction == "mean":
        return values.mean(1)
    raise ValueError("reduction must be 'sum' or 'mean'")


def kl_divergence(pi: Tensor, omega: Tensor) -> Tensor:
    """KL(pi || omega); omega must be strictly positive."""
    return (pi * (pi.clamp_min(torch.finfo(pi.dtype).tiny).log()
                  - omega.log())).sum(-1)


@torch.no_grad()
def retract_to_kl_ball(candidate: Tensor, omega: Tensor,
                       radius: float, steps: int = 24) -> Tensor:
    """Feasible radial contraction, NOT an exact KL/Euclidean projection.

    Search pi = (1-t)*omega + t*candidate. Convexity of KL in pi makes
    the feasible t values an interval starting at zero. Retain its feasible end.
    Input candidate and omega must already be probability vectors.
    """
    if radius == 0:
        return omega.clone()
    feasible = kl_divergence(candidate, omega) <= radius
    lo = torch.zeros_like(feasible, dtype=omega.dtype)
    hi = torch.ones_like(lo)
    for _ in range(steps):
        mid = (lo + hi) * 0.5
        pi = omega + mid[:, None] * (candidate - omega)
        ok = kl_divergence(pi, omega) <= radius
        lo = torch.where(ok, mid, lo)
        hi = torch.where(ok, hi, mid)
    shrunk = omega + lo[:, None] * (candidate - omega)
    return torch.where(feasible[:, None], candidate, shrunk)


def search_adversarial_weights(
    predict_inner: Callable[[Tensor], Tensor], target: Tensor,
    omega: Tensor, cfg: RobustConfig,
) -> Tensor:
    """Approximate per-sample maximization; no model optimizer step inside.

    predict_inner must be deterministic and sample-separable (no training-mode
    BatchNorm). Detach cached input features in this callback. Model parameters
    may require gradients: autograd.grad only requests weight-input gradients.
    Retaining each sample's best feasible candidate includes the nominal point.
    """
    if cfg.radius == 0:
        return omega.clone()
    best = omega.clone()
    with torch.no_grad():
        best_loss = squared_error(predict_inner(best), target, cfg.reduction)
    for restart in range(cfg.random_restarts + 1):
        if restart == 0:
            pi = omega.clone()
        else:
            proposal = (omega.log() + 0.2 * torch.randn_like(omega)).softmax(-1)
            pi = retract_to_kl_ball(proposal, omega, cfg.radius,
                                    cfg.bisection_steps)
        for step in range(cfg.inner_steps + 1):
            with torch.enable_grad():
                pi = pi.detach().requires_grad_(True)
                loss_i = squared_error(predict_inner(pi), target, cfg.reduction)
                if step < cfg.inner_steps and loss_i.requires_grad:
                    grad = torch.autograd.grad(loss_i.sum(), pi,
                                               allow_unused=True)[0]
                else:
                    grad = None
            with torch.no_grad():
                improve = loss_i.detach() > best_loss
                best = torch.where(improve[:, None], pi.detach(), best)
                best_loss = torch.maximum(best_loss, loss_i.detach())
                if step == cfg.inner_steps or grad is None:
                    break
                grad = grad - (omega * grad).sum(-1, keepdim=True)
                grad = grad / grad.abs().amax(-1, keepdim=True).clamp_min(1e-12)
                tiny = torch.finfo(pi.dtype).tiny
                proposal = (pi.clamp_min(tiny).log()
                            + cfg.step_size * grad).softmax(-1)
                pi = retract_to_kl_ball(proposal, omega, cfg.radius,
                                        cfg.bisection_steps)
    return best.detach()


def robust_relative_objective(
    predict: Callable[[Tensor], Tensor], target: Tensor, omega: Tensor,
    baseline_loss: Tensor, cfg: RobustConfig = RobustConfig(),
    predict_inner: Optional[Callable[[Tensor], Tensor]] = None,
):
    """Return (loss_with_graph, detached_diagnostics).

    omega is FIXED reference attention, reused for clean prediction and the KL
    centre. Do not pass freshly detached trainable attention and claim a fixed
    uncertainty-set objective: that is a different, moving-centre algorithm.
    baseline_loss must use the same targets, normalization and reduction.
    Ordinary random minibatches are supported; no environment sampler needed.
    """
    if cfg.rho < 0 or cfg.radius < 0 or cfg.inner_steps < 0:
        raise ValueError("rho, radius and inner_steps must be nonnegative")
    if cfg.step_size <= 0 or cfg.random_restarts < 0 or cfg.bisection_steps < 1:
        raise ValueError("invalid inner-search settings")
    if omega.requires_grad:
        raise ValueError("omega must come from a fixed reference weight generator")
    omega = omega.detach().float()
    if omega.ndim != 2 or omega.shape[0] != target.shape[0]:
        raise ValueError("omega must have shape [batch, K]")
    if not torch.isfinite(omega).all() or not (omega > 0).all():
        raise ValueError("reference softmax weights must be finite and positive")
    if not torch.allclose(omega.sum(-1), torch.ones_like(omega[:, 0]),
                          atol=1e-5, rtol=1e-5):
        raise ValueError("reference weights must sum to one")
    base = baseline_loss.detach().to(device=omega.device, dtype=torch.float32)
    if base.shape != (target.shape[0],) or not torch.isfinite(base).all():
        raise ValueError("baseline_loss must be finite and shaped [batch]")
    if (base < 0).any():
        raise ValueError("squared baseline losses must be nonnegative")

    inner = predict if predict_inner is None else predict_inner
    if cfg.rho == 0 or cfg.radius == 0:
        pi_star = omega.clone()
    else:
        pi_star = search_adversarial_weights(inner, target.detach(), omega, cfg)

    clean_i = squared_error(predict(omega), target, cfg.reduction)
    if cfg.rho == 0:
        loss = clean_i.mean()
        excess = clean_i.detach().new_zeros(clean_i.shape)
        adv_i = clean_i.detach()
    else:
        adv_i = (clean_i if cfg.radius == 0 else
                 squared_error(predict(pi_star), target, cfg.reduction))
        excess = (adv_i - base).clamp_min(0)
        loss = clean_i.mean() + cfg.rho * excess.mean()
    info = {
        "loss_pred": clean_i.detach().mean(),
        "loss_rob": excess.detach().mean(),
        "active_fraction": (excess.detach() > 0).float().mean(),
        "nominal_loss_per_sample": clean_i.detach(),
        "adversarial_loss_per_sample": adv_i.detach(),
        "kl": kl_divergence(pi_star, omega).detach(),
        "pi_star": pi_star,
    }
    return loss, info


class ExampleFusionHead(nn.Module):
    """Toy adapter for integration checks; NOT the user's RIDDE implementation."""
    def __init__(self, dim: int, horizon: int):
        super().__init__()
        self.query_gate = nn.Linear(2 * dim, dim)
        self.route_gate = nn.Linear(2 * dim, dim)
        self.inv_head = nn.Linear(dim, horizon)
        self.dyn_head = nn.Linear(dim, horizon)
        self.fusion = nn.Linear(2 * horizon, horizon)

    def forward(self, eq: Tensor, retrieved: Tensor, pi: Tensor) -> Tensor:
        hr = torch.einsum("bk,bkd->bd", pi, retrieved)
        lam = torch.sigmoid(self.query_gate(torch.cat([eq, hr], dim=-1)))
        h = lam * eq + (1 - lam) * hr
        gamma = torch.sigmoid(self.route_gate(torch.cat([h, hr], dim=-1)))
        inv = self.inv_head(gamma * h)
        dyn = self.dyn_head((1 - gamma) * h)
        return self.fusion(torch.cat([inv, dyn], dim=-1))


def functional_checks():
    """Numerical/gradient implementation checks only, not research experiments."""
    torch.manual_seed(17)
    batch, k, dim, horizon = 12, 5, 8, 4
    eq = torch.randn(batch, dim, requires_grad=True)
    retrieved = torch.randn(batch, k, dim, requires_grad=True)
    omega = torch.randn(batch, k).softmax(-1).detach()
    target = torch.randn(batch, horizon)
    baseline = torch.zeros(batch, requires_grad=True)
    model = ExampleFusionHead(dim, horizon).eval()
    outer = lambda pi: model(eq, retrieved, pi)
    inner = lambda pi: model(eq.detach(), retrieved.detach(), pi)
    cfg = RobustConfig(radius=0.03, inner_steps=4)
    loss, info = robust_relative_objective(outer, target, omega, baseline,
                                           cfg, inner)
    assert (info["kl"] <= cfg.radius + 2e-6).all()
    assert (info["pi_star"] >= 0).all()
    assert torch.allclose(info["pi_star"].sum(1), torch.ones(batch), atol=1e-6)
    assert (info["adversarial_loss_per_sample"] >=
            info["nominal_loss_per_sample"] - 2e-5).all()
    assert all(p.grad is None for p in model.parameters())
    loss.backward()
    assert baseline.grad is None
    assert model.fusion.weight.grad is not None
    assert torch.isfinite(model.fusion.weight.grad).all()
    assert model.fusion.weight.grad.norm() > 0
    assert retrieved.grad is not None and eq.grad is not None

    cfg0 = RobustConfig(radius=0)
    got, _ = robust_relative_objective(outer, target, omega, baseline, cfg0, inner)
    li = squared_error(outer(omega), target)
    expected = li.mean() + cfg0.rho * (li - baseline.detach()).relu().mean()
    assert torch.allclose(got, expected, atol=1e-6)
    erm, _ = robust_relative_objective(outer, target, omega, baseline,
                                      RobustConfig(rho=0), inner)
    assert torch.allclose(erm, li.mean(), atol=1e-6)
    _, inactive = robust_relative_objective(outer, target, omega,
        torch.full((batch,), 1e6), cfg0, inner)
    assert inactive["loss_rob"] == 0
    check_model = nn.Linear(k, 1).eval()
    check_target = torch.ones(batch, 1)
    optimizer = torch.optim.Adam(check_model.parameters(), lr=1e-3)
    cfg_check = RobustConfig(random_restarts=0, inner_steps=2)
    optimizer.zero_grad(set_to_none=True)
    value, _ = robust_relative_objective(check_model, check_target, omega,
                                         torch.zeros(batch), cfg_check)
    value.backward()
    optimizer.step()
    print("PASS: KL feasibility, simplex, best-candidate retention, fusion and")
    print("feature gradients, frozen baseline, zero radius, zero rho, inactive")
    print("hinge, and one optimizer update. No forecasting experiment was run.")
    print({"torch_version": torch.__version__, "device": "cpu",
           "max_kl": float(info["kl"].max()), "loss": float(loss.detach())})


if __name__ == "__main__":
    functional_checks()
