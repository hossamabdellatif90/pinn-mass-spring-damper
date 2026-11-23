#!/usr/bin/env python3
"""
Physics-Informed Neural Networks (PINNs) for a mass-spring-damper system
=======================================================================

Governing equation (1-DOF damped harmonic oscillator):

        m x''(t) + c x'(t) + k x(t) = F(t),      x(0) = x0,  x'(0) = v0

        m : mass [kg]        c : damping [N s/m]      k : stiffness [N/m]
        F(t) = F0 cos(wf t)  (optional harmonic forcing; F0 = 0 -> free vibration)

What this script does
---------------------
1. Reference solution  : closed form for free vibration (under/critically/over-damped),
                         high-accuracy ODE solver (scipy) when forcing is on.
2. Baseline            : an ordinary neural network fitted to a few data points only.
                         It matches the data but cannot extrapolate.
3. Forward PINN        : learns x(t) over the whole interval from the ODE + initial
                         conditions alone - no solution data at all.
4. Inverse PINN        : learns x(t) AND the unknown damping c and stiffness k from
                         noisy displacement measurements (system identification).
5. Graphs              : NN vs PINN, loss history, velocity / error / residual,
                         phase portrait, mechanical energy, parameter convergence.
6. Animations          : (a) training evolution NN vs PINN,
                         (b) physical spring-mass-damper moved by the PINN solution,
                         (c) inverse PINN discovering c and k.

Install & run
-------------
    pip install torch numpy matplotlib scipy pillow
    python pinn_mass_spring_damper.py                    # default problem
    python pinn_mass_spring_damper.py --c 1 --k 100      # a different oscillator
    python pinn_mass_spring_damper.py --F0 200 --wf 30   # forced vibration
    python pinn_mass_spring_damper.py --hard-ic          # enforce ICs exactly
    python pinn_mass_spring_damper.py --no-anim          # graphs only (faster)
    python pinn_mass_spring_damper.py --video both       # GIF + MP4 (needs ffmpeg)

All outputs are written to ./pinn_outputs/ (change with --outdir).
"""

import argparse
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # render to files; remove this line to use interactive windows
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.integrate import solve_ivp

DTYPE = torch.float64          # double precision -> cleaner derivatives & better L-BFGS
torch.set_default_dtype(DTYPE)


# --------------------------------------------------------------------------------------
# Plot style (one consistent palette for every figure and animation)
# --------------------------------------------------------------------------------------
PAL = dict(
    surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7",
    pinn="#2a78d6",     # blue   -> PINN (forward and inverse)
    nn="#eb6834",       # orange -> plain data-only neural network
    aux="#1baf7a",      # aqua   -> collocation points / physics loss
    aux2="#6250d6",     # violet -> initial-condition loss
    exact="#898781",    # gray   -> exact / reference solution (drawn as a wide underlay)
)


def set_style():
    plt.rcParams.update({
        "figure.facecolor": PAL["surface"], "axes.facecolor": PAL["surface"],
        "savefig.facecolor": PAL["surface"], "axes.edgecolor": PAL["axis"],
        "axes.labelcolor": PAL["ink2"], "xtick.color": PAL["muted"],
        "ytick.color": PAL["muted"], "text.color": PAL["ink"],
        "axes.grid": True, "grid.color": PAL["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlecolor": PAL["ink"], "font.size": 10, "legend.frameon": False,
        "legend.fontsize": 9, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def exact_line(ax, t, x, label="Exact solution", **kw):
    """Reference solution drawn as a wide translucent gray band underneath the models."""
    return ax.plot(t, x, color=PAL["exact"], lw=5, alpha=0.40, label=label, zorder=1, **kw)[0]


# --------------------------------------------------------------------------------------
# Physical system and its reference solution
# --------------------------------------------------------------------------------------
@dataclass
class Oscillator:
    m: float = 1.0
    c: float = 4.0
    k: float = 400.0
    x0: float = 1.0
    v0: float = 0.0
    F0: float = 0.0
    wf: float = 0.0
    T: float = 1.0

    @property
    def w0(self):            # natural frequency [rad/s]
        return math.sqrt(self.k / self.m)

    @property
    def zeta(self):          # damping ratio [-]
        return self.c / (2.0 * math.sqrt(self.k * self.m))

    def force(self, t):
        """External force F(t). Works on numpy arrays, floats and torch tensors."""
        if self.F0 == 0.0:
            return 0.0 * t
        cos = torch.cos if torch.is_tensor(t) else np.cos
        return self.F0 * cos(self.wf * t)

    def regime(self):
        z = self.zeta
        if self.F0 != 0.0:
            return "forced"
        return "underdamped" if z < 1 - 1e-9 else ("overdamped" if z > 1 + 1e-9 else "critically damped")


def exact_solution(osc: Oscillator, t):
    """Return displacement x(t) and velocity v(t) (numpy arrays)."""
    t = np.asarray(t, dtype=float)
    if osc.F0 != 0.0:  # forced: integrate numerically with tight tolerances
        def rhs(tt, y):
            return [y[1], (osc.force(tt) - osc.c * y[1] - osc.k * y[0]) / osc.m]
        sol = solve_ivp(rhs, (0.0, float(t.max())), [osc.x0, osc.v0], t_eval=t,
                        method="DOP853", rtol=1e-11, atol=1e-12)
        return sol.y[0], sol.y[1]

    w0, z, x0, v0 = osc.w0, osc.zeta, osc.x0, osc.v0
    if z < 1 - 1e-9:                                  # underdamped
        wd = w0 * np.sqrt(1 - z ** 2)
        A, B = x0, (v0 + z * w0 * x0) / wd
        e = np.exp(-z * w0 * t)
        x = e * (A * np.cos(wd * t) + B * np.sin(wd * t))
        v = e * ((-z * w0 * A + wd * B) * np.cos(wd * t) + (-z * w0 * B - wd * A) * np.sin(wd * t))
    elif z > 1 + 1e-9:                                # overdamped
        s = np.sqrt(z ** 2 - 1)
        r1, r2 = -w0 * (z - s), -w0 * (z + s)
        A = (v0 - r2 * x0) / (r1 - r2)
        B = x0 - A
        x = A * np.exp(r1 * t) + B * np.exp(r2 * t)
        v = A * r1 * np.exp(r1 * t) + B * r2 * np.exp(r2 * t)
    else:                                             # critically damped
        A, B = x0, v0 + w0 * x0
        e = np.exp(-w0 * t)
        x = (A + B * t) * e
        v = (B - w0 * (A + B * t)) * e
    return x, v


def reference_scale(osc: Oscillator):
    """Characteristic displacement used to make every loss term dimensionless."""
    return max(abs(osc.x0), abs(osc.v0) / osc.w0, abs(osc.F0) / osc.k, 1e-6)


# --------------------------------------------------------------------------------------
# Neural network
# --------------------------------------------------------------------------------------
class FCN(nn.Module):
    """Fully connected tanh network  t -> x(t).

    hard_ic=True builds the initial conditions into the output:
        x(t) = x0 + v0 t + (t/T)^2 * N(t)      ->  x(0) = x0 and x'(0) = v0 exactly.
    """

    def __init__(self, width=32, depth=3, T=1.0, hard_ic=False, x0=0.0, v0=0.0):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        self.T, self.hard_ic, self.x0, self.v0 = T, hard_ic, x0, v0
        for mod in self.net:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, t):
        out = self.net(2.0 * t / self.T - 1.0)        # map [0, T] -> [-1, 1]
        if self.hard_ic:
            out = self.x0 + self.v0 * t + (t / self.T) ** 2 * out
        return out


def d_dt(y, t):
    """dy/dt via automatic differentiation (keeps the graph for higher derivatives)."""
    return torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def ode_residual(model, t, m, c, k, osc):
    """r(t) = m x'' + c x' + k x - F(t)   (zero for the true solution)."""
    x = model(t)
    x_t = d_dt(x, t)
    x_tt = d_dt(x_t, t)
    return m * x_tt + c * x_t + k * x - osc.force(t)


def predict(model, t_np, with_velocity=False):
    t = torch.tensor(np.asarray(t_np, dtype=float)).view(-1, 1)
    if not with_velocity:
        with torch.no_grad():
            return model(t).numpy().ravel()
    t.requires_grad_(True)
    x = model(t)
    v = d_dt(x, t)
    return x.detach().numpy().ravel(), v.detach().numpy().ravel()


def residual_on_grid(model, osc, t_np):
    t = torch.tensor(np.asarray(t_np, dtype=float)).view(-1, 1).requires_grad_(True)
    r = ode_residual(model, t, osc.m, osc.c, osc.k, osc)
    return (r / (osc.k * reference_scale(osc))).detach().numpy().ravel()


def rel_l2(pred, true):
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))


# --------------------------------------------------------------------------------------
# 1) Baseline: plain neural network trained on data only
# --------------------------------------------------------------------------------------
def train_data_nn(osc, args, t_plot, t_data, x_data, snap_every):
    torch.manual_seed(args.seed)
    model = FCN(args.width, args.depth, osc.T)
    x_ref = reference_scale(osc)
    td = torch.tensor(t_data).view(-1, 1)
    xd = torch.tensor(x_data).view(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1 ** (1.0 / max(args.epochs, 1)))

    hist, snaps = dict(it=[], data=[]), []
    for it in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        loss = (((model(td) - xd) / x_ref) ** 2).mean()
        loss.backward()
        opt.step()
        sch.step()
        hist["it"].append(it)
        hist["data"].append(loss.item())
        if it % snap_every == 0:
            snaps.append((it, predict(model, t_plot)))
    snaps.append((args.epochs, predict(model, t_plot)))
    print(f"  data-only NN   final data loss {hist['data'][-1]:.3e}")
    return dict(model=model, hist=hist, snaps=snaps)


# --------------------------------------------------------------------------------------
# 2) Forward PINN: ODE + initial conditions, no solution data
# --------------------------------------------------------------------------------------
def train_forward_pinn(osc, args, t_plot, snap_every):
    torch.manual_seed(args.seed)
    model = FCN(args.width, args.depth, osc.T, hard_ic=args.hard_ic, x0=osc.x0, v0=osc.v0)
    x_ref = reference_scale(osc)
    t_col = torch.linspace(0, osc.T, args.n_col).view(-1, 1).requires_grad_(True)
    t0 = torch.zeros(1, 1, requires_grad=True)

    def losses():
        # physics residual, divided by k * x_ref  ->  x''/w0^2 + 2 zeta x'/w0 + x - F/k  (dimensionless)
        r = ode_residual(model, t_col, osc.m, osc.c, osc.k, osc) / (osc.k * x_ref)
        l_phys = (r ** 2).mean()
        x0p = model(t0)
        v0p = d_dt(x0p, t0)
        l_ic = (((x0p - osc.x0) / x_ref) ** 2 + ((v0p - osc.v0) / (osc.w0 * x_ref)) ** 2).squeeze()
        total = l_phys + (0.0 if args.hard_ic else args.w_ic) * l_ic
        return total, l_phys, l_ic

    hist, snaps = dict(it=[], total=[], phys=[], ic=[]), []
    it = 0

    def record(total, lp, li):
        hist["it"].append(it)
        hist["total"].append(total.item())
        hist["phys"].append(lp.item())
        hist["ic"].append(li.item())

    # ---- stage 1: Adam with exponential learning-rate decay
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1 ** (1.0 / max(args.epochs, 1)))
    t_start = time.time()
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        total, lp, li = losses()
        total.backward()
        opt.step()
        sch.step()
        record(total, lp, li)
        if it % snap_every == 0:
            snaps.append((it, predict(model, t_plot)))
        if it % max(args.epochs // 5, 1) == 0:
            print(f"  PINN Adam  it {it:6d}  loss {total.item():.3e}  "
                  f"(physics {lp.item():.2e}, IC {li.item():.2e})  {time.time() - t_start:5.1f}s")
        it += 1
    adam_end = it

    # ---- stage 2: L-BFGS polish (second-order, converges to much lower loss)
    if args.lbfgs > 0:
        opt2 = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=args.lbfgs,
                                 max_eval=int(args.lbfgs * 1.25), history_size=50,
                                 tolerance_grad=1e-12, tolerance_change=1e-15,
                                 line_search_fn="strong_wolfe")
        lb_snap = max(args.lbfgs // 20, 1)

        def closure():
            nonlocal it
            opt2.zero_grad(set_to_none=True)
            total, lp, li = losses()
            total.backward()
            record(total, lp, li)
            if (it - adam_end) % lb_snap == 0:
                snaps.append((it, predict(model, t_plot)))
            it += 1
            return total

        opt2.step(closure)
        print(f"  PINN L-BFGS  {it - adam_end} evals  loss {hist['total'][-1]:.3e}")
    snaps.append((it, predict(model, t_plot)))
    print(f"  forward PINN trained in {time.time() - t_start:.1f}s")
    return dict(model=model, hist=hist, snaps=snaps, adam_end=adam_end,
                t_col=t_col.detach().numpy().ravel())


# --------------------------------------------------------------------------------------
# 3) Inverse PINN: identify c and k from noisy measurements
# --------------------------------------------------------------------------------------
def train_inverse_pinn(osc, args, t_plot, t_obs, x_obs, snap_every):
    """m is assumed known. (In free vibration only c/m and k/m are identifiable,
    because scaling m, c, k by the same factor leaves the motion unchanged.)"""
    torch.manual_seed(args.seed + 1)
    model = FCN(args.width, args.depth, osc.T)
    x_ref = reference_scale(osc)
    log_c = nn.Parameter(torch.tensor(math.log(args.c_init)))   # log-parameters keep c, k > 0
    log_k = nn.Parameter(torch.tensor(math.log(args.k_init)))
    t_col = torch.linspace(0, osc.T, args.n_col).view(-1, 1).requires_grad_(True)
    to = torch.tensor(t_obs).view(-1, 1)
    xo = torch.tensor(x_obs).view(-1, 1)

    opt = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr},
                            {"params": [log_c, log_k], "lr": args.lr_phys}])
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1 ** (1.0 / max(args.inv_epochs, 1)))

    def losses():
        c, k = log_c.exp(), log_k.exp()
        r = ode_residual(model, t_col, osc.m, c, k, osc) / (k.detach() * x_ref)
        l_phys = (r ** 2).mean()
        l_data = (((model(to) - xo) / x_ref) ** 2).mean()
        return l_data + args.w_phys_inv * l_phys, l_data, l_phys, c, k

    hist, snaps = dict(it=[], data=[], phys=[], c=[], k=[]), []
    it = 0

    def record(l_data, l_phys, c, k):
        hist["it"].append(it)
        hist["data"].append(l_data.item())
        hist["phys"].append(l_phys.item())
        hist["c"].append(c.item())
        hist["k"].append(k.item())

    # ---- stage 1: Adam. The first `inv_warmup` iterations fit the data only (c, k frozen),
    #      so the physical parameters are not dragged around by a still-random network.
    t_start = time.time()
    for _ in range(args.inv_epochs):
        opt.zero_grad(set_to_none=True)
        loss, l_data, l_phys, c, k = losses()
        if it < args.inv_warmup:
            loss = l_data
        loss.backward()
        opt.step()
        sch.step()
        record(l_data, l_phys, c, k)
        if it % snap_every == 0:
            snaps.append((it, predict(model, t_plot), c.item(), k.item()))
        if it % max(args.inv_epochs // 5, 1) == 0:
            print(f"  inverse Adam it {it:6d}  data {l_data.item():.2e}  physics {l_phys.item():.2e}  "
                  f"c {c.item():7.3f}  k {k.item():8.2f}  {time.time() - t_start:5.1f}s")
        it += 1
    adam_end = it

    # ---- stage 2: L-BFGS polish of network weights AND physical parameters together
    if args.inv_lbfgs > 0:
        opt2 = torch.optim.LBFGS(list(model.parameters()) + [log_c, log_k], lr=1.0,
                                 max_iter=args.inv_lbfgs, max_eval=int(args.inv_lbfgs * 1.25),
                                 history_size=50, tolerance_grad=1e-12, tolerance_change=1e-15,
                                 line_search_fn="strong_wolfe")
        lb_snap = max(args.inv_lbfgs // 15, 1)

        def closure():
            nonlocal it
            opt2.zero_grad(set_to_none=True)
            loss, l_data, l_phys, c, k = losses()
            loss.backward()
            record(l_data, l_phys, c, k)
            if (it - adam_end) % lb_snap == 0:
                snaps.append((it, predict(model, t_plot), c.item(), k.item()))
            it += 1
            return loss

        opt2.step(closure)
        print(f"  inverse L-BFGS {it - adam_end} evals  data {hist['data'][-1]:.2e}  "
              f"physics {hist['phys'][-1]:.2e}  c {hist['c'][-1]:.4f}  k {hist['k'][-1]:.3f}")
    c_fin, k_fin = log_c.exp().item(), log_k.exp().item()
    snaps.append((it, predict(model, t_plot), c_fin, k_fin))
    print(f"  inverse PINN trained in {time.time() - t_start:.1f}s")
    return dict(model=model, hist=hist, snaps=snaps, c=c_fin, k=k_fin, adam_end=adam_end)


# --------------------------------------------------------------------------------------
# Static graphs
# --------------------------------------------------------------------------------------
def y_limits(x, pad_lo=0.30, pad_hi=0.18):
    lo, hi = float(np.min(x)), float(np.max(x))
    span = max(hi - lo, 1e-9)
    return lo - pad_lo * span, hi + pad_hi * span


def fig_comparison(osc, args, D, nn_res, pinn_res, path):
    t, xe = D["t_plot"], D["x_plot"]
    x_nn, x_pinn = nn_res["snaps"][-1][1], pinn_res["snaps"][-1][1]
    ylo, yhi = y_limits(xe)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)

    ax = axes[0]
    ax.axvspan(0, args.data_frac * osc.T, color=PAL["nn"], alpha=0.07, lw=0, label="Region with data")
    exact_line(ax, t, xe)
    ax.plot(t, x_nn, color=PAL["nn"], label="Neural network (data only)", zorder=3)
    ax.scatter(D["t_data"], D["x_data"], s=55, color=PAL["ink"], edgecolor=PAL["surface"],
               linewidth=1.5, zorder=4, label=f"Training data ({len(D['t_data'])} pts)")
    ax.set_title("Plain neural network: fits the data, fails to extrapolate")
    ax.text(0.02, 0.04, f"Relative L2 error = {rel_l2(x_nn, xe):.1%}", transform=ax.transAxes,
            color=PAL["ink2"])

    ax = axes[1]
    exact_line(ax, t, xe)
    ax.plot(t, x_pinn, color=PAL["pinn"], label="PINN", zorder=3)
    ax.plot(pinn_res["t_col"], np.full_like(pinn_res["t_col"], ylo + 0.03 * (yhi - ylo)), "|",
            color=PAL["aux"], ms=7, mew=1.2, label=f"Collocation points ({args.n_col})")
    ax.scatter([0], [osc.x0], s=70, color=PAL["ink"], edgecolor=PAL["surface"], linewidth=1.5,
               zorder=4, label="Initial condition")
    ax.set_title("PINN: ODE + initial conditions only, no data")
    ax.text(0.02, 0.10, f"Relative L2 error = {rel_l2(x_pinn, xe):.2e}", transform=ax.transAxes,
            color=PAL["ink2"])

    for ax in axes:
        ax.set_xlabel("time t [s]")
        ax.set_ylim(ylo, yhi)
        ax.legend(loc="upper right", ncol=1)
    axes[0].set_ylabel("displacement x(t) [m]")
    fig.suptitle(f"Mass-spring-damper  m={osc.m:g}, c={osc.c:g}, k={osc.k:g}  "
                 f"(ζ={osc.zeta:.3f}, ω₀={osc.w0:.2f} rad/s, {osc.regime()})",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_losses(nn_res, pinn_res, args, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    h = pinn_res["hist"]
    ax = axes[0]
    ax.semilogy(h["it"], h["total"], color=PAL["pinn"], label="Total loss", lw=2, zorder=3)
    ax.semilogy(h["it"], h["phys"], color=PAL["aux"], label="Physics (ODE residual)", lw=1.4)
    if not args.hard_ic:
        ax.semilogy(h["it"], h["ic"], color=PAL["aux2"], label="Initial conditions", lw=0.6, alpha=0.7)
    if args.lbfgs > 0:
        ax.axvline(pinn_res["adam_end"], color=PAL["muted"], ls="--", lw=1)
        ax.text(pinn_res["adam_end"], 0.97, "  Adam → L-BFGS", transform=ax.get_xaxis_transform(),
                va="top", color=PAL["ink2"], fontsize=9)
    ax.set_title("PINN training loss")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (dimensionless)")
    ax.legend(loc="lower left")

    ax = axes[1]
    ax.semilogy(nn_res["hist"]["it"], nn_res["hist"]["data"], color=PAL["nn"], lw=2)
    ax.set_title("Data-only NN training loss (mean squared error)")
    ax.set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_accuracy(osc, D, pinn_res, path):
    t = D["t_dense"]
    xe, ve = D["x_dense"], D["v_dense"]
    xp, vp = predict(pinn_res["model"], t, with_velocity=True)
    res = residual_on_grid(pinn_res["model"], osc, t)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    ax = axes[0]
    exact_line(ax, t, ve)
    ax.plot(t, vp, color=PAL["pinn"], label="PINN (autograd dx/dt)")
    ax.set_title("Velocity x'(t)")
    ax.set_ylabel("velocity [m/s]")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.semilogy(t, np.abs(xp - xe) + 1e-16, color=PAL["pinn"], lw=1.4, label="|x_PINN − x_exact|")
    ax.semilogy(t, np.abs(vp - ve) / osc.w0 + 1e-16, color=PAL["aux2"], lw=1.4,
                label="|v_PINN − v_exact| / ω₀")
    ax.set_title("Pointwise error")
    ax.set_ylabel("absolute error [m]")
    ax.legend(loc="lower right")

    ax = axes[2]
    ax.semilogy(t, np.abs(res) + 1e-16, color=PAL["aux"], lw=1.4)
    ax.set_title("ODE residual |m x'' + c x' + k x − F| / (k·x_ref)")
    ax.set_ylabel("normalized residual")
    for ax in axes:
        ax.set_xlabel("time t [s]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_phase_energy(osc, D, pinn_res, path):
    t = D["t_dense"]
    xe, ve = D["x_dense"], D["v_dense"]
    xp, vp = predict(pinn_res["model"], t, with_velocity=True)
    E = lambda x, v: 0.5 * osc.m * v ** 2 + 0.5 * osc.k * x ** 2
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    exact_line(ax, xe, ve)
    ax.plot(xp, vp, color=PAL["pinn"], label="PINN")
    ax.scatter([xe[0]], [ve[0]], s=60, color=PAL["ink"], edgecolor=PAL["surface"], lw=1.5,
               zorder=4, label="start (t = 0)")
    ax.set_title("Phase portrait")
    ax.set_xlabel("displacement x [m]")
    ax.set_ylabel("velocity x' [m/s]")
    ax.legend(loc="upper right")

    ax = axes[1]
    exact_line(ax, t, E(xe, ve))
    ax.plot(t, E(xp, vp), color=PAL["pinn"], label="PINN")
    ax.set_title("Mechanical energy  ½ m x'² + ½ k x²")
    ax.set_xlabel("time t [s]")
    ax.set_ylabel("energy [J]")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def mark_inverse_stages(ax, args, inv):
    """Shade the data-only warm-up and mark the switch from Adam to L-BFGS."""
    if args.inv_warmup > 0:
        ax.axvspan(0, args.inv_warmup, color=PAL["grid"], alpha=0.6, lw=0)
        ax.text(args.inv_warmup / 2, 0.96, "warm-up", transform=ax.get_xaxis_transform(),
                ha="center", va="top", color=PAL["ink2"], fontsize=8)
    if args.inv_lbfgs > 0:
        ax.axvline(inv["adam_end"], color=PAL["muted"], ls=":", lw=1)


def true_value_line(ax, key, true, values):
    """Dashed reference at the true parameter, labelled away from the converged curve,
    with head-room on top so the 'warm-up' label never touches it."""
    ax.axhline(true, color=PAL["ink2"], ls="--", lw=1.2)
    ax.text(0.30, true, f" true {key} = {true:g}", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", color=PAL["ink2"], fontsize=9)
    vals = np.append(np.asarray(values, dtype=float), true)
    span = max(np.ptp(vals), 1e-9)
    ax.set_ylim(vals.min() - 0.08 * span, vals.max() + 0.28 * span)


def fig_inverse(osc, args, D, inv, path):
    t, xe = D["t_plot"], D["x_plot"]
    h = inv["hist"]
    fig = plt.figure(figsize=(12.5, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.55, wspace=0.22)
    ax = fig.add_subplot(gs[:, 0])
    exact_line(ax, t, xe)
    ax.scatter(D["t_obs"], D["x_obs"], s=40, color=PAL["ink"], edgecolor=PAL["surface"], lw=1.2,
               zorder=3, label=f"Noisy measurements ({len(D['t_obs'])} pts)")
    ax.plot(t, inv["snaps"][-1][1], color=PAL["pinn"], label="Inverse PINN", zorder=4)
    ax.set_title("Inverse PINN: fit to noisy data + ODE with unknown c, k")
    ax.set_xlabel("time t [s]")
    ax.set_ylabel("displacement x(t) [m]")
    ax.legend(loc="upper right")

    for row, (key, true, unit) in enumerate([("c", osc.c, "N·s/m"), ("k", osc.k, "N/m")]):
        a = fig.add_subplot(gs[row, 1])
        mark_inverse_stages(a, args, inv)
        a.plot(h["it"], h[key], color=PAL["pinn"])
        true_value_line(a, key, true, h[key])
        err = abs(h[key][-1] - true) / true
        a.set_title(f"Learned {key}: {h[key][-1]:.4g} {unit}  (error {err:.2%})")
        a.set_xlabel("iteration")
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Animations
# --------------------------------------------------------------------------------------
def save_animation(anim, path_noext, args):
    saved = []
    if args.video in ("gif", "both"):
        anim.save(path_noext + ".gif", writer=animation.PillowWriter(fps=args.fps), dpi=args.anim_dpi)
        saved.append(path_noext + ".gif")
    if args.video in ("mp4", "both"):
        if animation.writers.is_available("ffmpeg"):
            w = animation.FFMpegWriter(fps=args.fps, codec="libx264", bitrate=3000,
                                       extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # H.264 needs even size
                                                   "-pix_fmt", "yuv420p", "-preset", "veryfast"])
            try:
                anim.save(path_noext + ".mp4", writer=w, dpi=args.anim_dpi)
                saved.append(path_noext + ".mp4")
            except Exception as e:  # never lose the run because of a video encoder problem
                print(f"  MP4 export failed ({e.__class__.__name__}); GIF output is unaffected")
        else:
            print("  ffmpeg not found - skipping MP4 (install ffmpeg or use --video gif)")
    for p in saved:
        print(f"  saved {p}")
    return saved


def animate_training(osc, args, D, nn_res, pinn_res, path_noext):
    """NN vs PINN predictions evolving during training, with the loss curves below."""
    t, xe = D["t_plot"], D["x_plot"]
    ylo, yhi = y_limits(xe)
    p_snaps, n_snaps = pinn_res["snaps"], nn_res["snaps"]
    hp, hn = pinn_res["hist"], nn_res["hist"]

    fig = plt.figure(figsize=(10.5, 6.6), dpi=90)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.45, wspace=0.15)
    axN, axP, axL = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    axN.axvspan(0, args.data_frac * osc.T, color=PAL["nn"], alpha=0.07, lw=0)
    exact_line(axN, t, xe)
    axN.scatter(D["t_data"], D["x_data"], s=45, color=PAL["ink"], edgecolor=PAL["surface"],
                lw=1.3, zorder=4, label="data")
    lnN, = axN.plot([], [], color=PAL["nn"], zorder=3, label="NN (data only)")
    axN.set_title("Plain neural network")

    exact_line(axP, t, xe)
    axP.plot(pinn_res["t_col"], np.full_like(pinn_res["t_col"], ylo + 0.03 * (yhi - ylo)), "|",
             color=PAL["aux"], ms=6, mew=1.0, label="collocation pts")
    axP.scatter([0], [osc.x0], s=55, color=PAL["ink"], edgecolor=PAL["surface"], lw=1.3, zorder=4,
                label="initial condition")
    lnP, = axP.plot([], [], color=PAL["pinn"], zorder=3, label="PINN")
    axP.set_title("Physics-informed neural network")
    axP.tick_params(labelleft=False)
    for ax in (axN, axP):
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(0, osc.T)
        ax.set_xlabel("time t [s]")
        ax.legend(loc="upper right", fontsize=8)
    axN.set_ylabel("x(t) [m]")

    lossP, = axL.semilogy([], [], color=PAL["pinn"], label="PINN total loss")
    lossN, = axL.semilogy([], [], color=PAL["nn"], label="NN data loss")
    allv = np.concatenate([hp["total"], hn["data"]])
    allv = allv[allv > 0]
    axL.set_xlim(0, hp["it"][-1])
    axL.set_ylim(allv.min() * 0.5, allv.max() * 2)
    if args.lbfgs > 0:
        axL.axvline(pinn_res["adam_end"], color=PAL["muted"], ls="--", lw=1)
        axL.text(pinn_res["adam_end"], 0.95, "  L-BFGS", transform=axL.get_xaxis_transform(),
                 va="top", color=PAL["ink2"], fontsize=9)
    axL.set_xlabel("iteration")
    axL.set_ylabel("loss")
    axL.legend(loc="upper right")
    title = fig.suptitle("", x=0.01, ha="left", fontsize=12, fontweight="bold")

    def update(i):
        itP, xP = p_snaps[i]
        itN, xN = n_snaps[min(i, len(n_snaps) - 1)]
        lnP.set_data(t, xP)
        lnN.set_data(t, xN)
        kP = np.searchsorted(hp["it"], itP, side="right")
        kN = np.searchsorted(hn["it"], itN, side="right")
        lossP.set_data(hp["it"][:kP], hp["total"][:kP])
        lossN.set_data(hn["it"][:kN], hn["data"][:kN])
        phase = "Adam" if itP < pinn_res["adam_end"] else "L-BFGS"
        title.set_text(f"Training iteration {itP:,}  ({phase})   —   "
                       f"rel. L2 error: NN {rel_l2(xN, xe):.1%},  PINN {rel_l2(xP, xe):.1%}")
        return lnP, lnN, lossP, lossN, title

    frames = list(range(len(p_snaps))) + [len(p_snaps) - 1] * int(args.fps * 1.5)  # hold last frame
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    out = save_animation(anim, path_noext, args)
    plt.close(fig)
    return out


def spring_xy(x_start, x_end, y, n_coils=9, amp=0.11, lead=0.25):
    """Zig-zag polyline for a coil spring from x_start to x_end at height y."""
    L = x_end - x_start
    lead = min(lead, 0.2 * L)
    xs = [x_start, x_start + lead]
    ys = [y, y]
    n = 2 * n_coils
    for j in range(1, n):
        xs.append(x_start + lead + (L - 2 * lead) * j / n)
        ys.append(y + (amp if j % 2 else -amp))
    xs += [x_end - lead, x_end]
    ys += [y, y]
    return np.array(xs), np.array(ys)


def animate_system(osc, args, D, pinn_res, path_noext):
    """Physical spring-mass-damper whose motion is the PINN prediction."""
    t = D["t_dense"]
    xe, ve = D["x_dense"], D["v_dense"]
    xp, vp = predict(pinn_res["model"], t, with_velocity=True)
    n_frames = int(args.fps * args.sys_seconds)
    idx = np.linspace(0, len(t) - 1, n_frames).round().astype(int)
    amp = 0.9 / max(np.abs(xe).max(), 1e-12)      # drawing scale: max excursion = 0.9 units

    fig = plt.figure(figsize=(10.5, 6.6), dpi=90)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1.6, 1], hspace=0.35, wspace=0.25)
    axS, axX, axV = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # --- mechanical drawing -------------------------------------------------------
    EQ, W, H, Y0 = 3.0, 1.0, 0.86, 0.14          # equilibrium left face, mass width/height, mass bottom
    axS.set_xlim(-0.45, 6.0)
    axS.set_ylim(-0.30, 1.42)
    axS.set_aspect("equal")
    axS.axis("off")
    axS.add_patch(Rectangle((-0.30, -0.05), 0.30, 1.30, facecolor="none", edgecolor=PAL["ink2"],
                            hatch="///", lw=0))
    axS.plot([0, 0], [-0.05, 1.25], color=PAL["ink2"], lw=2.5)
    axS.plot([-0.30, 6.0], [0, 0], color=PAL["ink2"], lw=2)
    axS.add_patch(Rectangle((0, -0.12), 6.0, 0.12, facecolor="none", edgecolor=PAL["axis"],
                            hatch="///", lw=0))
    axS.plot([EQ + W / 2] * 2, [0.02, 1.22], color=PAL["muted"], ls="--", lw=1)
    axS.text(EQ + W / 2, 1.27, "x = 0", ha="center", color=PAL["muted"], fontsize=9)

    yS, yD = Y0 + 0.66 * H, Y0 + 0.28 * H        # spring and damper heights
    spring, = axS.plot([], [], color=PAL["ink2"], lw=1.8)
    cyl_l, cyl_r, cyl_h = 0.55, 2.55, 0.10       # dashpot cylinder (fixed to wall)
    axS.plot([0, cyl_l], [yD, yD], color=PAL["ink2"], lw=1.8)
    axS.plot([cyl_l, cyl_r], [yD + cyl_h] * 2, color=PAL["ink2"], lw=1.8)
    axS.plot([cyl_l, cyl_r], [yD - cyl_h] * 2, color=PAL["ink2"], lw=1.8)
    axS.plot([cyl_l, cyl_l], [yD - cyl_h, yD + cyl_h], color=PAL["ink2"], lw=1.8)
    piston, = axS.plot([], [], color=PAL["ink2"], lw=2.6)
    rod, = axS.plot([], [], color=PAL["ink2"], lw=1.8)
    mass = Rectangle((EQ, Y0), W, H, facecolor=PAL["pinn"], edgecolor=PAL["surface"], lw=2, zorder=3)
    axS.add_patch(mass)
    wheels = [Circle((EQ + f * W, Y0 / 2), Y0 / 2 - 0.01, facecolor=PAL["ink2"], zorder=3)
              for f in (0.25, 0.75)]
    for w in wheels:
        axS.add_patch(w)
    mlabel = axS.text(EQ + W / 2, Y0 + H / 2, "m", ha="center", va="center", color="white",
                      fontsize=14, fontweight="bold", zorder=4)
    axS.text(1.3, yS + 0.2, "k", ha="center", color=PAL["ink2"], fontsize=11)
    axS.text(1.55, yD - cyl_h - 0.2, "c", ha="center", color=PAL["ink2"], fontsize=11)
    arrow = FancyArrowPatch((EQ + W / 2, 1.12), (EQ + W / 2, 1.12), arrowstyle="-|>",
                            mutation_scale=12, color=PAL["ink"], lw=1.5)
    axS.add_patch(arrow)
    info = axS.text(6.0, 1.30, "", ha="right", va="top", color=PAL["ink"], fontsize=10,
                    family="monospace")
    axS.text(-0.45, 1.42, f"m = {osc.m:g} kg   c = {osc.c:g} N·s/m   k = {osc.k:g} N/m   "
             f"ζ = {osc.zeta:.3f}", ha="left", va="top", color=PAL["ink2"], fontsize=10)

    # --- x(t) trace ---------------------------------------------------------------
    exact_line(axX, t, xe)
    trX, = axX.plot([], [], color=PAL["pinn"], label="PINN", zorder=3)
    dotX, = axX.plot([], [], "o", color=PAL["pinn"], ms=8, mec=PAL["surface"], mew=2, zorder=4)
    axX.set_xlim(0, osc.T)
    axX.set_ylim(*y_limits(xe, 0.12, 0.12))
    axX.set_xlabel("time t [s]")
    axX.set_ylabel("x(t) [m]")
    axX.set_title("Displacement")
    axX.legend(loc="upper right")

    # --- phase portrait ------------------------------------------------------------
    exact_line(axV, xe, ve)
    trV, = axV.plot([], [], color=PAL["pinn"], zorder=3)
    dotV, = axV.plot([], [], "o", color=PAL["pinn"], ms=8, mec=PAL["surface"], mew=2, zorder=4)
    axV.set_xlabel("x [m]")
    axV.set_ylabel("x' [m/s]")
    axV.set_title("Phase portrait")
    slow = args.sys_seconds / osc.T
    fig.suptitle(f"Mass-spring-damper driven by the PINN solution  (playback {slow:.1f}× slower)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")

    def update(f):
        j = idx[f]
        left = EQ + amp * xp[j]
        mass.set_x(left)
        for w, frac in zip(wheels, (0.25, 0.75)):
            w.center = (left + frac * W, Y0 / 2)
        mlabel.set_position((left + W / 2, Y0 + H / 2))
        spring.set_data(*spring_xy(0.0, left, yS))
        p = left - 1.45                                   # piston stays inside the cylinder
        piston.set_data([p, p], [yD - cyl_h * 0.8, yD + cyl_h * 0.8])
        rod.set_data([p, left], [yD, yD])
        arrow.set_positions((EQ + W / 2, 1.12), (left + W / 2, 1.12))
        info.set_text(f"t = {t[j]:6.3f} s\nx = {xp[j]:+7.3f} m\nv = {vp[j]:+7.2f} m/s")
        trX.set_data(t[:j + 1], xp[:j + 1])
        dotX.set_data([t[j]], [xp[j]])
        trV.set_data(xp[:j + 1], vp[:j + 1])
        dotV.set_data([xp[j]], [vp[j]])
        return mass, spring, piston, rod, trX, dotX, trV, dotV, info

    frames = list(range(n_frames)) + [n_frames - 1] * int(args.fps * 1.0)
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    out = save_animation(anim, path_noext, args)
    plt.close(fig)
    return out


def animate_inverse(osc, args, D, inv, path_noext):
    t, xe = D["t_plot"], D["x_plot"]
    h, snaps = inv["hist"], inv["snaps"]
    fig = plt.figure(figsize=(10.5, 5.2), dpi=90)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], hspace=0.6, wspace=0.25)
    ax = fig.add_subplot(gs[:, 0])
    axc, axk = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    exact_line(ax, t, xe)
    ax.scatter(D["t_obs"], D["x_obs"], s=35, color=PAL["ink"], edgecolor=PAL["surface"], lw=1.1,
               zorder=3, label="noisy data")
    ln, = ax.plot([], [], color=PAL["pinn"], zorder=4, label="inverse PINN")
    ax.set_ylim(*y_limits(np.concatenate([xe, D["x_obs"]]), 0.12, 0.25))
    ax.set_xlim(0, osc.T)
    ax.set_xlabel("time t [s]")
    ax.set_ylabel("x(t) [m]")
    ax.set_title("Fit to measurements")
    ax.legend(loc="upper right")

    lines = {}
    for a, key, true in ((axc, "c", osc.c), (axk, "k", osc.k)):
        mark_inverse_stages(a, args, inv)
        true_value_line(a, key, true, h[key])
        lines[key], = a.plot([], [], color=PAL["pinn"])
        a.set_xlim(0, h["it"][-1])
        a.set_xlabel("iteration")
        a.set_title(f"Learned {key}")
    title = fig.suptitle("", x=0.01, ha="left", fontsize=12, fontweight="bold")

    def update(i):
        it, xpred, c, k = snaps[i]
        ln.set_data(t, xpred)
        n = min(it + 1, len(h["it"]))
        lines["c"].set_data(h["it"][:n], h["c"][:n])
        lines["k"].set_data(h["it"][:n], h["k"][:n])
        title.set_text(f"Inverse PINN, iteration {it:,}:   c = {c:.3f}  (true {osc.c:g}),   "
                       f"k = {k:.1f}  (true {osc.k:g})")
        return ln, lines["c"], lines["k"], title

    frames = list(range(len(snaps))) + [len(snaps) - 1] * int(args.fps * 1.5)
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    out = save_animation(anim, path_noext, args)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PINN for a mass-spring-damper system",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("physical system")
    g.add_argument("--m", type=float, default=1.0, help="mass [kg]")
    g.add_argument("--c", type=float, default=4.0, help="damping coefficient [N s/m]")
    g.add_argument("--k", type=float, default=400.0, help="spring stiffness [N/m] (> 0)")
    g.add_argument("--x0", type=float, default=1.0, help="initial displacement [m]")
    g.add_argument("--v0", type=float, default=0.0, help="initial velocity [m/s]")
    g.add_argument("--F0", type=float, default=0.0, help="forcing amplitude, F(t) = F0 cos(wf t)")
    g.add_argument("--wf", type=float, default=0.0, help="forcing angular frequency [rad/s]")
    g.add_argument("--T", type=float, default=1.0, help="time horizon [s]")

    g = p.add_argument_group("network & forward PINN")
    g.add_argument("--width", type=int, default=32)
    g.add_argument("--depth", type=int, default=3, help="number of hidden layers")
    g.add_argument("--epochs", type=int, default=15000, help="Adam iterations (forward PINN and NN)")
    g.add_argument("--lbfgs", type=int, default=5000, help="L-BFGS iterations after Adam (0 = off)")
    g.add_argument("--lr", type=float, default=3e-3, help="initial Adam learning rate (decays 10x)")
    g.add_argument("--n-col", type=int, default=200, help="collocation points")
    g.add_argument("--w-ic", type=float, default=10.0, help="weight of the initial-condition loss")
    g.add_argument("--hard-ic", action="store_true", help="enforce ICs exactly through the network output")

    g = p.add_argument_group("data-only baseline")
    g.add_argument("--n-data", type=int, default=10, help="number of training points")
    g.add_argument("--data-frac", type=float, default=0.4, help="data lies in the first fraction of [0, T]")

    g = p.add_argument_group("inverse PINN")
    g.add_argument("--skip-inverse", action="store_true")
    g.add_argument("--inv-epochs", type=int, default=15000, help="Adam iterations")
    g.add_argument("--inv-lbfgs", type=int, default=2000, help="L-BFGS iterations after Adam (0 = off)")
    g.add_argument("--inv-warmup", type=int, default=2000, help="data-only iterations before physics is switched on")
    g.add_argument("--n-obs", type=int, default=40, help="number of noisy measurements")
    g.add_argument("--noise", type=float, default=0.04, help="noise std as a fraction of x_ref")
    g.add_argument("--c-init", type=float, default=1.0, help="initial guess for c (> 0)")
    g.add_argument("--k-init", type=float, default=150.0, help="initial guess for k (> 0)")
    g.add_argument("--lr-phys", type=float, default=1e-2, help="learning rate for log c, log k")
    g.add_argument("--w-phys-inv", type=float, default=100.0, help="physics-loss weight")

    g = p.add_argument_group("output")
    g.add_argument("--outdir", default="pinn_outputs")
    g.add_argument("--no-anim", action="store_true", help="skip animations")
    g.add_argument("--video", choices=["gif", "mp4", "both"], default="gif")
    g.add_argument("--fps", type=int, default=20)
    g.add_argument("--anim-dpi", type=int, default=100, help="animation resolution")
    g.add_argument("--sys-seconds", type=float, default=6.0, help="duration of the system animation")
    g.add_argument("--snaps", type=int, default=90, help="training snapshots per animation")
    g.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    set_style()
    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.k <= 0 or args.m <= 0:
        raise SystemExit("m and k must be positive")

    osc = Oscillator(args.m, args.c, args.k, args.x0, args.v0, args.F0, args.wf, args.T)
    print(f"System: m={osc.m}, c={osc.c}, k={osc.k}  ->  w0={osc.w0:.3f} rad/s, "
          f"zeta={osc.zeta:.4f} ({osc.regime()})")

    # ---- reference data --------------------------------------------------------------
    D = {}
    D["t_plot"] = np.linspace(0, osc.T, 400)
    D["x_plot"], _ = exact_solution(osc, D["t_plot"])
    D["t_dense"] = np.linspace(0, osc.T, 1200)
    D["x_dense"], D["v_dense"] = exact_solution(osc, D["t_dense"])
    D["t_data"] = np.linspace(0, args.data_frac * osc.T, args.n_data)
    D["x_data"], _ = exact_solution(osc, D["t_data"])
    rng = np.random.default_rng(args.seed)
    D["t_obs"] = np.sort(rng.uniform(0, osc.T, args.n_obs))
    x_clean, _ = exact_solution(osc, D["t_obs"])
    D["x_obs"] = x_clean + args.noise * reference_scale(osc) * rng.standard_normal(args.n_obs)

    snap_every = max(args.epochs // args.snaps, 1)

    # ---- training -------------------------------------------------------------------
    print("\n[1/3] Data-only neural network")
    nn_res = train_data_nn(osc, args, D["t_plot"], D["t_data"], D["x_data"], snap_every)
    print("\n[2/3] Forward PINN (physics + initial conditions)")
    pinn_res = train_forward_pinn(osc, args, D["t_plot"], snap_every)
    inv = None
    if not args.skip_inverse:
        print("\n[3/3] Inverse PINN (learn c and k from noisy data)")
        inv = train_inverse_pinn(osc, args, D["t_plot"], D["t_obs"], D["x_obs"],
                                 max(args.inv_epochs // args.snaps, 1))

    # ---- metrics --------------------------------------------------------------------
    x_nn, x_pinn = nn_res["snaps"][-1][1], pinn_res["snaps"][-1][1]
    lines = [f"System: m={osc.m:g}, c={osc.c:g}, k={osc.k:g}, x0={osc.x0:g}, v0={osc.v0:g}, "
             f"F0={osc.F0:g}, wf={osc.wf:g}, T={osc.T:g}",
             f"  natural frequency w0 = {osc.w0:.4f} rad/s, damping ratio zeta = {osc.zeta:.4f} ({osc.regime()})",
             f"Relative L2 error, data-only NN : {rel_l2(x_nn, D['x_plot']):.4e}",
             f"Relative L2 error, forward PINN : {rel_l2(x_pinn, D['x_plot']):.4e}"]
    if inv:
        lines += [f"Inverse PINN: c = {inv['c']:.4f} (true {osc.c:g}, error {abs(inv['c'] - osc.c) / osc.c:.2%})",
                  f"              k = {inv['k']:.3f} (true {osc.k:g}, error {abs(inv['k'] - osc.k) / osc.k:.2%})",
                  f"              relative L2 error of x(t): {rel_l2(inv['snaps'][-1][1], D['x_plot']):.4e}"]
    print("\n" + "\n".join(lines))
    with open(os.path.join(args.outdir, "results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    torch.save(pinn_res["model"].state_dict(), os.path.join(args.outdir, "forward_pinn.pt"))

    # ---- graphs ---------------------------------------------------------------------
    print("\nSaving graphs ...")
    out = os.path.join
    fig_comparison(osc, args, D, nn_res, pinn_res, out(args.outdir, "fig1_nn_vs_pinn.png"))
    fig_losses(nn_res, pinn_res, args, out(args.outdir, "fig2_training_loss.png"))
    fig_accuracy(osc, D, pinn_res, out(args.outdir, "fig3_velocity_error_residual.png"))
    fig_phase_energy(osc, D, pinn_res, out(args.outdir, "fig4_phase_energy.png"))
    if inv:
        fig_inverse(osc, args, D, inv, out(args.outdir, "fig5_inverse_parameters.png"))

    # ---- animations -----------------------------------------------------------------
    if not args.no_anim:
        print("Rendering animations (this takes a minute) ...")
        animate_training(osc, args, D, nn_res, pinn_res, out(args.outdir, "anim1_training"))
        animate_system(osc, args, D, pinn_res, out(args.outdir, "anim2_mass_spring_damper"))
        if inv:
            animate_inverse(osc, args, D, inv, out(args.outdir, "anim3_inverse_pinn"))
    print(f"Done. Outputs in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
