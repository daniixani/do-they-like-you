""" writing a monte carlo simulation to estimate dating success rate 
Preface: this code simulates the success rate of dating based on given parameters that are personal but
can be adjusted to fit different scenarios. It uses a closed-loop system with PID-like control. It gives points
based on events and includes a reply-delay parameter that simulates real-life delays in communication. 

In no way this code is meant to be taken seriously or as a real predictor of dating success. It is purely for entertainment and educational purposes.
However, I will be using it for my own personal use to help me understand my dating patterns better. I will record
the results and see if they align with my real-life experiences."""

from __future__ import annotations
import argparse
import dataclasses
from dataclasses import dataclass
from typing import Dict, List
import math
import csv
import sys
import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class Params:
    # Target & plant dynamics (toy)
    r: float = 0.82 # desired "connection" level
    a: float = 0.86 # state persistence
    b: float = 0.40 # controller influence
    c: float = 0.26 # care event influence
    d: float = 0.24 # disturbance influence
    process_sigma: float = 0.03 # process noise

    # PID-like controller gains
    kp: float = 0.95
    kd: float = 0.22
    ki: float = 0.10
    u_lo: float = -0.40
    u_hi: float = 0.60
    I_lo: float = -2.0
    I_hi: float = 2.0

    # Latent factors (means/sigmas in [0,1])
    external_load_mu: float = 0.75
    external_load_sigma: float = 0.12
    phone_noise_mu: float = 0.55
    phone_noise_sigma: float = 0.15
    comm_freq_mu: float = 0.50
    comm_freq_sigma: float = 0.18
    care_base_mu: float = 0.88
    care_base_sigma: float = 0.10
    openness_mu: float = 0.62
    openness_sigma: float = 0.12

    # Timeline & Monte Carlo
    T: int = 8
    N: int = 10000

    # Outcome thresholds
    schedule_threshold: float = 0.45
    defined_threshold: float = 0.72
    stable_threshold: float = 0.55
    fade_threshold: float = 0.35
    abrupt_early_cutoff: float = 0.20

    # Base event probabilities
    base_plan_prob: float = 0.38
    base_care_prob: float = 0.32
    care_event_mu: float = 0.65
    care_event_sigma: float = 0.18

    # Reply-delay effect
    reply_delay_days: float = 1.5
    delay_decay_per_day: float = 0.25 # attenuation rate per delay day


def _draw_latents(p: Params, rng: np.random.Generator):
    clamp01 = lambda z: 0.0 if z < 0.0 else 1.0 if z > 1.0 else z
    ext = clamp01(float(rng.normal(p.external_load_mu, p.external_load_sigma)))
    pho = clamp01(float(rng.normal(p.phone_noise_mu, p.phone_noise_sigma)))
    com = clamp01(float(rng.normal(p.comm_freq_mu, p.comm_freq_sigma)))
    car = clamp01(float(rng.normal(p.care_base_mu, p.care_base_sigma)))
    opn = clamp01(float(rng.normal(p.openness_mu, p.openness_sigma)))
    return ext, pho, com, car, opn


def _simulate_one(p: Params, rng: np.random.Generator, init_x: float) -> str:
    ext, pho, com, car, opn = _draw_latents(p, rng)

    # Reply-delay attenuation: comm effectiveness decays with delay
    att = math.exp(-p.delay_decay_per_day * max(0.0, p.reply_delay_days)) # in [0,1]
    com *= att
    # Long delay slightly increases effective phone "noise" (checking w/o engaging)
    pho = clamp(pho + 0.10 * (1.0 - att), 0.0, 1.0)

    x = clamp(init_x, 0.0, 1.0)
    y_prev = x
    I = 0.0
    abrupt_flag = False
    scheduled_count = 0

    for t in range(1, p.T + 1):
        y = x
        e = p.r - y
        I = clamp(I + e, p.I_lo, p.I_hi)
        Dy = y - y_prev

        u = p.kp * e - p.kd * Dy + p.ki * I
        u = clamp(u, p.u_lo, p.u_hi)

        # Disturbance = external load + phone/notification noise + small white noise
        w = 0.7 * ext + 0.3 * pho + float(rng.normal(0, 0.06))
        w = clamp(w, 0.0, 1.0)

        # Care event (tiny positive shocks)
        care_prob = p.base_care_prob + 0.25 * car + 0.15 * com + 0.10 * opn + 0.10 * y
        care_prob = clamp(care_prob, 0.0, 0.95)
        care_input = 0.0
        if rng.random() < care_prob:
            care_input = clamp(float(rng.normal(p.care_event_mu, p.care_event_sigma)), 0.0, 1.0)

        # Scheduling momentum (plans beget plans)
        plan_prob = p.base_plan_prob + 0.20 * com + 0.15 * y + 0.10 * opn
        if y > p.schedule_threshold:
            plan_prob += 0.10
        plan_prob = clamp(plan_prob, 0.0, 0.98)
        if rng.random() < plan_prob:
            scheduled_count += 1
            # Scheduling gives small bumps to com/care baseline
            com = clamp(com + float(rng.normal(0.05, 0.03)), 0.0, 1.0)
            car = clamp(car + float(rng.normal(0.04, 0.03)), 0.0, 1.0)

        # Plant update
        x_next = p.a * y + p.b * u + p.c * care_input - p.d * w + float(rng.normal(0, p.process_sigma))
        x_next = clamp(x_next, 0.0, 1.0)

        if t <= 2 and x_next < p.abrupt_early_cutoff:
            abrupt_flag = True

        y_prev = y
        x = x_next

    # Outcomes (mutually exclusive, ordered)
    if abrupt_flag:
        return "abrupt_disengagement"
    if x >= p.defined_threshold and scheduled_count >= 2:
        return "defined_relationship"
    if x >= p.stable_threshold and scheduled_count >= 1:
        return "stable_casual"
    if x <= p.fade_threshold:
        return "gradual_fade"
    return "stable_casual" if x >= 0.50 else "gradual_fade"


def monte_carlo(p: Params, seed: int, init_x: float) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    labels = ["defined_relationship", "stable_casual", "gradual_fade", "abrupt_disengagement"]
    counts = {k: 0 for k in labels}
    for _ in range(p.N):
        outcome = _simulate_one(p, rng, init_x=init_x)
        counts[outcome] += 1
    total = float(p.N)
    return {k: round(100.0 * v / total, 2) for k, v in counts.items()}


def cmd_run(args):
    p = Params(
    N=args.N,
    T=args.T,
    reply_delay_days=args.reply_delay_days,
    delay_decay_per_day=args.delay_decay_per_day,
)
res = monte_carlo(p, seed=args.seed, init_x=args.init_x)
print("Parameters:", dataclasses.asdict(p))
print(f"Initial x: {args.init_x}")
print("Results (%):")
for k in ["defined_relationship", "stable_casual", "gradual_fade", "abrupt_disengagement"]:
    print(f" {k:>22}: {res[k]:6.2f}%")


def cmd_sweep(args):
    delays = []
    d = args.min
    while d <= args.max + 1e-9:
        delays.append(round(d, 5))
        d += args.step

    rows: List[Dict[str, float]] = []
    for dd in delays:
        p = Params(
            N=args.N,
            T=args.T,
            reply_delay_days=dd,
            delay_decay_per_day=args.delay_decay_per_day,
        )
        res = monte_carlo(p, seed=args.seed, init_x=args.init_x)
        row = {"delay_days": dd, **res}
        rows.append(row)

    # Write CSV
    out_csv = args.out_csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_csv}")

    # Optional plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print("matplotlib not available; skipping plot.", file=sys.stderr)
            return

        x = [r["delay_days"] for r in rows]
        def_prob = [r["defined_relationship"] for r in rows]
        stable_prob = [r["stable_casual"] for r in rows]
        fade_prob = [r["gradual_fade"] for r in rows]
        abrupt_prob = [r["abrupt_disengagement"] for r in rows]

        plt.figure()
        plt.plot(x, def_prob, marker="o", label="defined_relationship")
        plt.plot(x, stable_prob, marker="o", label="stable_casual")
        plt.plot(x, fade_prob, marker="o", label="gradual_fade")
        plt.plot(x, abrupt_prob, marker="o", label="abrupt_disengagement")
        plt.xlabel("Reply delay (days)")
        plt.ylabel("Probability (%)")
        plt.title(f"Outcome probability vs reply delay (init_x={args.init_x})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_png = args.out_png
        plt.savefig(out_png, dpi=160)
        print(f"Saved: {out_png}")

def build_parser():
    ap = argparse.ArgumentParser(description="Monte Carlo control-system dating sim (with reply-delay)")
    sub = ap.add_subparsers(required=True, dest="cmd")

    # run
    runp = sub.add_parser("run", help="single Monte Carlo run")
    runp.add_argument("--N", type=int, default=8000)
    runp.add_argument("--T", type=int, default=8)
    runp.add_argument("--seed", type=int, default=2025)
    runp.add_argument("--init_x", type=float, default=0.70, help="initial connection level [0..1]")
    runp.add_argument("--reply_delay_days", type=float, default=1.5)
    runp.add_argument("--delay_decay_per_day", type=float, default=0.25)
    runp.set_defaults(func=cmd_run)

    # sweep
    swp = sub.add_parser("sweep", help="sweep reply delay and save CSV (+ optional plot)")
    swp.add_argument("--N", type=int, default=4000)
    swp.add_argument("--T", type=int, default=8)
    swp.add_argument("--seed", type=int, default=2025)
    swp.add_argument("--init_x", type=float, default=0.70)
    swp.add_argument("--delay_decay_per_day", type=float, default=0.25)
    swp.add_argument("--min", type=float, default=0.0)
    swp.add_argument("--max", type=float, default=4.0)
    swp.add_argument("--step", type=float, default=0.5)
    swp.add_argument("--out_csv", type=str, default="delay_sweep.csv")
    swp.add_argument("--out_png", type=str, default="delay_sweep.png")
    swp.add_argument("--plot", action="store_true")
swp.set_defaults(func=cmd_sweep)

return ap


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
	main() 