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
    r: float = 0.82
    a: float = 0.86
    b: float = 0.40
    c: float = 0.26
    d: float = 0.24
    process_sigma: float = 0.03

    kp: float = 0.95
    kd: float = 0.22
    ki: float = 0.10
    u_lo: float = -0.40
    u_hi: float = 0.60
    I_lo: float = -2.0
    I_hi: float = 2.0

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

    T: int = 8
    N: int = 10000

    schedule_threshold: float = 0.45
    defined_threshold: float = 0.72
    stable_threshold: float = 0.55
    fade_threshold: float = 0.35
    abrupt_early_cutoff: float = 0.20

    base_plan_prob: float = 0.38
    base_care_prob: float = 0.32
    care_event_mu: float = 0.65
    care_event_sigma: float = 0.18

    reply_delay_days: float = 1.5
    delay_decay_per_day: float = 0.25


def _draw_latents(p: Params, rng: np.random.Generator):
    def clamp01(z): return 0.0 if z < 0.0 else 1.0 if z > 1.0 else z
    ext = clamp01(float(rng.normal(p.external_load_mu, p.external_load_sigma)))
    pho = clamp01(float(rng.normal(p.phone_noise_mu, p.phone_noise_sigma)))
    com = clamp01(float(rng.normal(p.comm_freq_mu, p.comm_freq_sigma)))
    car = clamp01(float(rng.normal(p.care_base_mu, p.care_base_sigma)))
    opn = clamp01(float(rng.normal(p.openness_mu, p.openness_sigma)))
    return ext, pho, com, car, opn


# --- Friendly parsing helpers and presets ---------------------------------
WORD_MAP = {
    'low': 0.2,
    'medium': 0.5,
    'med': 0.5,
    'high': 0.8,
    'rare': 0.1,
    'occasional': 0.4,
    'often': 0.75,
    'calm': 0.2,
    'busy': 0.8,
    'fast': 0.5,
    'normal': 1.5,
    'slow': 3.0,
}

PRESETS = {
    'romantic': {
        'comm_freq_mu': 0.8,
        'care_base_mu': 0.9,
        'openness_mu': 0.7,
        'reply_delay_days': 0.5,
    },
    'casual': {
        'comm_freq_mu': 0.45,
        'care_base_mu': 0.35,
        'openness_mu': 0.6,
        'reply_delay_days': 2.0,
    }
}


def _percent_to_float(s: str) -> float:
    s = s.strip()
    if s.endswith('%'):
        return float(s[:-1]) / 100.0
    return float(s)


def parse_human_value(raw: str | None, param_name: str):
    if raw is None:
        return None
    s = str(raw).strip().lower()
    # numeric or percent
    try:
        if s.endswith('%'):
            return _percent_to_float(s)
        return float(s)
    except Exception:
        pass

    # word mapping
    if s in WORD_MAP:
        if param_name in {'reply_delay_days'}:
            return WORD_MAP[s]
        return float(max(0.0, min(1.0, WORD_MAP[s])))

    return None


def describe_params(p: Params) -> str:
    # Return a short, human-friendly description of key params
    lines = [
        f"Reply delay (days): {p.reply_delay_days} (larger = slower replies)",
        f"Communication frequency (mu): {p.comm_freq_mu if hasattr(p, 'comm_freq_mu') else getattr(p, 'comm_freq_mu', 'N/A')}",
        f"Care tendency (mu): {p.care_base_mu if hasattr(p, 'care_base_mu') else getattr(p, 'care_base_mu', 'N/A')}",
        f"Openness (mu): {p.openness_mu if hasattr(p, 'openness_mu') else getattr(p, 'openness_mu', 'N/A')}",
        f"Process noise: {p.process_sigma}",
    ]
    return "\n".join(lines)


def _build_params_from_args(args) -> Params:
    base = dataclasses.asdict(Params())
    overrides = {}
    if getattr(args, 'preset', None):
        ps = PRESETS.get(args.preset.lower())
        if ps:
            overrides.update(ps)
    for name in ('comm_freq_mu', 'care_base_mu', 'openness_mu'):
        raw = getattr(args, name, None)
        if raw is not None:
            v = parse_human_value(raw, name)
            if v is not None:
                overrides[name] = v
    if getattr(args, 'reply_delay_days', None) is not None:
        rd = parse_human_value(args.reply_delay_days, 'reply_delay_days')
        if rd is not None:
            overrides['reply_delay_days'] = rd
    # include N/T if provided
    if getattr(args, 'N', None) is not None:
        overrides['N'] = args.N
    if getattr(args, 'T', None) is not None:
        overrides['T'] = args.T
    return Params(**{**base, **overrides})


def _simulate_one(p: Params, rng: np.random.Generator, init_x: float) -> str:
    ext, pho, com, car, opn = _draw_latents(p, rng)

    att = math.exp(-p.delay_decay_per_day * max(0.0, p.reply_delay_days))
    com *= att
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

        w = 0.7 * ext + 0.3 * pho + float(rng.normal(0, 0.06))
        w = clamp(w, 0.0, 1.0)

        care_prob = p.base_care_prob + 0.25 * car + 0.15 * com + 0.10 * opn + 0.10 * y
        care_prob = clamp(care_prob, 0.0, 0.95)
        care_input = 0.0
        if rng.random() < care_prob:
            care_input = clamp(
                float(rng.normal(p.care_event_mu, p.care_event_sigma)), 0.0, 1.0)

        plan_prob = p.base_plan_prob + 0.20 * com + 0.15 * y + 0.10 * opn
        if y > p.schedule_threshold:
            plan_prob += 0.10
        plan_prob = clamp(plan_prob, 0.0, 0.98)
        if rng.random() < plan_prob:
            scheduled_count += 1
            com = clamp(com + float(rng.normal(0.05, 0.03)), 0.0, 1.0)
            car = clamp(car + float(rng.normal(0.04, 0.03)), 0.0, 1.0)

        x_next = p.a * y + p.b * u + p.c * care_input - \
            p.d * w + float(rng.normal(0, p.process_sigma))
        x_next = clamp(x_next, 0.0, 1.0)

        if t <= 2 and x_next < p.abrupt_early_cutoff:
            abrupt_flag = True

        y_prev = y
        x = x_next

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
    labels = ["defined_relationship", "stable_casual",
              "gradual_fade", "abrupt_disengagement"]
    counts = {k: 0 for k in labels}
    for _ in range(p.N):
        outcome = _simulate_one(p, rng, init_x=init_x)
        counts[outcome] += 1
    total = float(p.N)
    return {k: round(100.0 * v / total, 2) for k, v in counts.items()}


def cmd_run(args):
    # build Params with optional preset and friendly overrides
    base = dataclasses.asdict(Params())
    overrides = {}
    if getattr(args, 'preset', None):
        ps = PRESETS.get(args.preset.lower())
        if ps:
            overrides.update(ps)

    # friendly parsing for certain fields
    for name in ('comm_freq_mu', 'care_base_mu', 'openness_mu'):
        raw = getattr(args, name, None)
        if raw is not None:
            v = parse_human_value(raw, name)
            if v is not None:
                overrides[name] = v

    # reply delay special handling
    if getattr(args, 'reply_delay_days', None) is not None:
        rd = parse_human_value(args.reply_delay_days, 'reply_delay_days')
        if rd is not None:
            overrides['reply_delay_days'] = rd

    # numeric overrides
    overrides['N'] = args.N
    overrides['T'] = args.T
    overrides['delay_decay_per_day'] = args.delay_decay_per_day

    p = Params(**{**base, **overrides})

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
        # allow presets/overrides like in run
        base = dataclasses.asdict(Params())
        overrides = {}
        if getattr(args, 'preset', None):
            ps = PRESETS.get(args.preset.lower())
            if ps:
                overrides.update(ps)
        overrides['reply_delay_days'] = dd
        overrides['N'] = args.N
        overrides['T'] = args.T
        overrides['delay_decay_per_day'] = args.delay_decay_per_day

        p = Params(**{**base, **overrides})
        res = monte_carlo(p, seed=args.seed, init_x=args.init_x)
        row = {"delay_days": dd, **res}
        rows.append(row)

    out_csv = args.out_csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_csv}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available; skipping plot.", file=sys.stderr)
            return

        x = [r["delay_days"] for r in rows]
        def_prob = [r["defined_relationship"] for r in rows]
        stable_prob = [r["stable_casual"] for r in rows]
        fade_prob = [r["gradual_fade"] for r in rows]
        abrupt_prob = [r["abrupt_disengagement"] for r in rows]
        # (plotting handled by caller if requested)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monte Carlo dating model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a single monte-carlo experiment")
    p_run.add_argument("--seed", type=int, default=1)
    p_run.add_argument("--init_x", type=float, default=0.5)
    p_run.add_argument("--N", type=int, default=10000)
    p_run.add_argument("--T", type=int, default=8)
    p_run.add_argument("--reply_delay_days", type=str, default=None,
                       help="reply delay in days (float) or friendly words like fast/slow or '1.5'")
    p_run.add_argument("--delay_decay_per_day", type=float, default=0.25)
    p_run.add_argument("--preset", type=str, default=None,
                       help="preset name (romantic, casual)")
    p_run.add_argument("--comm_freq_mu", type=str, default=None,
                       help="communication frequency (0..1, 0.5, 50%, or words 'rare/occasional/often')")
    p_run.add_argument("--care_base_mu", type=str, default=None,
                       help="care tendency (0..1, percent, or words)")
    p_run.add_argument("--openness_mu", type=str, default=None,
                       help="openness (0..1, percent, or words)")
    p_run.set_defaults(func=cmd_run)

    p_sweep = sub.add_parser("sweep", help="Sweep reply delay and save CSV")
    p_sweep.add_argument("--min", type=float, default=0.0)
    p_sweep.add_argument("--max", type=float, default=3.0)
    p_sweep.add_argument("--step", type=float, default=0.1)
    p_sweep.add_argument("--out_csv", type=str, default="sweep.csv")
    p_sweep.add_argument("--plot", action="store_true")
    p_sweep.add_argument("--seed", type=int, default=1)
    p_sweep.add_argument("--init_x", type=float, default=0.5)
    p_sweep.add_argument("--N", type=int, default=10000)
    p_sweep.add_argument("--T", type=int, default=8)
    p_sweep.add_argument("--reply_delay_days", type=str, default=None,
                         help="reply delay in days; accepts friendly values as in run")
    p_sweep.add_argument("--delay_decay_per_day", type=float, default=0.25)
    p_sweep.set_defaults(func=cmd_sweep)

    p_desc = sub.add_parser(
        "describe", help="Print a human-friendly summary of params")
    p_desc.add_argument("--preset", type=str, default=None,
                        help="apply preset before describing")
    p_desc.add_argument("--comm_freq_mu", type=str, default=None)
    p_desc.add_argument("--care_base_mu", type=str, default=None)
    p_desc.add_argument("--openness_mu", type=str, default=None)
    p_desc.add_argument("--reply_delay_days", type=str, default=None)
    p_desc.set_defaults(func=lambda args: print(
        describe_params(_build_params_from_args(args))))

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
