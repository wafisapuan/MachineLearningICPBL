"""
Step 1: Exploration plots for the robot pick-and-place dataset.

Usage:
    python scripts/explore.py --data data/raw

Expects the folder to contain success/ and fail/ subfolders of episode_XXXXXX.csv.
Saves three plots and a starter feature table into ./plots/
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SUCCESS_COLOR, FAIL_COLOR = "#2a78d6", "#eb6834"
ARM = ["l_sh_p", "l_sh_r", "l_sh_y", "l_el_p", "l_wr_y", "l_wr_p", "l_wr_r"]  # left arm does the task
FINGERS = ["l_hand_index", "l_hand_middle", "l_hand_ring", "l_hand_pinky"]
GRIP_CLOSED = 0.8  # average finger angle (rad) above which we call the hand "closed"

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6"})


def load(data_dir):
    """Return {'success': {episode_id: DataFrame}, 'fail': {...}}."""
    eps = {}
    for label in ["success", "fail"]:
        files = sorted(glob.glob(os.path.join(data_dir, label, "*.csv")))
        eps[label] = {os.path.basename(f)[8:14]: pd.read_csv(f) for f in files}
    return eps


def grip(d, prefix="state."):
    return d[[prefix + c for c in FINGERS]].mean(axis=1)


def plot_single_episode(d, title, out):
    """Plot 1: what ONE episode looks like over time."""
    t = d["timestamp"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for c in ARM:
        ax[0].plot(t, d["state." + c], lw=1.8, label=c)
    ax[0].set_title(f"{title}: left-arm joints over time")
    ax[0].set_ylabel("joint angle (rad)")
    ax[0].legend(ncol=7, fontsize=8, frameon=False)
    ax[1].plot(t, grip(d), lw=2, color=SUCCESS_COLOR, label="measured (state)")
    ax[1].plot(t, grip(d, "action."), lw=1.5, ls="--", color=FAIL_COLOR, label="commanded (action)")
    ax[1].set_title("Left-hand finger closure: high = gripping the bottle")
    ax[1].set_xlabel("time (s)")
    ax[1].legend(frameon=False)
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def plot_overlay(eps, out):
    """Plot 2: all episodes on top of each other, time stretched to 0..1."""
    joints = ["l_sh_p", "l_el_p", "l_wr_y", "l_wr_r", "grip", "l_hand_thumb_p"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for a, c in zip(ax.ravel(), joints):
        for label, color in [("success", SUCCESS_COLOR), ("fail", FAIL_COLOR)]:
            for d in eps[label].values():
                y = grip(d) if c == "grip" else d["state." + c]
                a.plot(np.linspace(0, 1, len(d)), y, color=color, lw=0.8, alpha=0.45)
        a.set_title(c); a.set_xlabel("normalized time")
    fig.legend([Line2D([], [], color=SUCCESS_COLOR), Line2D([], [], color=FAIL_COLOR)],
               ["success", "fail"], loc="upper right", frameon=False)
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


def starter_features(eps):
    """Turn each episode into ONE row of numbers."""
    rows = []
    for label, group in eps.items():
        for ep, d in group.items():
            t = d["timestamp"].values
            g = grip(d).values
            closed = np.where(g > GRIP_CLOSED)[0]
            arm = d[["state." + c for c in ARM]].values
            speed = np.abs(np.diff(arm, axis=0)).sum(axis=1) * 30  # 30 fps
            rows.append(dict(
                episode=ep, label=label,
                duration=t[-1],
                grasp_time=t[closed[0]] if len(closed) else np.nan,
                hold_time=t[closed[-1]] - t[closed[0]] if len(closed) else np.nan,
                mean_speed=speed.mean(),
                max_speed=speed.max(),
                track_err=np.abs(arm - d[["action." + c for c in ARM]].values).mean(),
            ))
    return pd.DataFrame(rows)


def plot_features(F, out):
    """Plot 3: one box per class for each feature."""
    feats = [c for c in F.columns if c not in ("episode", "label")]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
    rng = np.random.default_rng(0)
    for a, f in zip(ax.ravel(), feats):
        for j, (label, color) in enumerate([("success", SUCCESS_COLOR), ("fail", FAIL_COLOR)]):
            v = F[F.label == label][f].dropna()
            a.boxplot(v, positions=[j], widths=0.5, showfliers=False)
            a.scatter(j + rng.uniform(-0.15, 0.15, len(v)), v, s=20, color=color, zorder=3)
        a.set_xticks([0, 1], ["success", "fail"]); a.set_title(f)
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw")
    p.add_argument("--out", default="results/figures/explore")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    eps = load(args.data)
    print({k: len(v) for k, v in eps.items()}, "episodes loaded")

    first = next(iter(eps["success"]))
    plot_single_episode(eps["success"][first], f"Success episode {first}", f"{args.out}/1_single_episode.png")
    plot_overlay(eps, f"{args.out}/2_overlay_success_vs_fail.png")
    F = starter_features(eps)
    F.to_csv(f"{args.out}/starter_features.csv", index=False)
    plot_features(F, f"{args.out}/3_feature_boxplots.png")
    print(F.groupby("label").median(numeric_only=True).round(3))
