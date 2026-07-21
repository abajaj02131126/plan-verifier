"""Task 9 support (no API calls): recall-vs-horizon figure INCLUDING the
Sonnet-5 points, so the paper shows the reversal with equal visibility
(Amendment 1: both Sonnet points must appear alongside Haiku's). Reads only
existing artifacts."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("pivot/results")
s = json.load(open("results/horizon/summary.json"))["pooled_recall_by_horizon"]
H = [5, 10, 20, 40, 80]


def series(name):
    return [s[name].get(str(h)) for h in H]


sonnet = {
    20: json.load(open(OUT / "sonnet_judge_h20.json"))["sonnet5_new"]["llm_judge_zeroshot"]["recall"],
    40: json.load(open(OUT / "sonnet_judge_h40.json"))["sonnet5_zeroshot"]["recall"],
    80: json.load(open(OUT / "sonnet_judge_h80.json"))["sonnet5_zeroshot"]["recall"],
}

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(H, series("hybrid"), "-o", color="#2a6", label="symbolic checker")
ax.plot(H, series("llm_judge_zeroshot"), "--s", color="#c55", label="judge: Haiku zero-shot")
ax.plot(H, series("llm_judge_cot"), "--^", color="#e59", label="judge: Haiku CoT")
sh = sorted(sonnet)
ax.plot(sh, [sonnet[h] for h in sh], ":D", color="#58c", ms=7, label="judge: Sonnet-5 zero-shot")
ax.set_xscale("log")
ax.set_xticks(H)
ax.set_xticklabels([str(h) for h in H])
ax.set_ylim(0.78, 1.02)
ax.set_xlabel("plan horizon (steps)")
ax.set_ylabel("flaw-detection recall")
ax.set_title("Recall vs horizon: Haiku degrades, Sonnet-5 and the checker do not")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(Path("paper/figures") / f"recall_vs_horizon_sonnet.{ext}", dpi=150)
print("wrote paper/figures/recall_vs_horizon_sonnet.{png,svg}")
print("sonnet recall:", sonnet)
