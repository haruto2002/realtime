"""time_counter.json の計測結果を可視化する。

- 全実験: 横棒グラフ + 表
- モデル別: 各モデル内でタイル解像度（H×W）ごとの ms を並べたグループ棒グラフ

実験 ID は `deimv2_{cpu|cuda0}_{Model}_{H}x{W}` 形式を想定。

例:
  uv run python time_vis.py
  uv run python time_vis.py path/to/time_counter.json -o out.png
  → out.png（全実験）, out_by_model.png（モデル×解像度）
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent

# set_yaml の model_sizes と同じ並び（グラフの左→右）
MODEL_ORDER = ["Atto", "Femto", "Pico", "N", "S", "M", "L", "X"]

_EXPERIMENT_RE = re.compile(
    r"^deimv2_(?P<dev>cpu|cuda0)_(?P<model>.+)_(?P<res>\d+x\d+)$"
)


def short_label(path_key: str) -> str:
    """experiments/results/<実験ID>/conf/detector.yaml → 実験ID。"""
    if "experiments/results/" in path_key:
        try:
            rest = path_key.split("experiments/results/", 1)[1]
            return rest.split("/conf/", 1)[0]
        except IndexError:
            pass
    return path_key


def parse_experiment_id(exp_id: str) -> tuple[str, str, str] | None:
    """deimv2_cpu_Atto_1080x1920 → (cpu, Atto, 1080x1920)。非対応形式は None。"""
    m = _EXPERIMENT_RE.match(exp_id)
    if not m:
        return None
    return m.group("dev"), m.group("model"), m.group("res")


def load_ms_pairs(json_path: Path) -> list[tuple[str, float]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object {path: ms, ...}")
    out: list[tuple[str, float]] = []
    for k, v in data.items():
        if isinstance(v, (int, float)):
            out.append((short_label(str(k)), float(v)))
    return out


def print_table(items: list[tuple[str, float]]) -> None:
    items_sorted = sorted(items, key=lambda x: x[1])
    print(f"\n{'experiment':<42} {'ms/frame':>12} {'FPS':>10}")
    print("-" * 66)
    for label, ms in items_sorted:
        fps = 1000.0 / ms if ms > 0 else 0.0
        print(f"{label:<42} {ms:>12.2f} {fps:>10.2f}")


def _mean_stdev(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def _model_sort_key(model: str) -> int:
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return 999


def _res_sort_key(res: str) -> tuple[int, int]:
    try:
        h, w = res.split("x", 1)
        return int(float(h)), int(float(w))
    except (ValueError, IndexError):
        return (0, 0)


def _print_model_resolution_table(
    nested: dict[str, dict[str, list[float]]],
    models: list[str],
    resolutions: list[str],
) -> None:
    print("\n=== By model × tile resolution (mean ms/frame; n runs) ===")
    w = 8
    head = f"{'model':<10}" + "".join(f"{r:^{w * 2}}" for r in resolutions)
    print(head)
    print("-" * len(head))
    for m in models:
        parts = [f"{m:<10}"]
        for res in resolutions:
            vals = nested[m].get(res, [])
            if not vals:
                parts.append(f"{'—':>{w}}")  # ms
                parts.append(f"{'—':>{w}}")  # FPS
            else:
                mu, _ = _mean_stdev(vals)
                fps = 1000.0 / mu if mu > 0 else 0.0
                n = len(vals)
                cell_ms = f"{mu:.1f}" if n == 1 else f"{mu:.1f}({n})"
                cell_fps = f"{fps:.1f}" if n == 1 else f"{fps:.1f}({n})"
                data = f"{cell_ms} ({cell_fps})"
                # data = f"{cell_ms:>{w}}{cell_fps:>{w}}"
                parts.append(f"{data:^{w * 2}}")
        # Header row with double columns? Need to adapt.
        print("".join(parts))


def save_bar_chart(
    items: list[tuple[str, float]],
    out_path: Path,
    title: str,
) -> None:
    items_sorted = sorted(items, key=lambda x: x[1])
    labels = [x[0] for x in items_sorted]
    vals = [x[1] for x in items_sorted]

    fig_h = max(5.0, 0.32 * len(labels))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    y = range(len(labels))
    ax.barh(y, vals, color="#4a90d9", edgecolor="white", linewidth=0.4, height=0.75)
    ax.set_yticks(list(y), labels, fontsize=8)
    ax.set_xlabel("ms / frame (lower is better)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    xmax = max(vals) * 1.18 if vals else 1.0
    ax.set_xlim(0, xmax)
    pad = max(vals) * 0.012 if vals else 0.5
    for i, v in enumerate(vals):
        ax.text(v + pad, i, f"{v:.1f}", va="center", fontsize=7, color="#333")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_by_model_chart(
    items: list[tuple[str, float]],
    out_path: Path,
    json_name: str,
) -> None:
    """モデルごとに、解像度別の棒を横に並べる（グループ棒グラフ）。"""
    nested: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for lab, ms in items:
        p = parse_experiment_id(lab)
        if p is None:
            continue
        _, model, res = p
        nested[model][res].append(ms)

    if not nested:
        print("[warn] No rows matched deimv2_*_*_*x*; skip by-model chart.")
        return

    models = sorted(nested.keys(), key=_model_sort_key)
    resolutions = sorted(
        {res for m in nested for res in nested[m]},
        key=_res_sort_key,
    )
    n_m, n_r = len(models), len(resolutions)
    if n_r == 0:
        return

    _print_model_resolution_table(nested, models, resolutions)

    x = np.arange(n_m, dtype=float)
    width = min(0.22, 0.82 / max(n_r, 1))
    fig_w = max(11.0, 1.2 * n_m + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))
    cmap = plt.cm.tab10(np.linspace(0, 0.92, max(n_r, 1)))

    ymax = 1.0
    for ri, res in enumerate(resolutions):
        means: list[float] = []
        errs: list[float] = []
        for m in models:
            vals = nested[m].get(res, [])
            if vals:
                mu, sd = _mean_stdev(vals)
                means.append(mu)
                errs.append(sd)
                ymax = max(ymax, mu + sd)
            else:
                means.append(0.0)
                errs.append(0.0)
        offset = (ri - (n_r - 1) / 2.0) * width
        use_yerr = any(e > 0 for e in errs)
        ax.bar(
            x + offset,
            means,
            width,
            yerr=errs if use_yerr else None,
            capsize=2.5 if use_yerr else 0,
            label=res,
            color=cmap[ri],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x, models)
    ax.set_xlabel("Model size")
    ax.set_ylabel("ms / frame (lower is better)")
    ax.set_title(f"Inference time by model & tile resolution — {json_name}")
    ax.legend(title="Tile H×W", fontsize=9, ncol=min(4, n_r), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(0, ymax * 1.12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize time_counter.json")
    p.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        default=_REPO / "time_counter.json",
        help="計測 JSON（既定: リポジトリ直下の time_counter.json）",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="メインの横棒 PNG（省略時は json と同じ stem の .png）",
    )
    args = p.parse_args()

    json_path: Path = args.json_path.resolve()
    if not json_path.is_file():
        raise SystemExit(f"Not found: {json_path}")

    items = load_ms_pairs(json_path)
    if not items:
        raise SystemExit("No numeric entries in JSON.")

    out_png = args.out if args.out is not None else json_path.with_suffix(".png")
    stem = out_png.stem
    parent = out_png.parent
    by_model_png = parent / f"{stem}_by_model.png"
    json_name = json_path.name

    print(f"Loaded {len(items)} entries from {json_path}")
    print_table(items)
    save_bar_chart(items, out_png, title=f"Inference time — {json_name}")
    print(f"\nChart (all runs): {out_png}")

    unparsed = sum(1 for lab, _ in items if parse_experiment_id(lab) is None)
    if unparsed:
        print(
            f"[info] {unparsed} experiment id(s) did not match deimv2_*_*_*x* (by-model chart skips them)."
        )

    save_by_model_chart(items, by_model_png, json_name)
    print(f"Chart (by model × resolution): {by_model_png}")


if __name__ == "__main__":
    main()
