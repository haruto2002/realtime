"""実験グリッドごとに results/<id>/ 以下へ Hydra 用 YAML を生成する。

プロジェクトルートで次のように実行する想定:
  uv run python experiments/set_yaml.py
全実験の run コマンドは `results/run_commands.txt` に 1 行ずつまとめる。
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

from omegaconf import OmegaConf

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
TEMPLATE_CONF = REPO_ROOT / "processor" / "conf"
BASE_DEIM_YAML = (
    REPO_ROOT / "pipeline" / "config" / "detector" / "deimv2" / "deimv2.yaml"
)
BASE_TRACK_YAML = (
    REPO_ROOT / "pipeline" / "config" / "tracker" / "bytetrack" / "bytetrack.yaml"
)
RESULTS_DIR = EXPERIMENTS_DIR / "results"
AGGREGATE_RUN_COMMANDS = "run_commands.txt"

detector_resolutions = [
    (1080, 1920),
    (1080, 1920 / 2),
    (1080 / 2, 1920 / 2),
    (1080 / 2, 1920 / 3),
]  # (height, width) → Detector.devide_size

devices = ["cpu", "cuda:0"]

model_sizes = ["Atto", "Femto", "Pico", "N", "S", "M", "L", "X"]


def _device_slug(device: str) -> str:
    return device.replace(":", "")


def experiment_id(device: str, model_size: str, height: float, width: float) -> str:
    h, w = int(height), int(width)
    return f"deimv2_{_device_slug(device)}_{model_size}_{h}x{w}"


def _rel_to_repo(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def run_command_for_cfg_dir(cfg_dir_rel: str) -> str:
    """リポジトリルートを cwd としたときにそのまま使える 1 行コマンド。"""
    return f"uv run python run.py --cfg_dir {cfg_dir_rel}"


def write_aggregate_run_commands() -> None:
    """`results/` 直下に、各実験の `run.py --cfg_dir ...` を 1 行ずつ並べたファイルを書く。"""
    lines: list[str] = []
    for exp_root in sorted(RESULTS_DIR.iterdir(), key=lambda p: p.name):
        if not exp_root.is_dir():
            continue
        exp_conf = exp_root / "conf"
        if not (exp_conf / "timer.yaml").is_file():
            continue
        lines.append(run_command_for_cfg_dir(_rel_to_repo(exp_conf)))
    out = RESULTS_DIR / AGGREGATE_RUN_COMMANDS
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(
        f"Wrote {len(lines)} command(s) → {_rel_to_repo(out)}",
    )


def write_deim_module(exp_conf: Path, device: str, model_size: str) -> Path:
    module_dir = exp_conf / "module"
    module_dir.mkdir(parents=True, exist_ok=True)
    out = module_dir / "deimv2.yaml"
    cfg = OmegaConf.load(BASE_DEIM_YAML)
    cfg.model_size = model_size
    cfg.device = device
    OmegaConf.save(cfg, out)
    return out


def write_tracker_module(exp_conf: Path) -> Path:
    module_dir = exp_conf / "module"
    module_dir.mkdir(parents=True, exist_ok=True)
    out = module_dir / "bytetrack.yaml"
    cfg = OmegaConf.load(BASE_TRACK_YAML)
    OmegaConf.save(cfg, out)
    return out


def write_processor_flat_yaml(
    exp_root: Path,
    exp_conf: Path,
    deim_yaml: Path,
    track_yaml: Path,
    divide_h: float,
    divide_w: float,
) -> None:
    """run.py が期待する conf/ 直下の YAML を書く。"""
    exp_conf.mkdir(parents=True, exist_ok=True)

    timer = OmegaConf.load(TEMPLATE_CONF / "timer.yaml")
    # TimeCounter は save_dir 配下に time_counter.json / summary.txt を保存
    timer.save_dir = _rel_to_repo(exp_root)
    OmegaConf.save(timer, exp_conf / "timer.yaml")

    for name in ("reader.yaml", "displayer.yaml"):
        src = TEMPLATE_CONF / name
        dst = exp_conf / name
        shutil.copyfile(src, dst)

    tracker = OmegaConf.create(
        {
            "_target_": "processor.components.Tracker",
            "cfg_path": _rel_to_repo(track_yaml),
        }
    )
    OmegaConf.save(tracker, exp_conf / "tracker.yaml")

    detector = OmegaConf.create(
        {
            "_target_": "processor.components.Detector",
            "cfg_path": _rel_to_repo(deim_yaml),
            "devide_size": [int(divide_h), int(divide_w)],
        }
    )
    OmegaConf.save(detector, exp_conf / "detector.yaml")


def materialize_one(
    device: str,
    model_size: str,
    height: float,
    width: float,
    *,
    force: bool,
    dry_run: bool,
) -> Path | None:
    exp_id = experiment_id(device, model_size, height, width)
    exp_root = RESULTS_DIR / exp_id
    exp_conf = exp_root / "conf"

    if dry_run:
        suffix = " (would remove first)" if exp_root.exists() and force else ""
        cfg_rel = _rel_to_repo(exp_conf)
        print(f"[dry-run] would write: {exp_conf}{suffix}")
        print(f"        + cmd → {run_command_for_cfg_dir(cfg_rel)}")
        return exp_conf

    if exp_root.exists() and not force:
        print(f"[skip] exists: {exp_id}")
        return None

    if exp_root.exists() and force:
        shutil.rmtree(exp_root)

    exp_root.mkdir(parents=True, exist_ok=True)
    deim_path = write_deim_module(exp_conf, device, model_size)
    track_path = write_tracker_module(exp_conf)
    write_processor_flat_yaml(
        exp_root, exp_conf, deim_path, track_path, height, width
    )
    print(f"[ok] {exp_id}  ({run_command_for_cfg_dir(_rel_to_repo(exp_conf))})")
    return exp_conf


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment YAML trees.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths only; do not write files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing experiment directory before writing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Generate at most N combinations (order: device → model_size → resolution).",
    )
    args = parser.parse_args()

    if not BASE_DEIM_YAML.is_file():
        raise FileNotFoundError(f"Missing base DEIM config: {BASE_DEIM_YAML}")
    if not BASE_TRACK_YAML.is_file():
        raise FileNotFoundError(f"Missing base tracker config: {BASE_TRACK_YAML}")
    if not TEMPLATE_CONF.is_dir():
        raise FileNotFoundError(f"Missing template conf: {TEMPLATE_CONF}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    combos = itertools.product(devices, model_sizes, detector_resolutions)
    if args.limit is not None:
        combos = itertools.islice(combos, args.limit)

    n = 0
    for device, model_size, (height, width) in combos:
        out = materialize_one(
            device,
            model_size,
            height,
            width,
            force=args.force,
            dry_run=args.dry_run,
        )
        if out is not None or args.dry_run:
            n += 1

    suffix = f" (limited to {args.limit})" if args.limit is not None else ""
    print(f"Done. ({n} experiment(s){suffix})")
    write_aggregate_run_commands()


if __name__ == "__main__":
    main()
