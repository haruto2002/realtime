"""複数 detector 設定で推論時間を計測する。

同一プロセスで連続して Detector（DEIM）を構築すると engine のグローバル状態が壊れることがあるため、
設定ごとにサブプロセスを分ける（debug.py と同じ考え方）。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from processor.components import Detector

_REPO_ROOT = Path(__file__).resolve().parent
_THIS = Path(__file__).resolve()


def count_time(cfg_dir: Path, image: np.ndarray, *, quiet: bool = False) -> float:
    detector = Detector(cfg_dir)

    num_frames = 30
    it = range(num_frames)
    if not quiet:
        it = tqdm(it, leave=False)
    warmup_frame = 5
    for i, _ in enumerate(it):
        if i == warmup_frame - 1:
            start_time = time.perf_counter()
        _ = detector.infer(image)
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / (num_frames - warmup_frame)
    if not quiet:
        print(f"Average time taken: {avg_time * 1000} ms")
    return avg_time * 1000


def _load_image() -> np.ndarray:
    image = cv2.imread(str(_REPO_ROOT / "hd_demo.jpg"))
    image = cv2.resize(image, (1920, 1080))
    return image


def run_one(det_cfg: Path) -> None:
    """子プロセス用: 最後の stdout の 1 行だけが数値（ms）。"""
    image = _load_image()
    avg_ms = count_time(det_cfg, image, quiet=True)
    print(avg_ms, flush=True)


def main() -> None:
    dir_path = _REPO_ROOT / "experiments" / "results"
    det_cfgs = sorted(dir_path.glob("*cpu*/conf/detector.yaml"))
    log_dict: dict[str, float] = {}

    for det_cfg in tqdm(det_cfgs, desc="configs"):
        key = str(det_cfg.relative_to(_REPO_ROOT))
        print(f"Processing {key}")
        r = subprocess.run(
            [sys.executable, str(_THIS), "--one", str(det_cfg.resolve())],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            raise SystemExit(r.returncode)
        lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
        if not lines:
            raise SystemExit(f"No stdout from subprocess for {key}")
        log_dict[key] = float(lines[-1])

    out_path = _REPO_ROOT / "time_counter.json"
    out_path.write_text(json.dumps(log_dict, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        run_one(Path(sys.argv[2]))
    else:
        main()
