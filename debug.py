"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

同一プロセスで複数サイズの DEIM を連続構築すると、DEIM 付属 engine の
グローバル設定が壊れることがある。本体は触らず、本スクリプトだけで
モデルごとにサブプロセスを分けて回避する。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pipeline.detector.deimv2 import DEIMv2Detector

MAP_MODEL_SIZE_TO_FILE_NAME = {
    "Atto": "deimv2_hgnetv2_atto_coco",
    "Femto": "deimv2_hgnetv2_femto_coco",
    "Pico": "deimv2_hgnetv2_pico_coco",
    "N": "deimv2_hgnetv2_n_coco",
    "S": "deimv2_dinov3_s_coco",
    "M": "deimv2_dinov3_m_coco",
    "L": "deimv2_dinov3_l_coco",
    "X": "deimv2_dinov3_x_coco",
}

_REPO_ROOT = Path(__file__).resolve().parent
_THIS = Path(__file__).resolve()


def detect(model_size: str) -> None:
    cfg_dir = "modules/detector/DEIMv2/configs/deimv2"
    weight_dir = "weights/deimv2"
    device = "cpu"
    threshold = 0.5
    DEIMv2Detector(model_size, cfg_dir, weight_dir, device, threshold)
    # image = cv2.imread("modules/detector/DEIMv2/example.jpg")
    # result = detector.infer(image)
    # image = detector.display_result(image, result)
    # cv2.imwrite("result.jpg", image)


def main_all_via_subprocess() -> None:
    """DEIM engine を汚さないよう、サイズごとに新しい Python で detect する。"""
    for model_size in MAP_MODEL_SIZE_TO_FILE_NAME:
        print(f"Processing {model_size}")
        r = subprocess.run(
            [sys.executable, str(_THIS), "--one", model_size],
            cwd=str(_REPO_ROOT),
            check=False,
        )
        if r.returncode != 0:
            raise SystemExit(r.returncode)
        print(f"Processed {model_size}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        detect(sys.argv[2])
    else:
        main_all_via_subprocess()
