import os
import time
from pathlib import Path

import cv2

from processor.components import Detector


def main(
    path2detector_cfg: str,
    path2image: str,
    output_path: str,
) -> None:

    detector = Detector(Path(path2detector_cfg))
    print("Detector Setup Done")

    pipeline(path2image, detector, output_path)


def pipeline(
    path2image: str,
    detector: Detector,
    output_path: str,
):
    img = cv2.imread(path2image)
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    start_ts = time.perf_counter()
    dets = detector.infer(img)
    detected_ts = time.perf_counter()
    img = detector.draw(img, dets)
    drawn_ts = time.perf_counter()
    cv2.imwrite(output_path, img)
    end_ts = time.perf_counter()
    print(
        f"total: {ms(end_ts - start_ts)}, detect: {ms(detected_ts - start_ts)}, drawn: {ms(drawn_ts - detected_ts)}, write: {ms(end_ts - drawn_ts)}"
    )


def ms(time: float) -> str:
    return f"{time * 1000:.2f} ms"


if __name__ == "__main__":
    path2detector_cfg = "img_conf/detector/yolo26/yolo26.yaml"
    path2image = "samples/img/images/WorldPorters_2023-07-31_17:51:20_DSC_5983_0000.jpg"
    save_dir = "vis/img/exp0512"
    output_path = f"{save_dir}/hanabi_yolo26.jpg"
    os.makedirs(save_dir, exist_ok=True)
    main(path2detector_cfg, path2image, output_path)
