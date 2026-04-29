import argparse
import time
from pathlib import Path

import cv2

from processor.components import Detector


def main(path2detector_cfg: str, path2image: str, output_path: str) -> None:

    detector = Detector(Path(path2detector_cfg))
    print("Detector Setup Done")

    pipeline(path2image, detector, output_path)


def pipeline(path2image: str, detector: Detector, output_path: str):
    img = cv2.imread(path2image)
    start_ts = time.perf_counter()
    dets = detector.infer(img)
    detected_ts = time.perf_counter()
    frame = detector.draw(img, dets)
    drawn_ts = time.perf_counter()
    cv2.imwrite(output_path, frame)
    end_ts = time.perf_counter()
    print(
        f"total: {ms(end_ts - start_ts)}, detect: {ms(detected_ts - start_ts)}, drawn: {ms(drawn_ts - detected_ts)}, write: {ms(end_ts - drawn_ts)}"
    )


def ms(time: float) -> str:
    return f"{time * 1000:.2f} ms"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path2detector_cfg", type=str, default="img_conf/detector.yaml"
    )
    parser.add_argument(
        "--path2image",
        type=str,
        default="hd_demo.jpg",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="hd_demo_output.jpg",
    )
    return parser.parse_args()


if __name__ == "__main__":
    path2detector_cfg = "pipeline/config/detector/p2pnet/p2pnet.yaml"
    path2image = "hd_demo.jpg"
    output_path = "hd_demo_output.jpg"
    main(path2detector_cfg, path2image, output_path)
