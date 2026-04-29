import time
from pathlib import Path

import cv2
import numpy as np
import torch

from processor.components import Detector


def main(
    path2point_detector_cfg: str,
    path2box_detector_cfg: str,
    path2image: str,
    output_path: str,
) -> None:

    point_detector = Detector(Path(path2point_detector_cfg))
    box_detector = Detector(Path(path2box_detector_cfg))
    print("Detectors Setup Done")

    img = cv2.imread(path2image)
    img = cv2.resize(img, (1920, 1080))

    parallel = False

    max_frames = 10
    frame_count = 0
    while frame_count < max_frames:
        frame_count += 1
        if parallel:
            stream_a = torch.cuda.Stream()
            stream_b = torch.cuda.Stream()
            parallel_processing(
                img,
                point_detector,
                box_detector,
                output_path,
                stream_a,
                stream_b,
            )
        else:
            sequential_processing(img, point_detector, box_detector, output_path)


def sequential_processing(
    img: np.ndarray, point_detector: Detector, box_detector: Detector, output_path: str
):
    start_ts = time.perf_counter()
    point_dets = point_detector.infer(img)
    box_dets = box_detector.infer(img)
    detected_ts = time.perf_counter()
    frame = point_detector.draw(img, point_dets)
    frame = box_detector.draw(frame, box_dets)
    drawn_ts = time.perf_counter()
    cv2.imwrite(output_path, frame)
    end_ts = time.perf_counter()
    print(
        f"total: {ms(end_ts - start_ts)}, detect: {ms(detected_ts - start_ts)}, drawn: {ms(drawn_ts - detected_ts)}, write: {ms(end_ts - drawn_ts)}"
    )


def parallel_processing(
    img: np.ndarray,
    point_detector: Detector,
    box_detector: Detector,
    output_path: str,
    stream_a: torch.cuda.Stream,
    stream_b: torch.cuda.Stream,
):
    start_ts = time.perf_counter()

    with torch.no_grad():
        with torch.cuda.stream(stream_a):
            point_dets = point_detector.infer(img)

        with torch.cuda.stream(stream_b):
            box_dets = box_detector.infer(img)

        # 両方のstreamの処理完了を待つ
        stream_a.synchronize()
        stream_b.synchronize()

    detected_ts = time.perf_counter()

    frame = point_detector.draw(img, point_dets)
    frame = box_detector.draw(img, box_dets)

    drawn_ts = time.perf_counter()

    cv2.imwrite(output_path, frame)

    end_ts = time.perf_counter()

    print(
        f"total: {ms(end_ts - start_ts)}, "
        f"detect: {ms(detected_ts - start_ts)}, "
        f"drawn: {ms(drawn_ts - detected_ts)}, "
        f"write: {ms(end_ts - drawn_ts)}"
    )


def ms(time: float) -> str:
    return f"{time * 1000:.2f} ms"


if __name__ == "__main__":
    path2point_detector_cfg = "img_conf/detector/p2pnet/p2pnet.yaml"
    path2box_detector_cfg = "img_conf/detector/yolo26/yolo26.yaml"
    path2image = "hd_demo.jpg"
    output_path = "hd_demo_output.jpg"
    main(path2point_detector_cfg, path2box_detector_cfg, path2image, output_path)
