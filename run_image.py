import time
from pathlib import Path

import cv2

from processor.components import Detector, Tracker


def main(
    path2detector_cfg: list[str],
    path2tracker_cfg: list[str],
    path2image: str,
    output_path: str,
) -> None:

    detectors = [
        Detector(Path(path2detector_cfg)) for path2detector_cfg in path2detector_cfg
    ]
    trackers = [
        Tracker(Path(path2tracker_cfg)) for path2tracker_cfg in path2tracker_cfg
    ]
    print("Detector Setup Done")
    print("Tracker Setup Done")

    pipeline(path2image, detectors, trackers, output_path)


def pipeline(
    path2image: str,
    detectors: list[Detector],
    trackers: list[Tracker],
    output_path: str,
):
    img = cv2.imread(path2image)
    start_ts = time.perf_counter()
    dets = [detector.infer(img) for detector in detectors]
    detected_ts = time.perf_counter()
    tracks = [
        tracker.update(tracker.convert_to_tracker_inputs(det))
        for tracker, det in zip(trackers, dets)
    ]
    [tracker.draw(img, track) for tracker, track in zip(trackers, tracks)]
    drawn_ts = time.perf_counter()
    cv2.imwrite(output_path, img)
    end_ts = time.perf_counter()
    print(
        f"total: {ms(end_ts - start_ts)}, detect: {ms(detected_ts - start_ts)}, drawn: {ms(drawn_ts - detected_ts)}, write: {ms(end_ts - drawn_ts)}"
    )


def ms(time: float) -> str:
    return f"{time * 1000:.2f} ms"


if __name__ == "__main__":
    path2detector_cfg = [
        "pipeline/config/detector/p2pnet/p2pnet.yaml",
        "pipeline/config/detector/yolo26/yolo26.yaml",
    ]
    path2tracker_cfg = [
        "pipeline/config/tracker/point_bytetrack/point_bytetrack.yaml",
        "pipeline/config/tracker/bytetrack/bytetrack.yaml",
    ]
    path2image = "hd_demo.jpg"
    output_path = "hd_demo_output.jpg"
    main(path2detector_cfg, path2tracker_cfg, path2image, output_path)
