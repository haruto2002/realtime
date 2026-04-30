import os
import time
from pathlib import Path

import cv2
import numpy as np

from processor.components import Detector, Tracker
from processor.worker import MotWorker


def run_image(
    image: np.ndarray, detectors: list[Detector], trackers: list[Tracker]
) -> np.ndarray:
    start_ts = time.perf_counter()
    frame, detected_ts, tracked_ts, drawn_ts = MotWorker.process_frame(
        image, detectors, trackers
    )
    end_ts = time.perf_counter()
    print(
        f"total: {ms(end_ts - start_ts)}, detect: {ms(detected_ts - start_ts)}, track: {ms(tracked_ts - detected_ts)}, drawn: {ms(drawn_ts - tracked_ts)}, write: {ms(end_ts - drawn_ts)}"
    )
    return frame


def run_video(
    path2video: str,
    detectors: list[Detector],
    trackers: list[Tracker],
    output_path: str,
    fps: int,
    scale: float,
):
    video_capture = cv2.VideoCapture(path2video)
    raw_fps = video_capture.get(cv2.CAP_PROP_FPS)

    step = int(raw_fps / fps)
    output_fps = raw_fps / step

    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

    print(f"fps: {output_fps}, width: {width}, height: {height}")

    video_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
    )

    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Video finished")
            break

        if frame_count % step == 0:
            frame = cv2.resize(frame, (width, height))
            frame = run_image(frame, detectors, trackers)
            video_writer.write(frame)

        frame_count += 1

    video_capture.release()
    video_writer.release()


def ms(time: float) -> str:
    return f"{time * 1000:.2f} ms"


def main():
    path2detector_cfg = [
        "pipeline/config/detector/p2pnet/p2pnet.yaml",
        "pipeline/config/detector/yolo26/yolo26.yaml",
    ]
    path2tracker_cfg = [
        "pipeline/config/tracker/point_bytetrack/point_bytetrack.yaml",
        "pipeline/config/tracker/bytetrack/bytetrack.yaml",
    ]

    detectors = [
        Detector(Path(path2detector_cfg)) for path2detector_cfg in path2detector_cfg
    ]
    print("Detector Setup Done")
    trackers = [
        Tracker(Path(path2tracker_cfg)) for path2tracker_cfg in path2tracker_cfg
    ]
    print("Tracker Setup Done")

    names = ["pasifico_a", "pasifico_b"]

    for name in names:
        path2video = f"samples/vid/videos/{name}.mp4"
        output_path = f"vis/videos/{name}.mp4"
        assert os.path.exists(path2video), f"Video file not found: {path2video}"

        fps = 5
        scale = 0.25
        run_video(path2video, detectors, trackers, output_path, fps, scale)


if __name__ == "__main__":
    main()
